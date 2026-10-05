import re

from unidiff import PatchSet
from unidiff.errors import UnidiffParseError

from pr_agent.algo.git_patch_processing import to_hunk_only_patch as _to_hunk_only_patch
from pr_agent.algo.types import EDIT_TYPE, FilePatchInfo
from pr_agent.log import get_logger

_C_ESCAPE = re.compile(rb'\\([0-3][0-7]{2}|[abfnrtv"\\])')
_C_ESCAPES = {b"a": b"\a", b"b": b"\b", b"f": b"\f", b"n": b"\n", b"r": b"\r", b"t": b"\t", b"v": b"\v"}
_QUOTED_PATH = r'"(?:[^"\\]|\\.)*"'
_GIT_HEADER = re.compile(rf'^diff --git (?P<source>{_QUOTED_PATH}|a/[^"]+?) (?P<target>{_QUOTED_PATH}|b/[^"]+)$')


def _normalize_git_path_headers(diff_text: str) -> str:
    """Keep C-quoted spaces out of unidiff's whitespace-based Git header split."""
    def quote_path(path: str) -> str:
        if path == "/dev/null":
            return path
        if not (path.startswith('"') and path.endswith('"')):
            path = f'"{path}"'
        return path.replace(" ", r"\040")

    lines = diff_text.split("\n")
    normalize = False
    in_hunk = False
    for index, line in enumerate(lines):
        ending = "\r" if line.endswith("\r") else ""
        content = line[:-1] if ending else line
        header = _GIT_HEADER.match(content)
        if header:
            source, target = header.group("source", "target")
            normalize = source.startswith('"') or target.startswith('"')
            in_hunk = False
            if normalize:
                lines[index] = f"diff --git {quote_path(source)} {quote_path(target)}"
        elif content.startswith("diff --git "):
            normalize = False
            in_hunk = False
        elif content.startswith("@@"):
            in_hunk = True
        elif normalize and not in_hunk and content.startswith(("--- ", "+++ ")):
            path, separator, timestamp = content[4:].partition("\t")
            lines[index] = f"{content[:4]}{quote_path(path)}{separator}{timestamp}"
        elif normalize and not in_hunk:
            lines[index] = content
    return "\n".join(lines)


def to_hunk_only_patch(patch_str: str) -> str:
    """Keep the existing provider-layer import path for callers."""
    return _to_hunk_only_patch(patch_str)


def _strip_prefix(path: str | None) -> str | None:
    if path is None:
        return None
    if path.startswith('"') and path.endswith('"'):
        def decode_escape(match: re.Match) -> bytes:
            escape = match.group(1)
            return bytes([int(escape, 8)]) if len(escape) == 3 else _C_ESCAPES.get(escape, escape)

        try:
            path = _C_ESCAPE.sub(decode_escape, path[1:-1].encode("utf-8")).decode("utf-8")
        except UnicodeError:
            return path
    if path.startswith(("a/", "b/")):
        return path[2:]
    return path


def parse_unified_diff(diff_text: str) -> list[FilePatchInfo]:
    """Parse a unified diff into FilePatchInfo objects (patch + metadata only).

    base_file / head_file are left empty here; the provider fills them from the
    working tree and by reverse-applying the patch. Binary files are skipped.
    """
    patch_set = PatchSet(_normalize_git_path_headers(diff_text))
    files: list[FilePatchInfo] = []
    for pf in patch_set:
        if pf.is_binary_file:
            get_logger().info(f"Skipping binary file in diff: {pf.path}")
            continue
        if pf.is_added_file:
            edit_type = EDIT_TYPE.ADDED
        elif pf.is_removed_file:
            edit_type = EDIT_TYPE.DELETED
        elif pf.is_rename:
            edit_type = EDIT_TYPE.RENAMED
        else:
            edit_type = EDIT_TYPE.MODIFIED

        if pf.is_removed_file:  # target is /dev/null: use source path as the name
            filename = _strip_prefix(pf.source_file)
        else:
            filename = _strip_prefix(pf.target_file)
        old_filename = _strip_prefix(pf.source_file) if pf.is_rename else None

        files.append(
            FilePatchInfo(
                base_file="",
                head_file="",
                patch=str(pf),
                filename=filename,
                edit_type=edit_type,
                old_filename=old_filename,
            )
        )
    return files


def reconstruct_base_file(head_file_str: str, patch_str: str) -> str:
    """Reverse-apply a single-file unified diff to head (new) content to recover
    base (original) content. Returns "" if the patch does not cleanly apply."""
    try:
        patch_set = PatchSet(_normalize_git_path_headers(patch_str))
    except UnidiffParseError as e:
        get_logger().info(f"Could not parse patch for base reconstruction: {e}")
        return ""
    if len(patch_set) != 1:
        return ""

    head_lines = head_file_str.splitlines()
    base_lines: list[str] = []
    head_idx = 0  # 0-based cursor into head_lines

    for hunk in patch_set[0]:
        # 1-based -> 0-based. A hunk with no head lines (a pure deletion, as in
        # `git diff -U0`, or a file emptied of all lines) numbers the line *before*
        # the change instead, so its start already is the 0-based insertion point.
        hunk_head_start = hunk.target_start if hunk.target_length == 0 else hunk.target_start - 1
        if hunk_head_start < head_idx or hunk_head_start > len(head_lines):
            return ""  # out-of-order / out-of-bounds hunk
        base_lines.extend(head_lines[head_idx:hunk_head_start])
        head_idx = hunk_head_start

        for line in hunk:
            value = line.value.rstrip("\r\n")
            if line.is_context:
                if head_idx >= len(head_lines) or head_lines[head_idx] != value:
                    return ""
                base_lines.append(head_lines[head_idx])
                head_idx += 1
            elif line.is_added:  # present only in head: verify + skip
                if head_idx >= len(head_lines) or head_lines[head_idx] != value:
                    return ""
                head_idx += 1
            elif line.is_removed:  # present only in base: emit, don't consume head
                base_lines.append(value)

    base_lines.extend(head_lines[head_idx:])
    result = "\n".join(base_lines)
    # Preserve a trailing newline only when the base actually has content; an
    # empty base (e.g. reversing an add-file patch) must stay "" so downstream
    # extend_patch() correctly treats it as having no original file.
    if base_lines and head_file_str.endswith("\n"):
        result += "\n"
    return result
