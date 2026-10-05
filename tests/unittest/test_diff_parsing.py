import os
import subprocess
import sys

import pytest

from pr_agent.algo.types import EDIT_TYPE
from pr_agent.git_providers.diff_parsing import parse_unified_diff, reconstruct_base_file

MODIFY_DIFF = """diff --git a/foo.py b/foo.py
index 1111111..2222222 100644
--- a/foo.py
+++ b/foo.py
@@ -1,3 +1,3 @@
 line1
-line2
+line2-changed
 line3
"""

ADD_DIFF = """diff --git a/new.py b/new.py
new file mode 100644
index 0000000..3333333
--- /dev/null
+++ b/new.py
@@ -0,0 +1,2 @@
+hello
+world
"""

DELETE_DIFF = """diff --git a/gone.py b/gone.py
deleted file mode 100644
index 4444444..0000000
--- a/gone.py
+++ /dev/null
@@ -1,1 +0,0 @@
-bye
"""

RENAME_DIFF = """diff --git a/old.py b/renamed.py
similarity index 100%
rename from old.py
rename to renamed.py
"""


def test_parse_modify():
    files = parse_unified_diff(MODIFY_DIFF)
    assert len(files) == 1
    f = files[0]
    assert f.filename == "foo.py"
    assert f.edit_type == EDIT_TYPE.MODIFIED
    assert f.old_filename is None
    assert "line2-changed" in f.patch


def test_parse_add():
    f = parse_unified_diff(ADD_DIFF)[0]
    assert f.filename == "new.py"
    assert f.edit_type == EDIT_TYPE.ADDED


def test_parse_delete():
    f = parse_unified_diff(DELETE_DIFF)[0]
    assert f.filename == "gone.py"
    assert f.edit_type == EDIT_TYPE.DELETED


def test_parse_rename():
    f = parse_unified_diff(RENAME_DIFF)[0]
    assert f.filename == "renamed.py"
    assert f.edit_type == EDIT_TYPE.RENAMED
    assert f.old_filename == "old.py"


_PATCH = """--- a/foo.py
+++ b/foo.py
@@ -1,3 +1,3 @@
 line1
-line2
+line2-changed
 line3
"""

HEAD = "line1\nline2-changed\nline3\n"
BASE = "line1\nline2\nline3\n"


def test_reconstruct_base_success():
    assert reconstruct_base_file(HEAD, _PATCH) == BASE


def test_reconstruct_base_drift_returns_empty():
    drifted_head = "completely\ndifferent\ncontent\n"
    assert reconstruct_base_file(drifted_head, _PATCH) == ""


# --- new tests for reconstruct_base_file ---

_MULTI_HUNK_PATCH = """--- a/bar.py
+++ b/bar.py
@@ -1,3 +1,3 @@
 alpha
-beta
+beta-new
 gamma
@@ -5,3 +5,3 @@
 delta
-epsilon
+epsilon-new
 zeta
"""

_MULTI_HUNK_HEAD = "alpha\nbeta-new\ngamma\n\ndelta\nepsilon-new\nzeta\n"
_MULTI_HUNK_BASE = "alpha\nbeta\ngamma\n\ndelta\nepsilon\nzeta\n"


def test_reconstruct_base_multi_hunk():
    """Base is correctly reconstructed across two hunks."""
    result = reconstruct_base_file(_MULTI_HUNK_HEAD, _MULTI_HUNK_PATCH)
    assert result == _MULTI_HUNK_BASE


_NO_TRAILING_NL_PATCH = """--- a/noeol.py
+++ b/noeol.py
@@ -1,3 +1,3 @@
 line1
-line2
+line2-changed
 line3
"""

_NO_TRAILING_NL_HEAD = "line1\nline2-changed\nline3"  # no trailing newline
_NO_TRAILING_NL_BASE = "line1\nline2\nline3"  # no trailing newline expected


def test_reconstruct_base_no_trailing_newline():
    """Result has no trailing newline when head has none."""
    result = reconstruct_base_file(_NO_TRAILING_NL_HEAD, _NO_TRAILING_NL_PATCH)
    assert result == _NO_TRAILING_NL_BASE
    assert not result.endswith("\n")


# A diff that added a file (base was empty); reversed: head is the added content,
# base must be exactly "" — never "\n" — so downstream extend_patch() correctly
# treats the original file as non-existent.
_ADD_FILE_PATCH = """--- /dev/null
+++ b/new.py
@@ -0,0 +1,2 @@
+hello
+world
"""

_ADD_FILE_HEAD = "hello\nworld\n"


def test_reconstruct_base_add_to_empty():
    """Reversing an add-file patch yields a truly empty base, not a lone newline."""
    result = reconstruct_base_file(_ADD_FILE_HEAD, _ADD_FILE_PATCH)
    # Even though head ends with "\n", an empty base must stay "" (no trailing
    # newline appended) so it is falsy for extend_patch().
    assert result == ""


# --- hunks with no head lines (target length 0) ---

_ZERO_CONTEXT_DELETE_PATCH = """--- a/f.txt
+++ b/f.txt
@@ -3,2 +2,0 @@ b
-c
-d
"""


def test_reconstruct_base_zero_context_deletion():
    # `git diff -U0` numbers a pure deletion by the head line before it (`+2,0`),
    # so the removed lines belong after head line 2, not before it.
    assert reconstruct_base_file("a\nb\ne\n", _ZERO_CONTEXT_DELETE_PATCH) == "a\nb\nc\nd\ne\n"


# Verbatim `git diff -U0` output for the base/head pair in the test below.
_ZERO_CONTEXT_MIXED_PATCH = """--- a/f.txt
+++ b/f.txt
@@ -1,0 +2 @@ a
+x
@@ -3 +3,0 @@ b
-c
@@ -5,0 +6,2 @@ e
+y
+z
"""


def test_reconstruct_base_zero_context_mixed_hunks():
    base = "a\nb\nc\nd\ne\n"
    head = "a\nx\nb\nd\ne\ny\nz\n"
    assert reconstruct_base_file(head, _ZERO_CONTEXT_MIXED_PATCH) == base


_EMPTIED_FILE_PATCH = """--- a/f.txt
+++ b/f.txt
@@ -1,2 +0,0 @@
-a
-b
"""


def test_reconstruct_base_file_emptied():
    # Every line removed but the file kept: `+0,0` must not be read as out of bounds.
    # (Only the lines are checked; the trailing newline follows the EOF handling.)
    assert reconstruct_base_file("", _EMPTIED_FILE_PATCH).splitlines() == ["a", "b"]


@pytest.mark.parametrize(("name", "escaped"), [
    ("café.py", r"caf\303\251.py"),
    ("café notes.py", r"caf\303\251 notes.py"),
    ("src/你好.py", r"src/\344\275\240\345\245\275.py"),
    ('quo"te.py', r'quo\"te.py'),
    ("back\\slash.py", r"back\\slash.py"),
    ('café"note.py', r'café\"note.py'),
])
@pytest.mark.parametrize("edit_type", [EDIT_TYPE.MODIFIED, EDIT_TYPE.ADDED, EDIT_TYPE.DELETED, EDIT_TYPE.RENAMED])
def test_parse_git_quoted_paths(name, escaped, edit_type):
    # Match the byte-oriented C quoting emitted by git diff with core.quotePath=true.
    source = f'"a/{escaped}"'
    target = f'"b/{escaped}"'
    old_filename = None
    if edit_type == EDIT_TYPE.ADDED:
        patch = f"new file mode 100644\n--- /dev/null\n+++ {target}\n@@ -0,0 +1 @@\n+new\n"
    elif edit_type == EDIT_TYPE.DELETED:
        patch = f"deleted file mode 100644\n--- {source}\n+++ /dev/null\n@@ -1 +0,0 @@\n-old\n"
    elif edit_type == EDIT_TYPE.RENAMED:
        old_filename = "old " + name
        source = f'"a/old {escaped}"'
        patch = f'similarity index 100%\nrename from "old {escaped}"\nrename to "{escaped}"\n'
    else:
        patch = f"--- {source}\n+++ {target}\n@@ -1 +1 @@\n-old\n+new\n"

    diff = f"diff --git {source} {target}\n{patch}"
    files = parse_unified_diff(diff)
    assert len(files) == 1
    file = files[0]

    assert file.filename == name
    assert file.old_filename == old_filename
    assert file.edit_type == edit_type


def test_parse_unquoted_space_path():
    diff = MODIFY_DIFF.replace("foo.py", "plain file.py")
    file = parse_unified_diff(diff)[0]
    assert file.filename == "plain file.py"
    assert file.edit_type == EDIT_TYPE.MODIFIED
    assert len(parse_unified_diff(diff)) == 1


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_quoted_path_normalization_preserves_hunk_text(newline):
    from pr_agent.git_providers.diff_parsing import _normalize_git_path_headers, to_hunk_only_patch

    metadata = newline.join([
        r'diff --git "a/caf\303\251 notes.py" "b/caf\303\251 notes.py"',
        r'--- "a/caf\303\251 notes.py"',
        r'+++ "b/caf\303\251 notes.py"',
    ]) + newline
    hunk = newline.join([
        "@@ -1,2 +1,2 @@",
        '--- "a/file with spaces"',
        '+++ "b/file with spaces"',
        " unchanged",
        "",
    ])
    diff = metadata + hunk

    normalized = _normalize_git_path_headers(diff)
    files = parse_unified_diff(diff)

    assert normalized[normalized.index("@@"):] == hunk
    assert len(files) == 1
    # unidiff serializes the hunk header, so compare its source lines separately.
    assert to_hunk_only_patch(files[0].patch).split("\n", 1)[1] == hunk.split("\n", 1)[1]
    assert files[0].filename == "café notes.py"


@pytest.mark.parametrize("source,target,old_name,new_name", [
    ('a/old b/file.py', r'"b/caf\303\251.py"', "old b/file.py", "café.py"),
    ('a/old name.py', r'"b/caf\303\251 notes.py"', "old name.py", "café notes.py"),
    (r'"a/caf\303\251 notes.py"', 'b/new name.py', "café notes.py", "new name.py"),
])
def test_rename_only_with_mixed_quoted_paths(source, target, old_name, new_name):
    diff = f"diff --git {source} {target}\nsimilarity index 100%\nrename from old\nrename to new\n"
    files = parse_unified_diff(diff)

    assert len(files) == 1
    assert files[0].filename == new_name
    assert files[0].old_filename == old_name
    assert files[0].edit_type == EDIT_TYPE.RENAMED


def test_mode_only_quoted_path_is_not_a_rename():
    diff = (
        r'diff --git "a/caf\303\251 notes.py" "b/caf\303\251 notes.py"' "\n"
        "old mode 100644\nnew mode 100755\n"
    )
    files = parse_unified_diff(diff)

    assert len(files) == 1
    assert files[0].filename == "café notes.py"
    assert files[0].edit_type == EDIT_TYPE.MODIFIED
    assert files[0].old_filename is None
    assert "old mode 100644\nnew mode 100755\n" in files[0].patch


def test_reconstruct_base_with_quoted_space_path():
    diff = (
        r'diff --git "a/caf\303\251 notes.py" "b/caf\303\251 notes.py"' "\n"
        r'--- "a/caf\303\251 notes.py"' "\n"
        r'+++ "b/caf\303\251 notes.py"' "\n"
        "@@ -1,3 +1,3 @@\n header\n-old\n+new\n footer\n"
    )

    assert reconstruct_base_file("header\nnew\nfooter\n", diff) == "header\nold\nfooter\n"


def test_quoted_path_normalization_resets_for_unprefixed_diff():
    diff = (
        r'diff --git "caf\303\251.py" "caf\303\251.py"' "\n"
        "old mode 100644\nnew mode 100755\n"
        "diff --git plain.py plain.py\n"
        "--- plain.py\n+++ plain.py\n@@ -1 +1 @@\n-old\n+new\n"
    )
    files = parse_unified_diff(diff)

    assert [file.filename for file in files] == ["café.py", "plain.py"]
    assert reconstruct_base_file("new\n", files[1].patch) == "old\n"


def test_diff_parser_does_not_require_git_executable(tmp_path):
    env = dict(os.environ, PATH=str(tmp_path), GIT_PYTHON_GIT_EXECUTABLE=str(tmp_path / "missing-git"))
    result = subprocess.run(
        [sys.executable, "-c", "from pr_agent.git_providers.diff_parsing import parse_unified_diff"],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr


def test_undecodable_git_path_preserves_surrogateescaped_input():
    from pr_agent.git_providers.diff_parsing import _strip_prefix

    path = '"b/bad' + chr(0xDCFF) + r'\".py"'
    assert _strip_prefix(path) == path
