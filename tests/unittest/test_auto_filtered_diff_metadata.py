from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from starlette_context import context, request_cycle_context

import pr_agent.algo.pr_processing as pr_processing
import pr_agent.algo.token_budget as token_budget
from pr_agent.algo.pr_processing import PreparedPRDiff, get_pr_diff
from pr_agent.algo.types import EDIT_TYPE
from pr_agent.config_loader import get_settings
from pr_agent.git_providers.git_provider import GitProvider, IncrementalPR
from pr_agent.git_providers.github_provider import GithubProvider
from pr_agent.git_providers.gitlab_provider import GitLabProvider


@pytest.fixture(autouse=True)
def ignore_settings(monkeypatch):
    monkeypatch.setitem(get_settings().ignore, "glob", [])
    monkeypatch.setitem(get_settings().ignore, "regex", [])
    monkeypatch.setitem(get_settings().config, "ignore_language_framework", [])


@pytest.fixture(params=["github", "gitlab"])
def provider_factory(request):
    def make(names, status="modified"):
        if request.param == "github":
            provider = GithubProvider.__new__(GithubProvider)
            raw_files = [SimpleNamespace(
                filename=name, status=status, previous_filename="old/" + name,
                patch="@@ -1 +1 @@\n-old\n+new\n", additions=1, deletions=1,
            ) for name in names]
            provider.pr = SimpleNamespace(
                base=SimpleNamespace(sha="base"), head=SimpleNamespace(sha="head"),
                get_files=Mock(return_value=raw_files), changed_files=len(raw_files),
            )
            provider.repo_obj = Mock()
            provider.repo_obj.compare.return_value = SimpleNamespace(merge_base_commit=provider.pr.base)
            provider._get_pr_file_content = Mock(return_value="new\n")
            provider.content_reads = provider._get_pr_file_content
        else:
            provider = GitLabProvider.__new__(GitLabProvider)
            raw_files = [dict(
                old_path="old/" + name if status == "renamed" else name, new_path=name,
                new_file=status == "added", deleted_file=status == "removed", renamed_file=status == "renamed",
                diff="@@ -1 +1 @@\n-old\n+new\n",
            ) for name in names]
            provider._get_merge_request_changes = Mock(return_value={
                "changes": raw_files, "diff_refs": {"base_sha": "base", "head_sha": "head"},
            })
            provider._expand_submodule_changes = lambda changes, refs: changes
            provider.get_pr_file_content = Mock(return_value="new\n")
            provider.content_reads = provider.get_pr_file_content
            provider.id_mr = 7
        provider.diff_files = None
        provider.git_files = None
        provider.incremental = IncrementalPR(False)
        provider.get_languages = lambda: {}
        return provider, raw_files

    return make


def _handler():
    return SimpleNamespace(prompt_tokens=7, count_tokens=lambda text: len(text.encode()))


@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("line_numbers", [False, True])
def test_lockfile_metadata_reaches_full_diff_without_fetching_its_contents(provider_factory, prepared, line_numbers):
    provider, _ = provider_factory(["package.json", "pnpm-lock.yaml"])
    result = get_pr_diff(provider, _handler(), "gpt-4", disable_extra_lines=True,
                         add_line_numbers_to_hunks=line_numbers, return_prepared=prepared)
    diff = result.diff if prepared else result

    assert "package.json" in diff
    assert "modified: 'pnpm-lock.yaml'" in diff
    assert "## File: 'pnpm-lock.yaml'" not in diff
    assert [file.filename for file in provider.get_diff_files()] == ["package.json"]
    assert provider.content_reads.call_count == 2
    if prepared:
        assert isinstance(result, PreparedPRDiff)
        assert result.remaining_files_list == []


@pytest.mark.parametrize("status, expected", [
    ("added", EDIT_TYPE.ADDED), ("modified", EDIT_TYPE.MODIFIED),
    ("removed", EDIT_TYPE.DELETED), ("renamed", EDIT_TYPE.RENAMED),
])
def test_omitted_only_changes_preserve_status_without_blob_reads(provider_factory, status, expected):
    provider, _ = provider_factory(["uv.lock", "logo.png"], status)
    diff = get_pr_diff(provider, _handler(), "gpt-4", disable_extra_lines=True)

    assert provider.get_auto_filtered_files() == {"uv.lock": expected, "logo.png": expected}
    assert f"{expected.name.lower()}: 'uv.lock'" in diff
    assert f"{expected.name.lower()}: 'logo.png'" in diff
    assert "old/" not in diff
    provider.content_reads.assert_not_called()


def test_explicit_ignore_names_are_not_disclosed(provider_factory, monkeypatch):
    monkeypatch.setitem(get_settings().ignore, "glob", ["private/*"])
    provider, _ = provider_factory(["package.json", "pnpm-lock.yaml", "private/uv.lock", "private/notes.py"])
    diff = get_pr_diff(provider, _handler(), "gpt-4", disable_extra_lines=True)

    assert "pnpm-lock.yaml" in diff
    assert "private/" not in diff
    assert provider.get_auto_filtered_files() == {"pnpm-lock.yaml": EDIT_TYPE.MODIFIED}
    assert provider.content_reads.call_count == 2


def test_compressed_diff_retains_metadata_within_output_reserve(provider_factory, monkeypatch):
    provider, _ = provider_factory(["package.json", "pnpm-lock.yaml"])
    handler = _handler()
    monkeypatch.setattr(token_budget, "get_max_tokens", lambda *args, **kwargs: 1700)
    # The extended diff exceeds the 200-token soft budget but the compressed patch
    # and metadata fit within the 700-token hard budget, preserving the output reserve.
    provider.get_diff_files()[0].patch = "@@ -1 +1 @@\n-" + "x" * 300 + "\n+new\n"
    diff = get_pr_diff(provider, handler, "gpt-4", disable_extra_lines=True, return_remaining_files=True)[0]

    assert "modified: 'pnpm-lock.yaml'" in diff
    assert handler.prompt_tokens + handler.count_tokens(diff) <= 700


@pytest.mark.parametrize("in_request", [False, True])
def test_cached_metadata_and_command_reset(provider_factory, in_request):
    provider, _ = provider_factory(["package.json", "pnpm-lock.yaml"])
    with request_cycle_context({}) if in_request else nullcontext():
        first = get_pr_diff(provider, _handler(), "gpt-4", disable_extra_lines=True)
        reads = provider.content_reads.call_count
        assert get_pr_diff(provider, _handler(), "gpt-4", disable_extra_lines=True) == first
        assert provider.content_reads.call_count == reads
        if isinstance(provider, GithubProvider):
            provider.reset_diff_cache_for_command()
            assert provider.get_auto_filtered_files() == {}
            if in_request:
                assert "auto_filtered_files" not in context
        else:
            provider.get_incremental_commits(IncrementalPR(False))
            assert provider.get_auto_filtered_files() == {}
        get_settings().ignore.glob = ["*.yaml"]
        rebuilt = get_pr_diff(provider, _handler(), "gpt-4", disable_extra_lines=True)
        assert "pnpm-lock.yaml" not in rebuilt
        assert provider.get_auto_filtered_files() == {}


def test_incremental_scope_replaces_full_metadata(provider_factory):
    provider, raw = provider_factory(["pnpm-lock.yaml", "uv.lock"])
    provider.get_diff_files()
    assert set(provider.get_auto_filtered_files()) == {"pnpm-lock.yaml", "uv.lock"}

    def select_incremental():
        provider.unreviewed_files_map = {"uv.lock": raw[1]}
        provider._incremental_head_sha = "head"
        provider.incremental.last_seen_commit = SimpleNamespace(sha="base")

    provider._get_incremental_commits = select_incremental
    provider.get_incremental_commits(IncrementalPR(True))
    diff = get_pr_diff(provider, _handler(), "gpt-4", disable_extra_lines=True)
    assert "uv.lock" in diff
    assert "pnpm-lock.yaml" not in diff


def test_provider_default_does_not_add_metadata():
    provider = SimpleNamespace()
    assert GitProvider.get_auto_filtered_files(provider) == {}
    assert pr_processing._append_auto_filtered_files("patch", provider, _handler(), 1000) == "patch"


@pytest.mark.parametrize("extra_budget", [-1, 0, 1, 1000])
def test_metadata_uses_complete_escaped_names_and_exact_token_budget(extra_budget):
    provider = SimpleNamespace(get_auto_filtered_files=lambda: {
        "first\nfile.lock": EDIT_TYPE.MODIFIED, "second.lock": EDIT_TYPE.DELETED,
    })
    handler = _handler()
    prefix = "patch"
    first = (prefix + "\n\nFiles changed but automatically filtered from the diff (contents not reviewed):"
             "\nmodified: 'first\\nfile.lock'")
    limit = handler.prompt_tokens + handler.count_tokens(first) + extra_budget
    result = pr_processing._append_auto_filtered_files(prefix, provider, handler, limit)

    assert handler.prompt_tokens + handler.count_tokens(result) <= limit
    assert "first\nfile.lock" not in result
    if extra_budget < 0:
        assert result == prefix
    else:
        assert "modified: 'first\\nfile.lock'" in result
    assert "second.lock" not in result or "deleted: 'second.lock'" in result


def test_no_automatic_exclusions_leave_diff_unchanged(provider_factory):
    provider, _ = provider_factory(["package.json"])
    diff = get_pr_diff(provider, _handler(), "gpt-4", disable_extra_lines=True)
    assert "omitted" not in diff
    assert provider.get_auto_filtered_files() == {}


def test_request_cache_restores_metadata_on_another_github_instance():
    first = GithubProvider.__new__(GithubProvider)
    first.diff_files = None
    with request_cycle_context({"diff_files": ["cached diff"], "auto_filtered_files": {"uv.lock": EDIT_TYPE.ADDED}}):
        assert first.get_diff_files() == ["cached diff"]
        assert first.get_auto_filtered_files() == {"uv.lock": EDIT_TYPE.ADDED}


def test_exact_combined_count_rejects_metadata_that_does_not_fit():
    provider = SimpleNamespace(get_auto_filtered_files=lambda: {"uv.lock": EDIT_TYPE.MODIFIED})
    handler = SimpleNamespace(prompt_tokens=10, count_tokens=lambda text: 100 if "patch\n\nFiles" in text else 1)
    assert pr_processing._append_auto_filtered_files("patch", provider, handler, 109) == "patch"


def test_configured_bad_extension_is_automatically_filtered(provider_factory, monkeypatch):
    monkeypatch.setitem(get_settings().bad_extensions, "default", ["py"])
    provider, _ = provider_factory(["source.py"])
    diff = get_pr_diff(provider, _handler(), "gpt-4", disable_extra_lines=True)
    assert "modified: 'source.py'" in diff
    assert "automatically filtered" in diff
    assert "contents not reviewed" in diff
    assert "binary" not in diff
    provider.content_reads.assert_not_called()
