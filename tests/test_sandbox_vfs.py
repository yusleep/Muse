"""Tests for VFS path mapping (muse.sandbox.vfs)."""

from __future__ import annotations

import pytest

from muse.sandbox.vfs import MountPoint, VFSMapper, build_vfs


class TestVFSMapper:
    @pytest.fixture
    def mapper(self):
        return VFSMapper(
            mounts=[
                MountPoint(host_path="/home/user/runs/r1/workspace", sandbox_path="/mnt/workspace"),
                MountPoint(host_path="/home/user/runs/r1/outputs", sandbox_path="/mnt/outputs"),
                MountPoint(host_path="/home/user/refs", sandbox_path="/mnt/refs", readonly=True),
            ]
        )

    def test_to_sandbox_workspace(self, mapper):
        assert mapper.to_sandbox("/home/user/runs/r1/workspace/ch1.tex") == "/mnt/workspace/ch1.tex"

    def test_to_sandbox_outputs(self, mapper):
        assert mapper.to_sandbox("/home/user/runs/r1/outputs/thesis.pdf") == "/mnt/outputs/thesis.pdf"

    def test_to_sandbox_refs(self, mapper):
        assert mapper.to_sandbox("/home/user/refs/paper.pdf") == "/mnt/refs/paper.pdf"

    def test_to_sandbox_root_mount(self, mapper):
        assert mapper.to_sandbox("/home/user/runs/r1/workspace") == "/mnt/workspace"

    def test_to_sandbox_unmapped(self, mapper):
        assert mapper.to_sandbox("/tmp/random/file.txt") is None

    def test_to_host_workspace(self, mapper):
        assert mapper.to_host("/mnt/workspace/ch1.tex") == "/home/user/runs/r1/workspace/ch1.tex"

    def test_to_host_outputs(self, mapper):
        assert mapper.to_host("/mnt/outputs/thesis.pdf") == "/home/user/runs/r1/outputs/thesis.pdf"

    def test_to_host_unmapped(self, mapper):
        assert mapper.to_host("/var/log/syslog") is None

    def test_is_writable_workspace(self, mapper):
        assert mapper.is_writable("/mnt/workspace/ch1.tex") is True

    def test_is_writable_outputs(self, mapper):
        assert mapper.is_writable("/mnt/outputs/out.pdf") is True

    def test_is_writable_refs_readonly(self, mapper):
        assert mapper.is_writable("/mnt/refs/paper.pdf") is False

    def test_is_writable_unmapped(self, mapper):
        assert mapper.is_writable("/tmp/file") is False

    def test_nested_paths(self, mapper):
        assert (
            mapper.to_sandbox("/home/user/runs/r1/workspace/chapters/ch1/main.tex")
            == "/mnt/workspace/chapters/ch1/main.tex"
        )

    def test_trailing_slashes_handled(self):
        mapper = VFSMapper(mounts=[MountPoint(host_path="/a/b/", sandbox_path="/mnt/x/")])
        assert mapper.to_sandbox("/a/b/file.txt") == "/mnt/x/file.txt"
        assert mapper.to_host("/mnt/x/file.txt") == "/a/b/file.txt"


class TestBuildVfs:
    def test_standard_layout(self):
        vfs = build_vfs(
            workspace="/home/user/runs/r1/workspace",
            outputs="/home/user/runs/r1/outputs",
            refs="/home/user/refs",
        )
        assert len(vfs.mounts) == 3
        assert vfs.to_sandbox("/home/user/refs/p.pdf") == "/mnt/refs/p.pdf"

    def test_no_refs(self):
        vfs = build_vfs(
            workspace="/ws",
            outputs="/out",
        )
        assert len(vfs.mounts) == 2
