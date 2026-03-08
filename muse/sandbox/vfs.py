"""Virtual filesystem path mapping between host and sandbox."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import PurePosixPath


@dataclass(frozen=True)
class MountPoint:
    """A single host-to-sandbox directory mapping."""

    host_path: str
    sandbox_path: str
    readonly: bool = False


@dataclass
class VFSMapper:
    """Bidirectional host <-> sandbox path mapper."""

    mounts: list[MountPoint] = field(default_factory=list)

    def to_sandbox(self, host_path: str) -> str | None:
        for mount in self.mounts:
            relative = _relative_to(host_path, mount.host_path)
            if relative is not None:
                return str(PurePosixPath(mount.sandbox_path) / relative)
        return None

    def to_host(self, sandbox_path: str) -> str | None:
        for mount in self.mounts:
            relative = _relative_to(sandbox_path, mount.sandbox_path)
            if relative is not None:
                return str(PurePosixPath(mount.host_path) / relative)
        return None

    def is_writable(self, sandbox_path: str) -> bool:
        for mount in self.mounts:
            relative = _relative_to(sandbox_path, mount.sandbox_path)
            if relative is not None:
                return not mount.readonly
        return False


def build_vfs(
    workspace: str,
    outputs: str,
    refs: str | None = None,
) -> VFSMapper:
    """Create a VFSMapper with the standard Muse mount layout."""

    mounts = [
        MountPoint(host_path=workspace, sandbox_path="/mnt/workspace"),
        MountPoint(host_path=outputs, sandbox_path="/mnt/outputs"),
    ]
    if refs:
        mounts.append(MountPoint(host_path=refs, sandbox_path="/mnt/refs", readonly=True))
    return VFSMapper(mounts=mounts)


def _relative_to(path: str, base: str) -> str | None:
    """Return the relative portion if *path* starts with *base*, else None."""

    normalized_path = path.rstrip("/")
    normalized_base = base.rstrip("/")
    if normalized_path == normalized_base:
        return "."
    if normalized_path.startswith(normalized_base + "/"):
        return normalized_path[len(normalized_base) + 1 :]
    return None
