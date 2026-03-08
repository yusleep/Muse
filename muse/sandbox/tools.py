"""High-level sandbox tool functions for agent use."""

from __future__ import annotations

from pathlib import Path

from muse.sandbox.base import ExecResult, Sandbox


async def shell(
    sandbox: Sandbox,
    command: str,
    *,
    timeout: int = 60,
    workdir: str | None = None,
) -> str:
    """Execute a shell command in the sandbox and return a summary."""

    result = await sandbox.exec(command, timeout=timeout, workdir=workdir)
    return result.summary()


async def latex_compile(
    sandbox: Sandbox,
    tex_file: str,
    *,
    timeout: int = 120,
    workdir: str | None = None,
) -> str:
    """Run the thesis-oriented LaTeX build cycle."""

    base_name = tex_file[:-4] if tex_file.endswith(".tex") else tex_file
    steps = [
        f"pdflatex -interaction=nonstopmode -halt-on-error {tex_file}",
        f"bibtex {base_name} 2>/dev/null || true",
        f"pdflatex -interaction=nonstopmode -halt-on-error {tex_file}",
        f"pdflatex -interaction=nonstopmode -halt-on-error {tex_file}",
    ]

    step_timeout = max(1, timeout // len(steps))
    all_stdout: list[str] = []
    all_stderr: list[str] = []
    final_exit = 0

    for index, step_command in enumerate(steps, start=1):
        result = await sandbox.exec(step_command, timeout=step_timeout, workdir=workdir)
        all_stdout.append(f"=== Step {index}: {step_command.split()[0]} ===\n{result.stdout}")
        if result.stderr.strip():
            all_stderr.append(f"=== Step {index} stderr ===\n{result.stderr}")
        if result.timed_out:
            return ExecResult(
                exit_code=137,
                stdout="\n".join(all_stdout),
                stderr="\n".join(all_stderr) + f"\nBuild timed out at step {index}",
                timed_out=True,
            ).summary()
        final_exit = result.exit_code

    pdf_name = f"{base_name}.pdf"
    summary_parts = []
    if final_exit == 0:
        summary_parts.append(f"[OK] LaTeX compilation succeeded. Output: {pdf_name}")
    else:
        summary_parts.append(f"[FAILED] LaTeX compilation failed (exit={final_exit})")

    log_name = f"{base_name}.log"
    try:
        log_path = f"{workdir}/{log_name}" if workdir else log_name
        log_text = (await sandbox.read_file(log_path)).decode("utf-8", errors="replace")
        errors = _extract_latex_errors(log_text)
        if errors:
            summary_parts.append("Errors:\n" + "\n".join(f"  - {error}" for error in errors[:20]))
    except FileNotFoundError:
        pass

    if all_stderr:
        summary_parts.append("stderr:\n" + "\n".join(all_stderr)[:2000])

    return "\n".join(summary_parts)


async def run_python(
    sandbox: Sandbox,
    script: str,
    *,
    timeout: int = 60,
    workdir: str | None = None,
) -> str:
    """Execute a Python script in the sandbox."""

    file_name = "_muse_script.py"
    script_path = f"{workdir}/{file_name}" if workdir else file_name
    await sandbox.write_file(script_path, script.encode("utf-8"))
    command_path = file_name if workdir else script_path
    result = await sandbox.exec(
        f"python3 {command_path}",
        timeout=timeout,
        workdir=workdir,
    )
    return result.summary()


async def present_file(
    sandbox: Sandbox,
    source_path: str,
    *,
    dest_name: str | None = None,
) -> str:
    """Copy a file from the sandbox workspace to a user-facing outputs area."""

    try:
        content = await sandbox.read_file(source_path)
    except FileNotFoundError:
        return f"[FAILED] Source file not found: {source_path}"

    target_name = dest_name or Path(source_path).name
    outputs_dir = getattr(sandbox, "outputs_dir", None)
    if outputs_dir is not None:
        target_path = Path(outputs_dir) / target_name
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(content)
        return f"[OK] File presented: {target_name} (saved to outputs)"

    result = await sandbox.exec(f"cp /mnt/workspace/{source_path} /mnt/outputs/{target_name}")
    if result.success:
        return f"[OK] File presented: {target_name} (copied to outputs)"
    return f"[FAILED] Could not copy file: {result.stderr}"


def _extract_latex_errors(log_text: str) -> list[str]:
    """Extract concise error lines from a LaTeX log file."""

    errors: list[str] = []
    for line in log_text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("!"):
            errors.append(stripped)
        elif "Error:" in stripped and len(stripped) < 200:
            errors.append(stripped)
        elif "Fatal error" in stripped:
            errors.append(stripped)
    return errors
