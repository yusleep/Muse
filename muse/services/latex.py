"""Vendored BUPT LaTeX project export helpers."""

from __future__ import annotations

import re
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Any

TEMPLATE_ROOT = Path(__file__).resolve().parents[1] / "templates" / "bupt_latex"
REQUIRED_TEMPLATE_ASSETS = (
    Path("main.tex"),
    Path("Bib"),
    Path("Chapter"),
    Path("config"),
    Path("resources"),
)

_TITLE_PLACEHOLDER = r"\input{Chapter/chapter1}"
_BIBLIOGRAPHY_SENTINEL = r"\bibliographystyle{plainnat}"
_MARKDOWN_IMAGE_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<path>[^)]+)\)")
_LATEX_PASSTHROUGH_RE = re.compile(
    r"(\\\((?:.*?)\\\)|\\\[(?:.*?)\\\]|\\[A-Za-z]+(?:\[[^\]]*\])?(?:\{[^{}]*\})*|\$\$.*?\$\$|\$[^$\n]+\$)"
)
_JSON_ESCAPED_LATEX_CONTROL_RE = re.compile(r"([\b\t\f\r])(?=[A-Za-z])")
_DOUBLE_ESCAPED_MATH_DELIMITER_RE = re.compile(r"\\\\([()\[\]])")
_JSON_ESCAPED_LATEX_CONTROL_MAP = {
    "\b": "b",
    "\t": "t",
    "\f": "f",
    "\r": "r",
}
_PDF_OUTPUT_NAME = "thesis.pdf"
_ZIP_OUTPUT_NAME = "latex_project.zip"


def _restore_json_escaped_latex_controls(value: Any) -> str:
    text = str(value or "")
    text = _DOUBLE_ESCAPED_MATH_DELIMITER_RE.sub(r"\\\1", text)
    return _JSON_ESCAPED_LATEX_CONTROL_RE.sub(
        lambda match: "\\" + _JSON_ESCAPED_LATEX_CONTROL_MAP[match.group(1)],
        text,
    )


def _metadata_bucket(state: dict[str, Any]) -> dict[str, Any]:
    metadata = state.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _first_present(state: dict[str, Any], *keys: str) -> Any:
    metadata = _metadata_bucket(state)
    for key in keys:
        value = state.get(key)
        if value not in (None, "", []):
            return value
        value = metadata.get(key)
        if value not in (None, "", []):
            return value
    return None


def _latex_escape(value: Any) -> str:
    text = _restore_json_escaped_latex_controls(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for source, replacement in replacements.items():
        text = text.replace(source, replacement)
    return text


def _join_keywords(raw_value: Any, *, separator: str, default: str) -> str:
    if isinstance(raw_value, str):
        value = raw_value.strip()
        return value or default
    if isinstance(raw_value, list):
        parts = [str(item).strip() for item in raw_value if str(item).strip()]
        if parts:
            return separator.join(parts)
    return default


def _latex_escape_with_passthrough(value: Any) -> str:
    text = _restore_json_escaped_latex_controls(value)
    rendered: list[str] = []
    cursor = 0

    for match in _LATEX_PASSTHROUGH_RE.finditer(text):
        if match.start() > cursor:
            rendered.append(_latex_escape(text[cursor:match.start()]))
        rendered.append(match.group(0))
        cursor = match.end()

    if cursor < len(text):
        rendered.append(_latex_escape(text[cursor:]))

    return "".join(rendered)


def _render_info_tex(state: dict[str, Any]) -> str:
    title_zh = _first_present(state, "title_zh", "thesis_title_zh", "thesis_title", "title", "topic")
    title_en = _first_present(state, "title_en", "thesis_title_en", "english_title")
    author_name = _first_present(state, "author_name", "author", "student_name")
    student_id = _first_present(state, "student_id", "student_no", "student_number")
    discipline_name = _first_present(state, "discipline_name", "major", "discipline")
    supervisor_name = _first_present(state, "supervisor_name", "advisor", "supervisor")
    graduation_date = _first_present(state, "graduation_date", "graduate_date")

    keywords_zh = _join_keywords(state.get("keywords_zh"), separator="；", default="关键词待填写")
    keywords_en = _join_keywords(state.get("keywords_en"), separator="; ", default="keywords pending")

    title_zh = _latex_escape(title_zh or "待填写中文题目")
    title_en = _latex_escape(title_en or title_zh or "Thesis Title Placeholder")
    author_name = _latex_escape(author_name or "待填写作者")
    student_id = _latex_escape(student_id or "2025000000")
    discipline_name = _latex_escape(discipline_name or "待填写学科专业")
    supervisor_name = _latex_escape(supervisor_name or "待填写导师")
    graduation_date = _latex_escape(graduation_date or "2026年3月")
    keywords_zh = _latex_escape(keywords_zh)
    keywords_en = _latex_escape(keywords_en)

    return "\n".join(
        [
            f"\\newcommand{{\\thesistitlezh}}{{{title_zh}}}",
            f"\\newcommand{{\\thesistitleen}}{{{title_en}}}",
            f"\\newcommand{{\\authorname}}{{{author_name}}}",
            f"\\newcommand{{\\studentid}}{{{student_id}}}",
            f"\\newcommand{{\\disciplinename}}{{{discipline_name}}}",
            f"\\newcommand{{\\supervisorname}}{{{supervisor_name}}}",
            f"\\newcommand{{\\graduatedate}}{{{graduation_date}}}",
            f"\\newcommand{{\\keywordszh}}{{{keywords_zh}}}",
            f"\\newcommand{{\\keywordsen}}{{{keywords_en}}}",
            "",
        ]
    )


def _render_abstract(text: Any, keywords: Any, *, keyword_label: str, separator: str, placeholder: str) -> str:
    body = str(text or "").strip() or placeholder
    lines = [_latex_escape(line) for line in body.splitlines() if line.strip()]
    rendered = "\n\n".join(lines) if lines else _latex_escape(placeholder)

    joined_keywords = _join_keywords(keywords, separator=separator, default="")
    if joined_keywords:
        rendered = f"{rendered}\n\n{_latex_escape(keyword_label)}{_latex_escape(joined_keywords)}"
    return f"{rendered}\n"


def _candidate_asset_roots(state: dict[str, Any], *, store: Any) -> list[Path]:
    roots: list[Path] = []
    seen: set[str] = set()

    def _add(raw_value: Any) -> None:
        text = str(raw_value or "").strip()
        if not text:
            return
        resolved = Path(text).expanduser().resolve()
        key = str(resolved)
        if key not in seen:
            seen.add(key)
            roots.append(resolved)

    _add(Path.cwd())

    base_dir = getattr(store, "base_dir", None)
    if base_dir:
        _add(base_dir)
        _add(Path(base_dir).expanduser().resolve().parent)

    metadata = _metadata_bucket(state)
    for key in ("asset_root", "source_root", "source_dir", "working_dir", "workspace_root"):
        _add(metadata.get(key))

    asset_roots = metadata.get("asset_roots")
    if isinstance(asset_roots, list):
        for item in asset_roots:
            _add(item)

    return roots


def _resolve_asset_source(raw_path: str, *, search_roots: list[Path]) -> Path | None:
    candidate = str(raw_path or "").strip().strip("<>")
    if not candidate or "://" in candidate:
        return None

    candidate_path = Path(candidate).expanduser()
    probes = [candidate_path]
    if not candidate_path.is_absolute():
        probes.extend((root / candidate_path).resolve() for root in search_roots)

    for probe in probes:
        if probe.is_file():
            return probe.resolve()
    return None


def _copy_asset_into_project(
    source_path: Path,
    *,
    project_dir: Path,
    copied_assets: dict[str, str],
) -> str:
    source_key = str(source_path.resolve())
    existing = copied_assets.get(source_key)
    if existing:
        return existing

    asset_root = project_dir / "resources" / "generated_assets"
    asset_root.mkdir(parents=True, exist_ok=True)

    suffix = source_path.suffix.lower() or ".bin"
    relative_path = Path("resources") / "generated_assets" / f"asset_{len(copied_assets) + 1}{suffix}"
    destination = project_dir / relative_path
    shutil.copy2(source_path, destination)
    copied_assets[source_key] = relative_path.as_posix()
    return copied_assets[source_key]


def _render_figure_block(asset_path: str, alt_text: str) -> list[str]:
    figure_lines = [
        r"\begin{figure}[htbp]",
        r"\centering",
        f"\\includegraphics[width=0.9\\textwidth]{{{asset_path}}}",
    ]
    if alt_text:
        figure_lines.append(f"\\caption{{{_latex_escape(alt_text)}}}")
    figure_lines.append(r"\end{figure}")
    return figure_lines


def _render_markdown_body(
    text: str,
    *,
    project_dir: Path | None = None,
    warnings: list[str] | None = None,
    copied_assets: dict[str, str] | None = None,
    search_roots: list[Path] | None = None,
) -> str:
    text = _restore_json_escaped_latex_controls(text)
    heading_commands = {
        1: "section",
        2: "subsection",
        3: "subsubsection",
        4: "paragraph",
        5: "subparagraph",
    }
    rendered_lines: list[str] = []
    in_display_math = False

    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            rendered_lines.append("")
            continue

        if stripped == r"\[":
            rendered_lines.append(r"\[")
            in_display_math = True
            continue

        if stripped == r"\]":
            rendered_lines.append(r"\]")
            in_display_math = False
            continue

        if in_display_math:
            rendered_lines.append(_restore_json_escaped_latex_controls(stripped))
            continue

        heading_match = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if heading_match:
            level = min(len(heading_match.group(1)), 5)
            command = heading_commands[level]
            title = _latex_escape(heading_match.group(2).strip())
            rendered_lines.append(f"\\{command}{{{title}}}")
            rendered_lines.append("")
            continue

        image_match = _MARKDOWN_IMAGE_RE.fullmatch(stripped)
        if image_match and project_dir is not None and warnings is not None and copied_assets is not None:
            source_path = _resolve_asset_source(
                image_match.group("path"),
                search_roots=search_roots or [Path.cwd()],
            )
            if source_path is None:
                warnings.append(
                    f"Could not resolve referenced asset '{image_match.group('path')}'. Leaving the original markdown image text in chapter output."
                )
                rendered_lines.append(_latex_escape(stripped))
                continue

            asset_path = _copy_asset_into_project(source_path, project_dir=project_dir, copied_assets=copied_assets)
            rendered_lines.extend(_render_figure_block(asset_path, image_match.group("alt").strip()))
            rendered_lines.append("")
            continue

        rendered_lines.append(_latex_escape_with_passthrough(stripped))

    return "\n".join(rendered_lines).strip()


def _render_chapter_tex(
    chapter_title: str,
    chapter_text: str,
    *,
    project_dir: Path,
    warnings: list[str],
    copied_assets: dict[str, str],
    search_roots: list[Path],
) -> str:
    rendered_body = _render_markdown_body(
        chapter_text,
        project_dir=project_dir,
        warnings=warnings,
        copied_assets=copied_assets,
        search_roots=search_roots,
    )
    parts = [f"\\chapter{{{_latex_escape(chapter_title)}}}"]
    if rendered_body:
        parts.extend(["", rendered_body])
    parts.append("")
    return "\n".join(parts)


def _chapter_payloads(state: dict[str, Any]) -> list[tuple[str, str]]:
    chapter_results = state.get("chapter_results", [])
    payloads: list[tuple[str, str]] = []

    if isinstance(chapter_results, list):
        for idx, chapter in enumerate(chapter_results, start=1):
            if not isinstance(chapter, dict):
                continue
            title = str(chapter.get("chapter_title") or f"第{idx}章")
            text = str(chapter.get("merged_text") or "").strip()
            payloads.append((title, text))

    if payloads:
        return payloads

    # Fallback: reconstruct from chapters dict (uses _merge_dict, always preserved)
    chapters = state.get("chapters")
    if isinstance(chapters, dict) and chapters:
        chapter_plans = state.get("chapter_plans", [])
        if chapter_plans:
            for idx, plan in enumerate(chapter_plans, start=1):
                cid = str(plan.get("chapter_id", ""))
                ch = chapters.get(cid)
                if isinstance(ch, dict):
                    title = str(ch.get("chapter_title") or plan.get("chapter_title") or f"第{idx}章")
                    text = str(ch.get("merged_text") or "").strip()
                    payloads.append((title, text))
        else:
            for idx, ch in enumerate(chapters.values(), start=1):
                if isinstance(ch, dict):
                    title = str(ch.get("chapter_title") or f"第{idx}章")
                    text = str(ch.get("merged_text") or "").strip()
                    payloads.append((title, text))
        if payloads:
            return payloads

    final_text = str(state.get("final_text") or "").strip()
    if final_text:
        return [(str(_first_present(state, "title_zh", "topic") or "正文"), final_text)]

    return [("正文示例章节", "")]


def _normalized_cite_key(cite_key: Any) -> str:
    key = str(cite_key or "").strip()
    if key.startswith("@"):
        key = key[1:]
    key = re.sub(r"[^A-Za-z0-9:_-]+", "_", key)
    return key or "reference"


def _ordered_citation_keys(state: dict[str, Any]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()

    for use in state.get("citation_uses", []):
        if not isinstance(use, dict):
            continue
        cite_key = str(use.get("cite_key") or "").strip()
        if cite_key and cite_key not in seen:
            seen.add(cite_key)
            ordered.append(cite_key)

    for ref in state.get("references", []):
        if not isinstance(ref, dict):
            continue
        ref_id = str(ref.get("ref_id") or "").strip()
        if ref_id and ref_id not in seen:
            seen.add(ref_id)
            ordered.append(ref_id)

    return ordered


def _reference_lookup(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for ref in state.get("references", []):
        if not isinstance(ref, dict):
            continue
        ref_id = str(ref.get("ref_id") or "").strip()
        if not ref_id:
            continue
        lookup[ref_id] = ref
        lookup.setdefault(_normalized_cite_key(ref_id), ref)
    return lookup


def _format_bibtex_entry(entry_type: str, cite_key: str, fields: list[tuple[str, str]]) -> str:
    lines = [f"@{entry_type}{{{cite_key},"]
    for key, value in fields:
        lines.append(f"  {key} = {{{_latex_escape(value)}}},")
    lines[-1] = lines[-1].rstrip(",")
    lines.append("}")
    return "\n".join(lines)


def _is_complete_reference(ref: dict[str, Any]) -> bool:
    title = str(ref.get("title") or "").strip()
    authors = ref.get("authors") or []
    author_names = [str(author).strip() for author in authors if str(author).strip()]
    year = ref.get("year")
    return bool(title and author_names and year not in (None, ""))


def _render_bibliography_entry(cite_key: str, ref: dict[str, Any] | None, warnings: list[str]) -> str:
    normalized_key = _normalized_cite_key(cite_key)

    if ref is None:
        warnings.append(
            f"Reference {normalized_key} is missing from state['references']; wrote a placeholder BibTeX entry instead of dropping it."
        )
        return _format_bibtex_entry(
            "misc",
            normalized_key,
            [
                ("title", f"Missing reference metadata for {normalized_key}"),
                ("note", "Placeholder bibliography entry generated from missing reference metadata."),
            ],
        )

    title = str(ref.get("title") or "").strip()
    authors = [str(author).strip() for author in (ref.get("authors") or []) if str(author).strip()]
    year = ref.get("year")
    venue = str(ref.get("venue") or "").strip()
    doi = str(ref.get("doi") or "").strip()

    if not _is_complete_reference(ref):
        warnings.append(
            f"Reference {normalized_key} has incomplete metadata; wrote a placeholder BibTeX entry instead of dropping it."
        )
        fields = [
            ("title", title or f"Untitled reference {normalized_key}"),
            ("note", "Placeholder bibliography entry generated from incomplete metadata."),
        ]
        if authors:
            fields.insert(1, ("author", " and ".join(authors)))
        if year not in (None, ""):
            fields.insert(min(len(fields), 2), ("year", str(year)))
        return _format_bibtex_entry("misc", normalized_key, fields)

    entry_type = "article" if venue else "misc"
    fields = [
        ("title", title),
        ("author", " and ".join(authors)),
        ("year", str(year)),
    ]
    if venue:
        fields.append(("journal", venue))
    if doi:
        fields.append(("doi", doi))
    return _format_bibtex_entry(entry_type, normalized_key, fields)


def _write_bibliography(project_dir: Path, state: dict[str, Any], warnings: list[str]) -> None:
    cite_keys = _ordered_citation_keys(state)
    ref_lookup = _reference_lookup(state)
    entries = [
        _render_bibliography_entry(cite_key, ref_lookup.get(cite_key) or ref_lookup.get(_normalized_cite_key(cite_key)), warnings)
        for cite_key in cite_keys
    ]

    if not entries:
        entries = [
            _format_bibtex_entry(
                "misc",
                "placeholder",
                [
                    ("title", "Placeholder Reference"),
                    ("author", "Muse"),
                    ("year", "2026"),
                    ("note", "Replace with exported bibliography entries."),
                ],
            )
        ]

    bib_path = project_dir / "Bib" / "thesis.bib"
    bib_path.write_text("\n\n".join(entries) + "\n", encoding="utf-8")


def _write_rendered_files(project_dir: Path, state: dict[str, Any], warnings: list[str], *, store: Any) -> None:
    copied_assets: dict[str, str] = {}
    search_roots = _candidate_asset_roots(state, store=store)

    info_path = project_dir / "config" / "info.tex"
    info_path.write_text(_render_info_tex(state), encoding="utf-8")

    abstract_zh_path = project_dir / "Chapter" / "abstract_zh.tex"
    abstract_zh_path.write_text(
        _render_abstract(
            state.get("abstract_zh"),
            state.get("keywords_zh"),
            keyword_label="关键词：",
            separator="；",
            placeholder="待填写中文摘要。",
        ),
        encoding="utf-8",
    )

    abstract_en_path = project_dir / "Chapter" / "abstract_en.tex"
    abstract_en_path.write_text(
        _render_abstract(
            state.get("abstract_en"),
            state.get("keywords_en"),
            keyword_label="Keywords: ",
            separator="; ",
            placeholder="English abstract goes here.",
        ),
        encoding="utf-8",
    )

    chapter_inputs: list[str] = []
    for idx, (chapter_title, chapter_text) in enumerate(_chapter_payloads(state), start=1):
        chapter_name = f"chapter{idx}"
        chapter_path = project_dir / "Chapter" / f"{chapter_name}.tex"
        chapter_path.write_text(
            _render_chapter_tex(
                chapter_title,
                chapter_text,
                project_dir=project_dir,
                warnings=warnings,
                copied_assets=copied_assets,
                search_roots=search_roots,
            ),
            encoding="utf-8",
        )
        chapter_inputs.append(f"\\input{{Chapter/{chapter_name}}}")

    main_path = project_dir / "main.tex"
    main_text = main_path.read_text(encoding="utf-8")
    if _TITLE_PLACEHOLDER not in main_text:
        raise RuntimeError(
            "Unable to update LaTeX chapter inputs: expected placeholder "
            f"'{_TITLE_PLACEHOLDER}' in {main_path}."
        )
    rendered_main = main_text.replace(_TITLE_PLACEHOLDER, "\n".join(chapter_inputs))
    cite_keys = [_normalized_cite_key(cite_key) for cite_key in _ordered_citation_keys(state)]
    if cite_keys:
        if _BIBLIOGRAPHY_SENTINEL not in rendered_main:
            raise RuntimeError(
                "Unable to inject bibliography references: expected sentinel "
                f"'{_BIBLIOGRAPHY_SENTINEL}' in {main_path}."
            )
        rendered_main = rendered_main.replace(
            _BIBLIOGRAPHY_SENTINEL,
            f"\\nocite{{{','.join(cite_keys)}}}\n\n{_BIBLIOGRAPHY_SENTINEL}",
            1,
        )
    main_path.write_text(rendered_main, encoding="utf-8")

    _write_bibliography(project_dir, state, warnings)


def _missing_template_assets(template_root: Path) -> list[str]:
    return [
        asset.as_posix()
        for asset in REQUIRED_TEMPLATE_ASSETS
        if not (template_root / asset).exists()
    ]


def _validate_template_assets(template_root: Path) -> None:
    missing = _missing_template_assets(template_root)
    if missing:
        missing_str = ", ".join(missing)
        raise RuntimeError(
            "Missing LaTeX template assets: "
            f"{missing_str}. Expected them under {template_root} "
            "and re-vendor the BUPT template assets before exporting."
        )


def _remove_existing_artifact(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _write_project_archive(project_dir: Path, *, store: Any, run_id: str) -> Path:
    zip_path = Path(store.artifact_path(run_id, f"output/{_ZIP_OUTPUT_NAME}"))
    _remove_existing_artifact(zip_path)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source_path in sorted(project_dir.rglob("*")):
            if source_path.is_file():
                archive.write(source_path, arcname=source_path.relative_to(project_dir).as_posix())

    return zip_path


def _compiler_command(name: str) -> list[str]:
    if name == "latexmk":
        return [
            "latexmk",
            "-xelatex",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-file-line-error",
            "main.tex",
        ]
    return [
        "xelatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-file-line-error",
        "main.tex",
    ]


def _available_tex_compilers() -> list[str]:
    compilers: list[str] = []
    for name in ("latexmk", "xelatex"):
        if shutil.which(name):
            compilers.append(name)
    return compilers


def _compile_project_pdf(project_dir: Path, *, store: Any, run_id: str, warnings: list[str]) -> Path | None:
    pdf_output_path = Path(store.artifact_path(run_id, f"output/{_PDF_OUTPUT_NAME}"))
    _remove_existing_artifact(pdf_output_path)

    compilers = _available_tex_compilers()
    if not compilers:
        warnings.append(
            "Local PDF compilation skipped: install latexmk or xelatex to build output/thesis.pdf locally, or upload output/latex_project.zip to Overleaf."
        )
        return None

    attempted: list[str] = []
    failures: list[str] = []
    project_pdf = project_dir / "main.pdf"
    _remove_existing_artifact(project_pdf)

    for compiler in compilers:
        attempted.append(compiler)
        try:
            subprocess.run(
                _compiler_command(compiler),
                cwd=str(project_dir),
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            details = (exc.stderr or exc.stdout or "").strip()
            if details:
                failures.append(f"{compiler}: {details}")
            else:
                failures.append(f"{compiler}: exited with status {exc.returncode}")
            continue

        if project_pdf.is_file():
            shutil.copy2(project_pdf, pdf_output_path)
            return pdf_output_path

        failures.append(f"{compiler}: completed without producing {project_pdf.name}")

    warning = (
        "Local PDF compilation failed after attempting "
        f"{', '.join(attempted)}. Project directory and zip archive were still generated. "
        "Upload output/latex_project.zip to Overleaf or inspect the TeX toolchain locally."
    )
    if failures:
        warning = f"{warning} Details: {' | '.join(failures)}"
    warnings.append(warning)
    return None


def export_latex_project(state: dict[str, Any], store: Any, run_id: str) -> str:
    _validate_template_assets(TEMPLATE_ROOT)

    project_dir = Path(store.artifact_path(run_id, "output/latex_project"))
    _remove_existing_artifact(project_dir)

    shutil.copytree(TEMPLATE_ROOT, project_dir)
    warnings: list[str] = []
    _write_rendered_files(project_dir, state, warnings, store=store)
    zip_path = _write_project_archive(project_dir, store=store, run_id=run_id)
    pdf_path = _compile_project_pdf(project_dir, store=store, run_id=run_id, warnings=warnings)
    state["export_artifacts"] = {
        "latex_project_dir": str(project_dir),
        "latex_zip_path": str(zip_path),
        "pdf_path": str(pdf_path) if pdf_path is not None else None,
    }
    state["export_warnings"] = warnings
    return str(project_dir)
