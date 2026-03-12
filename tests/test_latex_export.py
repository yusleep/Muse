import os
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from muse.schemas import new_thesis_state
from muse.store import RunStore


class LatexExportTests(unittest.TestCase):
    def test_export_latex_project_scaffolds_required_template_layout(self):
        from muse.latex_export import export_latex_project

        with tempfile.TemporaryDirectory() as tmp:
            store = RunStore(base_dir=tmp)
            run_id = store.create_run(topic="topic")
            state = new_thesis_state(
                project_id=run_id,
                topic="topic",
                discipline="cs",
                language="zh",
                format_standard="GB/T 7714-2015",
            )

            project_dir = export_latex_project(state, store, run_id)

            self.assertTrue(os.path.isdir(project_dir))
            self.assertTrue(os.path.isfile(os.path.join(project_dir, "main.tex")))
            for dirname in ("Bib", "Chapter", "config", "resources"):
                self.assertTrue(os.path.isdir(os.path.join(project_dir, dirname)))

    def test_export_latex_project_recreates_existing_project_directory(self):
        from muse.latex_export import export_latex_project

        with tempfile.TemporaryDirectory() as tmp:
            store = RunStore(base_dir=tmp)
            run_id = store.create_run(topic="topic")
            state = new_thesis_state(
                project_id=run_id,
                topic="topic",
                discipline="cs",
                language="zh",
                format_standard="GB/T 7714-2015",
            )

            project_dir = export_latex_project(state, store, run_id)
            stale_path = os.path.join(project_dir, "stale.txt")
            with open(stale_path, "w", encoding="utf-8") as fh:
                fh.write("obsolete")

            rebuilt_dir = export_latex_project(state, store, run_id)

            self.assertEqual(project_dir, rebuilt_dir)
            self.assertFalse(os.path.exists(stale_path))
            self.assertTrue(os.path.isfile(os.path.join(rebuilt_dir, "main.tex")))

    def test_export_latex_project_fails_fast_when_required_assets_are_missing(self):
        from muse import latex_export

        with tempfile.TemporaryDirectory() as tmp:
            store = RunStore(base_dir=tmp)
            run_id = store.create_run(topic="topic")
            state = new_thesis_state(
                project_id=run_id,
                topic="topic",
                discipline="cs",
                language="zh",
                format_standard="GB/T 7714-2015",
            )
            template_root = Path(tmp) / "template"
            template_root.mkdir(parents=True, exist_ok=True)
            (template_root / "main.tex").write_text("test", encoding="utf-8")

            with patch.object(latex_export, "TEMPLATE_ROOT", template_root):
                with self.assertRaisesRegex(RuntimeError, "Missing LaTeX template assets:.*Bib.*Chapter"):
                    latex_export.export_latex_project(state, store, run_id)

    def test_export_latex_project_renders_metadata_and_abstracts(self):
        from muse.latex_export import export_latex_project

        with tempfile.TemporaryDirectory() as tmp:
            store = RunStore(base_dir=tmp)
            run_id = store.create_run(topic="topic")
            state = new_thesis_state(
                project_id=run_id,
                topic="分布式系统容错研究",
                discipline="计算机科学与技术",
                language="zh",
                format_standard="GB/T 7714-2015",
            )
            state.update(
                {
                    "title_zh": "分布式系统容错研究",
                    "title_en": "Fault Tolerance in Distributed Systems",
                    "author_name": "张三",
                    "student_id": "2025000123",
                    "discipline_name": "计算机科学与技术",
                    "supervisor_name": "李四教授",
                    "graduation_date": "2026年6月",
                    "abstract_zh": "本文研究分布式系统中的容错机制。",
                    "keywords_zh": ["分布式系统", "容错"],
                    "abstract_en": "This thesis studies fault tolerance in distributed systems.",
                    "keywords_en": ["distributed systems", "fault tolerance"],
                }
            )

            project_dir = export_latex_project(state, store, run_id)

            info_text = Path(project_dir, "config", "info.tex").read_text(encoding="utf-8")
            self.assertIn(r"\newcommand{\thesistitlezh}{分布式系统容错研究}", info_text)
            self.assertIn(r"\newcommand{\thesistitleen}{Fault Tolerance in Distributed Systems}", info_text)
            self.assertIn(r"\newcommand{\authorname}{张三}", info_text)
            self.assertIn(r"\newcommand{\studentid}{2025000123}", info_text)
            self.assertIn(r"\newcommand{\disciplinename}{计算机科学与技术}", info_text)
            self.assertIn(r"\newcommand{\supervisorname}{李四教授}", info_text)
            self.assertIn(r"\newcommand{\graduatedate}{2026年6月}", info_text)
            self.assertIn(r"\newcommand{\keywordszh}{分布式系统；容错}", info_text)
            self.assertIn(r"\newcommand{\keywordsen}{distributed systems; fault tolerance}", info_text)

            abstract_zh = Path(project_dir, "Chapter", "abstract_zh.tex").read_text(encoding="utf-8")
            abstract_en = Path(project_dir, "Chapter", "abstract_en.tex").read_text(encoding="utf-8")
            self.assertIn("本文研究分布式系统中的容错机制。", abstract_zh)
            self.assertIn("关键词：分布式系统；容错", abstract_zh)
            self.assertIn("This thesis studies fault tolerance in distributed systems.", abstract_en)
            self.assertIn("Keywords: distributed systems; fault tolerance", abstract_en)

    def test_export_latex_project_writes_per_chapter_tex_files_and_updates_main(self):
        from muse.latex_export import export_latex_project

        with tempfile.TemporaryDirectory() as tmp:
            store = RunStore(base_dir=tmp)
            run_id = store.create_run(topic="topic")
            state = new_thesis_state(
                project_id=run_id,
                topic="topic",
                discipline="cs",
                language="zh",
                format_standard="GB/T 7714-2015",
            )
            state["chapter_results"] = [
                {
                    "chapter_id": "ch_01",
                    "chapter_title": "绪论",
                    "merged_text": "# 研究背景\n\n背景内容。\n\n## 研究问题\n\n问题内容。\n\n### 研究贡献\n\n贡献内容。",
                },
                {
                    "chapter_id": "ch_02",
                    "chapter_title": "系统设计",
                    "merged_text": "# 总体架构\n\n架构说明。",
                },
            ]

            project_dir = export_latex_project(state, store, run_id)

            chapter1 = Path(project_dir, "Chapter", "chapter1.tex").read_text(encoding="utf-8")
            chapter2 = Path(project_dir, "Chapter", "chapter2.tex").read_text(encoding="utf-8")
            main_tex = Path(project_dir, "main.tex").read_text(encoding="utf-8")

            self.assertIn(r"\chapter{绪论}", chapter1)
            self.assertIn(r"\section{研究背景}", chapter1)
            self.assertIn(r"\subsection{研究问题}", chapter1)
            self.assertIn(r"\subsubsection{研究贡献}", chapter1)
            self.assertIn("背景内容。", chapter1)
            self.assertIn("问题内容。", chapter1)
            self.assertIn("贡献内容。", chapter1)

            self.assertIn(r"\chapter{系统设计}", chapter2)
            self.assertIn(r"\section{总体架构}", chapter2)
            self.assertIn("架构说明。", chapter2)

            self.assertIn(r"\input{Chapter/chapter1}", main_tex)
            self.assertIn(r"\input{Chapter/chapter2}", main_tex)
            self.assertNotIn("这是一个占位章节", chapter1)

    def test_export_latex_project_includes_math_packages_for_common_formulas(self):
        from muse.latex_export import export_latex_project

        with tempfile.TemporaryDirectory() as tmp:
            store = RunStore(base_dir=tmp)
            run_id = store.create_run(topic="topic")
            state = new_thesis_state(
                project_id=run_id,
                topic="topic",
                discipline="cs",
                language="zh",
                format_standard="GB/T 7714-2015",
            )
            state["chapter_results"] = [
                {
                    "chapter_id": "ch_01",
                    "chapter_title": "理论分析",
                    "merged_text": "# 组合公式\n\n组合项写作 \\(\\binom{n}{k}\\)。",
                }
            ]

            project_dir = export_latex_project(state, store, run_id)

            main_tex = Path(project_dir, "main.tex").read_text(encoding="utf-8")
            self.assertIn(r"\usepackage{amsmath}", main_tex)
            self.assertIn(r"\usepackage{amssymb}", main_tex)

    def test_export_latex_project_preserves_math_and_existing_latex_commands(self):
        from muse.latex_export import export_latex_project

        with tempfile.TemporaryDirectory() as tmp:
            store = RunStore(base_dir=tmp)
            run_id = store.create_run(topic="topic")
            state = new_thesis_state(
                project_id=run_id,
                topic="topic",
                discipline="cs",
                language="zh",
                format_standard="GB/T 7714-2015",
            )
            state["chapter_results"] = [
                {
                    "chapter_id": "ch_01",
                    "chapter_title": "理论分析",
                    "merged_text": "# 分析\n\n能量公式为 $E=mc^2$，并参考 \\cite{ref_complete}。",
                }
            ]
            state["references"] = [
                {
                    "ref_id": "@ref_complete",
                    "title": "Complete Reference",
                    "authors": ["Alice Zhang"],
                    "year": 2024,
                    "doi": "10.1000/complete",
                    "venue": "Journal of Testing",
                    "abstract": "Complete reference abstract.",
                    "source": "semantic_scholar",
                    "verified_metadata": True,
                }
            ]
            state["citation_uses"] = [
                {"cite_key": "@ref_complete", "claim_id": "c1", "chapter_id": "ch_01", "subtask_id": "s1"},
            ]

            project_dir = export_latex_project(state, store, run_id)

            chapter1 = Path(project_dir, "Chapter", "chapter1.tex").read_text(encoding="utf-8")
            self.assertIn(r"$E=mc^2$", chapter1)
            self.assertIn(r"\cite{ref_complete}", chapter1)
            self.assertNotIn(r"\$E=mc\textasciicircum{}2\$", chapter1)

    def test_export_latex_project_preserves_parenthesized_math_delimiters(self):
        from muse.latex_export import export_latex_project

        with tempfile.TemporaryDirectory() as tmp:
            store = RunStore(base_dir=tmp)
            run_id = store.create_run(topic="topic")
            state = new_thesis_state(
                project_id=run_id,
                topic="topic",
                discipline="cs",
                language="zh",
                format_standard="GB/T 7714-2015",
            )
            state["chapter_results"] = [
                {
                    "chapter_id": "ch_01",
                    "chapter_title": "理论分析",
                    "merged_text": (
                        "# 模型\n\n"
                        "节点总数定义为 \\(N=|V|\\)，单片攻陷概率写作 "
                        "\\(P_c=\\frac{\\binom{B}{x}\\binom{N-B}{m-x}}{\\binom{N}{m}}\\)。\n\n"
                        "展示公式为 \\[P_{atk}=P(E_s \\cup E_c)\\]。"
                    ),
                }
            ]

            project_dir = export_latex_project(state, store, run_id)

            chapter1 = Path(project_dir, "Chapter", "chapter1.tex").read_text(encoding="utf-8")
            self.assertIn(r"\(N=|V|\)", chapter1)
            self.assertIn(r"\(P_c=\frac{\binom{B}{x}\binom{N-B}{m-x}}{\binom{N}{m}}\)", chapter1)
            self.assertIn(r"\[P_{atk}=P(E_s \cup E_c)\]", chapter1)
            self.assertNotIn(r"\textbackslash\{\}(N", chapter1)

    def test_export_latex_project_restores_json_escaped_latex_control_sequences(self):
        from muse.latex_export import export_latex_project

        with tempfile.TemporaryDirectory() as tmp:
            store = RunStore(base_dir=tmp)
            run_id = store.create_run(topic="topic")
            state = new_thesis_state(
                project_id=run_id,
                topic="topic",
                discipline="cs",
                language="zh",
                format_standard="GB/T 7714-2015",
            )
            state["chapter_results"] = [
                {
                    "chapter_id": "ch_01",
                    "chapter_title": "理论分析",
                    "merged_text": (
                        "# 模型\n\n"
                        "单片攻陷概率为 \\(P="
                        + chr(12)
                        + "rac{1}{2}\\)，恶意比例为 \\("
                        + chr(13)
                        + "ho=b/N\\)，并满足 \\("
                        + chr(12)
                        + "orall x\\in X\\)。"
                    ),
                }
            ]

            project_dir = export_latex_project(state, store, run_id)

            chapter1 = Path(project_dir, "Chapter", "chapter1.tex").read_text(encoding="utf-8")
            self.assertIn(r"\(P=\frac{1}{2}\)", chapter1)
            self.assertIn(r"\(\rho=b/N\)", chapter1)
            self.assertIn(r"\(\forall x\in X\)", chapter1)
            self.assertNotIn("\x0c", chapter1)
            self.assertNotIn("\r", chapter1)

    def test_export_latex_project_preserves_multiline_display_math_blocks(self):
        from muse.latex_export import export_latex_project

        with tempfile.TemporaryDirectory() as tmp:
            store = RunStore(base_dir=tmp)
            run_id = store.create_run(topic="topic")
            state = new_thesis_state(
                project_id=run_id,
                topic="topic",
                discipline="cs",
                language="zh",
                format_standard="GB/T 7714-2015",
            )
            state["chapter_results"] = [
                {
                    "chapter_id": "ch_01",
                    "chapter_title": "理论分析",
                    "merged_text": (
                        "# 模型\n\n"
                        "超几何分布可写为\n"
                        "\\[\n"
                        "P_{comp}(m,t)=\\sum_{x=t}^{m} \\frac{\\binom{b}{x}\\binom{N-b}{m-x}}{\\binom{N}{m}}\n"
                        "\\]\n"
                        "其中阈值为 \\(t=\\lfloor m/3\\rfloor+1\\)。"
                    ),
                }
            ]

            project_dir = export_latex_project(state, store, run_id)

            chapter1 = Path(project_dir, "Chapter", "chapter1.tex").read_text(encoding="utf-8")
            self.assertIn("\\[\nP_{comp}(m,t)=\\sum_{x=t}^{m} \\frac{\\binom{b}{x}\\binom{N-b}{m-x}}{\\binom{N}{m}}\n\\]", chapter1)
            self.assertNotIn(r"\textbackslash\{\}[", chapter1)

    def test_export_latex_project_normalizes_double_escaped_math_delimiters(self):
        from muse.latex_export import export_latex_project

        with tempfile.TemporaryDirectory() as tmp:
            store = RunStore(base_dir=tmp)
            run_id = store.create_run(topic="topic")
            state = new_thesis_state(
                project_id=run_id,
                topic="topic",
                discipline="cs",
                language="zh",
                format_standard="GB/T 7714-2015",
            )
            state["chapter_results"] = [
                {
                    "chapter_id": "ch_01",
                    "chapter_title": "理论分析",
                    "merged_text": (
                        "# 模型\n\n"
                        "对事务 \\\\(tx\\\\) ，其提交条件为 \\\\(P(E_{tx})\\approx p^2\\\\)。"
                    ),
                }
            ]

            project_dir = export_latex_project(state, store, run_id)

            chapter1 = Path(project_dir, "Chapter", "chapter1.tex").read_text(encoding="utf-8")
            self.assertIn(r"\(tx\)", chapter1)
            self.assertIn(r"\(P(E_{tx})\approx p^2\)", chapter1)
            self.assertNotIn(r"\textbackslash\{\}\(", chapter1)

    def test_export_latex_project_writes_bibliography_and_placeholder_warnings(self):
        from muse.latex_export import export_latex_project

        with tempfile.TemporaryDirectory() as tmp:
            store = RunStore(base_dir=tmp)
            run_id = store.create_run(topic="topic")
            state = new_thesis_state(
                project_id=run_id,
                topic="topic",
                discipline="cs",
                language="zh",
                format_standard="GB/T 7714-2015",
            )
            state["references"] = [
                {
                    "ref_id": "@ref_complete",
                    "title": "Complete Reference",
                    "authors": ["Alice Zhang", "Bob Li"],
                    "year": 2024,
                    "doi": "10.1000/complete",
                    "venue": "Journal of Testing",
                    "abstract": "Complete reference abstract.",
                    "source": "semantic_scholar",
                    "verified_metadata": True,
                },
                {
                    "ref_id": "@ref_partial",
                    "title": "",
                    "authors": [],
                    "year": None,
                    "doi": None,
                    "venue": None,
                    "abstract": None,
                    "source": "local",
                    "verified_metadata": False,
                },
            ]
            state["citation_uses"] = [
                {"cite_key": "@ref_complete", "claim_id": "c1", "chapter_id": "ch_01", "subtask_id": "s1"},
                {"cite_key": "@ref_partial", "claim_id": "c2", "chapter_id": "ch_01", "subtask_id": "s1"},
                {"cite_key": "@ref_missing", "claim_id": "c3", "chapter_id": "ch_01", "subtask_id": "s1"},
            ]

            project_dir = export_latex_project(state, store, run_id)

            bib_text = Path(project_dir, "Bib", "thesis.bib").read_text(encoding="utf-8")
            main_text = Path(project_dir, "main.tex").read_text(encoding="utf-8")
            self.assertIn("@article{ref_complete,", bib_text)
            self.assertIn("title = {Complete Reference}", bib_text)
            self.assertIn("author = {Alice Zhang and Bob Li}", bib_text)
            self.assertIn("journal = {Journal of Testing}", bib_text)
            self.assertIn("doi = {10.1000/complete}", bib_text)
            self.assertIn("@misc{ref_partial,", bib_text)
            self.assertIn(r"title = {Untitled reference ref\_partial}", bib_text)
            self.assertIn("note = {Placeholder bibliography entry generated from incomplete metadata.}", bib_text)
            self.assertIn("@misc{ref_missing,", bib_text)
            self.assertIn(r"title = {Missing reference metadata for ref\_missing}", bib_text)
            self.assertIn(r"\nocite{ref_complete,ref_partial,ref_missing}", main_text)

            self.assertIn("export_warnings", state)
            self.assertTrue(any("ref_partial" in warning for warning in state["export_warnings"]))
            self.assertTrue(any("ref_missing" in warning for warning in state["export_warnings"]))

    def test_export_latex_project_copies_markdown_images_and_rewrites_paths(self):
        from muse.latex_export import export_latex_project

        with tempfile.TemporaryDirectory() as tmp:
            store = RunStore(base_dir=tmp)
            run_id = store.create_run(topic="topic")
            state = new_thesis_state(
                project_id=run_id,
                topic="topic",
                discipline="cs",
                language="zh",
                format_standard="GB/T 7714-2015",
            )

            source_dir = Path(tmp) / "source_assets"
            source_dir.mkdir(parents=True, exist_ok=True)
            image_path = source_dir / "系统 架构图.png"
            image_path.write_bytes(b"fake-png")

            state["chapter_results"] = [
                {
                    "chapter_id": "ch_01",
                    "chapter_title": "系统设计",
                    "merged_text": f"# 总体架构\n\n![系统架构]({image_path})\n\n图示说明。",
                }
            ]

            project_dir = export_latex_project(state, store, run_id)

            chapter_text = Path(project_dir, "Chapter", "chapter1.tex").read_text(encoding="utf-8")
            self.assertIn(r"\includegraphics[width=0.9\textwidth]{resources/generated_assets/asset_1.png}", chapter_text)
            self.assertNotIn(str(image_path), chapter_text)

            copied_asset = Path(project_dir, "resources", "generated_assets", "asset_1.png")
            self.assertTrue(copied_asset.is_file())
            self.assertEqual(copied_asset.read_bytes(), b"fake-png")

    def test_export_latex_project_resolves_relative_image_paths_from_store_context(self):
        from muse.latex_export import export_latex_project

        with tempfile.TemporaryDirectory() as tmp:
            runs_dir = Path(tmp) / "runs"
            store = RunStore(base_dir=str(runs_dir))
            run_id = store.create_run(topic="topic")
            state = new_thesis_state(
                project_id=run_id,
                topic="topic",
                discipline="cs",
                language="zh",
                format_standard="GB/T 7714-2015",
            )

            asset_dir = Path(tmp) / "assets"
            asset_dir.mkdir(parents=True, exist_ok=True)
            image_path = asset_dir / "fig.png"
            image_path.write_bytes(b"relative-png")

            state["chapter_results"] = [
                {
                    "chapter_id": "ch_01",
                    "chapter_title": "系统设计",
                    "merged_text": "# 总体架构\n\n![系统架构](assets/fig.png)\n\n图示说明。",
                }
            ]

            project_dir = export_latex_project(state, store, run_id)

            chapter_text = Path(project_dir, "Chapter", "chapter1.tex").read_text(encoding="utf-8")
            self.assertIn(r"\includegraphics[width=0.9\textwidth]{resources/generated_assets/asset_1.png}", chapter_text)
            self.assertNotIn("![系统架构](assets/fig.png)", chapter_text)

            copied_asset = Path(project_dir, "resources", "generated_assets", "asset_1.png")
            self.assertTrue(copied_asset.is_file())
            self.assertEqual(copied_asset.read_bytes(), b"relative-png")

    def test_export_latex_project_writes_overleaf_zip_archive(self):
        from muse.latex_export import export_latex_project

        with tempfile.TemporaryDirectory() as tmp:
            store = RunStore(base_dir=tmp)
            run_id = store.create_run(topic="topic")
            state = new_thesis_state(
                project_id=run_id,
                topic="topic",
                discipline="cs",
                language="zh",
                format_standard="GB/T 7714-2015",
            )
            state["chapter_results"] = [
                {
                    "chapter_id": "ch_01",
                    "chapter_title": "绪论",
                    "merged_text": "# 背景\n\n内容。",
                }
            ]

            with patch("muse.latex_export.shutil.which", return_value=None):
                project_dir = export_latex_project(state, store, run_id)

            self.assertEqual(project_dir, state["export_artifacts"]["latex_project_dir"])
            zip_path = state["export_artifacts"]["latex_zip_path"]
            self.assertTrue(os.path.isfile(zip_path))

            with zipfile.ZipFile(zip_path) as archive:
                members = set(archive.namelist())

            self.assertIn("main.tex", members)
            self.assertIn("Bib/thesis.bib", members)
            self.assertIn("config/info.tex", members)
            self.assertIn("Chapter/chapter1.tex", members)

    def test_export_latex_project_warns_when_tex_tooling_is_unavailable(self):
        from muse.latex_export import export_latex_project

        with tempfile.TemporaryDirectory() as tmp:
            store = RunStore(base_dir=tmp)
            run_id = store.create_run(topic="topic")
            state = new_thesis_state(
                project_id=run_id,
                topic="topic",
                discipline="cs",
                language="zh",
                format_standard="GB/T 7714-2015",
            )

            with patch("muse.latex_export.shutil.which", return_value=None):
                project_dir = export_latex_project(state, store, run_id)

            self.assertTrue(os.path.isdir(project_dir))
            self.assertTrue(os.path.isfile(state["export_artifacts"]["latex_zip_path"]))
            self.assertIsNone(state["export_artifacts"]["pdf_path"])
            self.assertTrue(any("latexmk or xelatex" in warning for warning in state["export_warnings"]))

    def test_export_latex_project_compiles_pdf_with_latexmk_when_available(self):
        from muse.latex_export import export_latex_project

        with tempfile.TemporaryDirectory() as tmp:
            store = RunStore(base_dir=tmp)
            run_id = store.create_run(topic="topic")
            state = new_thesis_state(
                project_id=run_id,
                topic="topic",
                discipline="cs",
                language="zh",
                format_standard="GB/T 7714-2015",
            )
            calls: list[tuple[list[str], str]] = []

            def fake_run(cmd, cwd=None, check=None, capture_output=None, text=None):
                calls.append((list(cmd), cwd))
                Path(cwd, "main.pdf").write_bytes(b"%PDF-1.7 fake")
                return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

            def fake_which(name: str):
                if name == "latexmk":
                    return "/usr/bin/latexmk"
                if name == "xelatex":
                    return "/usr/bin/xelatex"
                return None

            with patch("muse.latex_export.shutil.which", side_effect=fake_which):
                with patch("muse.latex_export.subprocess.run", side_effect=fake_run):
                    export_latex_project(state, store, run_id)

            self.assertTrue(calls)
            self.assertEqual(calls[0][0][0], "latexmk")
            self.assertIn("-xelatex", calls[0][0])
            self.assertEqual(calls[0][1], state["export_artifacts"]["latex_project_dir"])
            pdf_path = state["export_artifacts"]["pdf_path"]
            self.assertTrue(os.path.isfile(pdf_path))
            self.assertEqual(Path(pdf_path).read_bytes(), b"%PDF-1.7 fake")
            self.assertFalse(state["export_warnings"])

    def test_export_latex_project_warns_when_pdf_compile_fails(self):
        from muse.latex_export import export_latex_project

        with tempfile.TemporaryDirectory() as tmp:
            store = RunStore(base_dir=tmp)
            run_id = store.create_run(topic="topic")
            state = new_thesis_state(
                project_id=run_id,
                topic="topic",
                discipline="cs",
                language="zh",
                format_standard="GB/T 7714-2015",
            )

            failure = subprocess.CalledProcessError(
                1,
                ["latexmk", "-xelatex", "main.tex"],
                output="",
                stderr="Undefined control sequence",
            )

            with patch("muse.latex_export.shutil.which", side_effect=lambda name: "/usr/bin/latexmk" if name == "latexmk" else None):
                with patch("muse.latex_export.subprocess.run", side_effect=failure):
                    project_dir = export_latex_project(state, store, run_id)

            self.assertTrue(os.path.isdir(project_dir))
            self.assertTrue(os.path.isfile(state["export_artifacts"]["latex_zip_path"]))
            self.assertIsNone(state["export_artifacts"]["pdf_path"])
            self.assertTrue(any("latexmk" in warning for warning in state["export_warnings"]))
            self.assertTrue(any("Undefined control sequence" in warning for warning in state["export_warnings"]))


if __name__ == "__main__":
    unittest.main()
