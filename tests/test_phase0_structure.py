import importlib
import unittest


class Phase0StructureTests(unittest.TestCase):
    def test_services_modules_exist(self):
        for module_name in [
            "muse.services",
            "muse.services.http",
            "muse.services.search",
            "muse.services.citation_meta",
            "muse.services.providers",
            "muse.services.citation",
            "muse.services.store",
            "muse.services.audit",
            "muse.services.planning",
            "muse.services.latex",
            "muse.schemas.run",
            "muse.schemas.reference",
        ]:
            with self.subTest(module_name=module_name):
                importlib.import_module(module_name)

    def test_legacy_shims_preserve_core_symbols(self):
        from muse import audit, citation, latex_export, planning, providers, schemas, store
        from muse.services import (
            CitationMetadataClient,
            HttpClient,
            LLMClient,
            ProviderError,
        )
        from muse.schemas.reference import CitationUse, FlaggedCitation, ReferenceRecord
        from muse.schemas.run import ThesisState, hydrate_thesis_state, new_thesis_state, validate_thesis_state

        self.assertIs(providers.HttpClient, HttpClient)
        self.assertIs(providers.LLMClient, LLMClient)
        self.assertIs(providers.CitationMetadataClient, CitationMetadataClient)
        self.assertIs(providers.ProviderError, ProviderError)

        self.assertIs(citation.verify_all_citations, importlib.import_module("muse.services.citation").verify_all_citations)
        self.assertIs(store.RunStore, importlib.import_module("muse.services.store").RunStore)
        self.assertIs(audit.JsonlAuditSink, importlib.import_module("muse.services.audit").JsonlAuditSink)
        self.assertIs(planning.plan_subtasks, importlib.import_module("muse.services.planning").plan_subtasks)
        self.assertTrue(callable(latex_export.export_latex_project))

        self.assertIs(schemas.ReferenceRecord, ReferenceRecord)
        self.assertIs(schemas.CitationUse, CitationUse)
        self.assertIs(schemas.FlaggedCitation, FlaggedCitation)
        self.assertIs(schemas.ThesisState, ThesisState)
        self.assertIs(schemas.new_thesis_state, new_thesis_state)
        self.assertIs(schemas.hydrate_thesis_state, hydrate_thesis_state)
        self.assertIs(schemas.validate_thesis_state, validate_thesis_state)


if __name__ == "__main__":
    unittest.main()
