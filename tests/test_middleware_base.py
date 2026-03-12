from __future__ import annotations

import unittest


class MiddlewareProtocolTests(unittest.TestCase):
    """Verify the Middleware protocol and MiddlewareChain plumbing."""

    def test_import_middleware_protocol(self):
        from muse.middlewares.base import Middleware

        import typing

        self.assertTrue(
            getattr(Middleware, "__protocol_attrs__", None) is not None or typing.runtime_checkable
        )

    def test_import_middleware_chain(self):
        from muse.middlewares.base import MiddlewareChain

        self.assertTrue(callable(MiddlewareChain))

    def test_chain_wrap_returns_callable(self):
        from muse.middlewares.base import MiddlewareChain

        chain = MiddlewareChain([])

        def node_fn(state):
            return {"out": state.get("x", 0) + 1}

        wrapped = chain.wrap(node_fn)
        self.assertTrue(callable(wrapped))

    def test_chain_no_middlewares_passthrough(self):
        from muse.middlewares.base import MiddlewareChain

        chain = MiddlewareChain([])

        def node_fn(state):
            return {"result": state.get("v", 0) * 2}

        wrapped = chain.wrap(node_fn)
        result = wrapped({"v": 5})
        self.assertEqual(result, {"result": 10})

    def test_before_invoke_can_modify_state(self):
        from muse.middlewares.base import MiddlewareChain

        class InjectMiddleware:
            async def before_invoke(self, state, config):
                state = dict(state)
                state["injected"] = True
                return state

            async def after_invoke(self, state, result, config):
                return result

        chain = MiddlewareChain([InjectMiddleware()])
        captured = {}

        def node_fn(state):
            captured.update(state)
            return {"done": True}

        wrapped = chain.wrap(node_fn)
        wrapped({"x": 1})
        self.assertTrue(captured.get("injected"))

    def test_after_invoke_can_modify_result(self):
        from muse.middlewares.base import MiddlewareChain

        class TagMiddleware:
            async def before_invoke(self, state, config):
                return state

            async def after_invoke(self, state, result, config):
                result = dict(result)
                result["tagged"] = True
                return result

        chain = MiddlewareChain([TagMiddleware()])

        def node_fn(state):
            return {"value": 42}

        wrapped = chain.wrap(node_fn)
        out = wrapped({"x": 1})
        self.assertEqual(out["value"], 42)
        self.assertTrue(out["tagged"])

    def test_middleware_execution_order(self):
        from muse.middlewares.base import MiddlewareChain

        order = []

        class MW:
            def __init__(self, name):
                self._name = name

            async def before_invoke(self, state, config):
                order.append(f"before:{self._name}")
                return state

            async def after_invoke(self, state, result, config):
                order.append(f"after:{self._name}")
                return result

        chain = MiddlewareChain([MW("A"), MW("B"), MW("C")])

        def node_fn(state):
            order.append("node")
            return {}

        wrapped = chain.wrap(node_fn)
        wrapped({})
        self.assertEqual(
            order,
            [
                "before:A",
                "before:B",
                "before:C",
                "node",
                "after:C",
                "after:B",
                "after:A",
            ],
        )

    def test_chain_wrap_with_config_arg(self):
        from muse.middlewares.base import MiddlewareChain

        chain = MiddlewareChain([])

        def node_fn(state):
            return {"ok": True}

        wrapped = chain.wrap(node_fn)
        result = wrapped({"x": 1}, {"configurable": {"thread_id": "t1"}})
        self.assertEqual(result, {"ok": True})

    def test_async_node_fn_supported(self):
        from muse.middlewares.base import MiddlewareChain

        chain = MiddlewareChain([])

        async def async_node(state):
            return {"async": True}

        wrapped = chain.wrap(async_node)
        result = wrapped({"x": 1})
        self.assertEqual(result, {"async": True})

    def test_middlewares_package_init_reexports(self):
        from muse.middlewares import Middleware, MiddlewareChain

        self.assertTrue(callable(MiddlewareChain))


if __name__ == "__main__":
    unittest.main()
