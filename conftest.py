"""Test config.

Runs coroutine tests without pytest-asyncio. The plugin's fixture machinery buys us
nothing here -- every async test is self-contained and wants a fresh loop -- so this is
one hook instead of a dependency. ponytail: install pytest-asyncio if async fixtures or
session-scoped loops ever become necessary.
"""
import asyncio
import inspect

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "asyncio: coroutine test, run via asyncio.run")
    config.addinivalue_line(
        "markers", "gpu: needs a live llama-server with the model loaded; deselected by "
                   "default with -m 'not gpu'")


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem):
    if not inspect.iscoroutinefunction(pyfuncitem.obj):
        return None
    kwargs = {name: pyfuncitem.funcargs[name] for name in pyfuncitem._fixtureinfo.argnames}
    asyncio.run(pyfuncitem.obj(**kwargs))
    return True
