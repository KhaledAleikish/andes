"""
Tests for the unified logging system.

Verifies that:
- TqdmStreamHandler routes through tqdm.write
- config_logger installs TqdmStreamHandler with optional ColoredFormatter
- set_logger_level updates handler levels
- Timer event messages (Toggle, Fault, Alter) use logger.info
- Convergence diagnostics use logger.debug
- Library usage (no config_logger) is silent at INFO
- No tqdm.write() or verbose guards remain in operational code
"""

import io
import logging
import logging.handlers
import os
import unittest

import numpy as np

import andes
from andes.main import TqdmStreamHandler, config_logger, set_logger_level
from andes.utils.misc import ColoredFormatter, supports_color
from andes.utils.paths import get_case

CASE14 = get_case('ieee14/ieee14_esst3a.xlsx')


# ---------------------------------------------------------------------------
# TqdmStreamHandler unit tests
# ---------------------------------------------------------------------------

class TestTqdmStreamHandler(unittest.TestCase):
    """Unit tests for the TqdmStreamHandler class."""

    def test_is_stream_handler_subclass(self):
        h = TqdmStreamHandler()
        self.assertIsInstance(h, logging.StreamHandler)

    def test_emit_calls_tqdm_write(self):
        """Handler.emit() should delegate to tqdm.write()."""
        from unittest.mock import patch

        buf = io.StringIO()
        h = TqdmStreamHandler(stream=buf)
        h.setFormatter(logging.Formatter("%(message)s"))

        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="hello from handler", args=(), exc_info=None,
        )
        with patch("tqdm.tqdm.write") as mock_write:
            h.emit(record)
            mock_write.assert_called_once()
            written = mock_write.call_args[0][0]
            self.assertEqual(written, "hello from handler")


# ---------------------------------------------------------------------------
# config_logger tests
# ---------------------------------------------------------------------------

class TestConfigLogger(unittest.TestCase):
    """Tests for config_logger setup and level switching."""

    def setUp(self):
        self._lg = logging.getLogger("andes")
        self._orig_handlers = self._lg.handlers[:]
        self._lg.handlers.clear()

    def tearDown(self):
        self._lg.handlers = self._orig_handlers

    def test_installs_stream_handler(self):
        """config_logger should add at least one StreamHandler."""
        config_logger(stream=True, file=False)
        sh = [h for h in self._lg.handlers
              if isinstance(h, logging.StreamHandler)]
        self.assertTrue(len(sh) > 0, "Expected at least one StreamHandler")

    def test_handler_is_tqdm(self):
        """The stream handler should be TqdmStreamHandler."""
        config_logger(stream=True, file=False)
        sh = [h for h in self._lg.handlers
              if isinstance(h, logging.StreamHandler)
              and not isinstance(h, logging.FileHandler)]
        self.assertTrue(len(sh) > 0)
        h = sh[0]
        self.assertIsInstance(h, TqdmStreamHandler,
                              f"Expected TqdmStreamHandler, got {type(h).__name__}")

    def test_set_logger_level_updates_stream(self):
        """set_logger_level should change the level on StreamHandlers."""
        config_logger(stream_level=logging.INFO, stream=True, file=False)
        set_logger_level(self._lg, logging.StreamHandler, logging.WARNING)
        sh = [h for h in self._lg.handlers
              if isinstance(h, logging.StreamHandler)
              and not isinstance(h, logging.FileHandler)]
        self.assertTrue(all(h.level == logging.WARNING for h in sh))

    def test_no_file_handler_when_disabled(self):
        config_logger(stream=True, file=False)
        fh = [h for h in self._lg.handlers
              if isinstance(h, logging.FileHandler)]
        self.assertEqual(len(fh), 0)


# ---------------------------------------------------------------------------
# Integration: Toggle event logging
# ---------------------------------------------------------------------------

class TestToggleLogging(unittest.TestCase):
    """Toggle events should emit logger.info messages."""

    @classmethod
    def setUpClass(cls):
        cls.ss = andes.load(CASE14, default_config=True, no_output=True)
        cls.ss.PFlow.run()
        cls.ss.TDS.init()

    def setUp(self):
        self.ss.TDS.reinit()
        self._logger = logging.getLogger("andes.models.timer")
        self._handler = logging.handlers.MemoryHandler(capacity=200)
        self._handler.setLevel(logging.DEBUG)
        self._logger.addHandler(self._handler)

    def tearDown(self):
        self._logger.removeHandler(self._handler)

    def test_toggle_emits_info(self):
        """Toggle event at t=1.0 should produce an INFO log record."""
        self.ss.TDS.config.tf = 1.05
        self.ss.TDS.run(no_summary=True)

        messages = [r for r in self._handler.buffer
                    if r.levelno == logging.INFO and 'Toggle' in r.getMessage()]
        self.assertTrue(len(messages) > 0,
                        "Expected Toggle INFO message at t=1.0")
        self.assertIn('status changed', messages[0].getMessage())

    def test_toggle_not_at_warning(self):
        """Toggle messages are INFO — a WARNING filter should exclude them."""
        self._handler.setLevel(logging.WARNING)

        self.ss.TDS.config.tf = 1.05
        self.ss.TDS.run(no_summary=True)

        toggle_warnings = [r for r in self._handler.buffer
                           if 'Toggle' in r.getMessage()]
        self.assertEqual(len(toggle_warnings), 0,
                         "Toggle messages should not appear at WARNING level")


# ---------------------------------------------------------------------------
# Integration: convergence diagnostics logging
# ---------------------------------------------------------------------------

class TestConvergenceLogging(unittest.TestCase):
    """Convergence diagnostics should use logger.debug."""

    @classmethod
    def setUpClass(cls):
        cls.ss = andes.load(CASE14, default_config=True, no_output=True)
        cls.ss.PFlow.run()
        cls.ss.TDS.init()

    def setUp(self):
        self.ss.TDS.reinit()
        self._logger = logging.getLogger("andes.routines.daeint")
        self._handler = logging.handlers.MemoryHandler(capacity=2000)
        self._handler.setLevel(logging.DEBUG)
        self._logger.addHandler(self._handler)

    def tearDown(self):
        self._logger.removeHandler(self._handler)

    def test_convergence_emits_debug(self):
        """Successful convergence should log at DEBUG level."""
        self.ss.TDS.config.tf = 0.1
        self.ss.TDS.run(no_summary=True)

        converged = [r for r in self._handler.buffer
                     if 'Converged in' in r.getMessage()]
        self.assertTrue(len(converged) > 0,
                        "Expected 'Converged in' DEBUG messages")
        for rec in converged:
            self.assertEqual(rec.levelno, logging.DEBUG)

    def test_convergence_not_at_info(self):
        """Convergence messages are DEBUG — an INFO filter should exclude them."""
        self._handler.setLevel(logging.INFO)

        self.ss.TDS.config.tf = 0.1
        self.ss.TDS.run(no_summary=True)

        converged = [r for r in self._handler.buffer
                     if 'Converged in' in r.getMessage()]
        self.assertEqual(len(converged), 0,
                         "Convergence messages should not appear at INFO level")


# ---------------------------------------------------------------------------
# NaN / busted path
# ---------------------------------------------------------------------------

class TestNaNBustedLogging(unittest.TestCase):
    """NaN injection should log debug without UnboundLocalError."""

    def test_nan_logs_busted_message(self):
        ss = andes.load(CASE14, default_config=True, no_output=True)
        ss.PFlow.run()
        ss.TDS.init()

        logger = logging.getLogger("andes.routines.daeint")
        handler = logging.handlers.MemoryHandler(capacity=200)
        handler.setLevel(logging.DEBUG)
        logger.addHandler(handler)

        try:
            ss.dae.x[:] = np.nan
            ss.TDS.config.tf = 0.1
            ss.TDS.run(no_summary=True)

            nan_msgs = [r for r in handler.buffer
                        if 'NaN' in r.getMessage()]
            self.assertTrue(len(nan_msgs) > 0,
                            "Expected NaN debug message when solution busts")
        finally:
            logger.removeHandler(handler)


# ---------------------------------------------------------------------------
# Library silence: no config_logger → no INFO output
# ---------------------------------------------------------------------------

class TestLibrarySilence(unittest.TestCase):
    """Library usage without config_logger should not emit INFO messages
    to a WARNING-level handler."""

    def test_no_toggle_at_warning_level(self):
        """With only a WARNING handler, Toggle INFO messages should not appear."""
        andes_logger = logging.getLogger("andes")
        original_handlers = andes_logger.handlers[:]
        andes_logger.handlers.clear()

        capture = logging.handlers.MemoryHandler(capacity=200)
        capture.setLevel(logging.WARNING)
        andes_logger.addHandler(capture)

        try:
            ss = andes.load(CASE14, default_config=True, no_output=True)
            ss.PFlow.run()
            ss.TDS.init()
            ss.TDS.config.tf = 1.5
            ss.TDS.run(no_summary=True)

            toggle_msgs = [r for r in capture.buffer
                           if 'Toggle' in r.getMessage()]
            self.assertEqual(len(toggle_msgs), 0,
                             "Toggle INFO messages should not pass WARNING filter")
        finally:
            andes_logger.removeHandler(capture)
            andes_logger.handlers = original_handlers


# ---------------------------------------------------------------------------
# Source-level regression: no tqdm.write or verbose guards remain
# ---------------------------------------------------------------------------

class TestNoTqdmWriteRemains(unittest.TestCase):
    """Ensure no direct tqdm.write() calls remain in operational code."""

    def _get_source(self, module):
        import inspect
        return inspect.getsource(module)

    def test_no_tqdm_write_in_timer(self):
        from andes.models import timer
        self.assertNotIn('tqdm.write(', self._get_source(timer))

    def test_no_tqdm_write_in_daeint(self):
        from andes.routines import daeint
        self.assertNotIn('tqdm.write(', self._get_source(daeint))

    def test_no_tqdm_write_in_tds(self):
        from andes.routines import tds
        self.assertNotIn('tqdm.write(', self._get_source(tds))

    def test_no_tqdm_write_in_qndf(self):
        from andes.routines import qndf
        self.assertNotIn('tqdm.write(', self._get_source(qndf))

    def test_no_tqdm_write_in_timeseries(self):
        from andes.models import timeseries
        self.assertNotIn('tqdm.write(', self._get_source(timeseries))


class TestNoVerboseGuards(unittest.TestCase):
    """Ensure ad-hoc verbose guards were removed from converted files."""

    def _get_source(self, module):
        import inspect
        return inspect.getsource(module)

    def test_no_verbose_guard_in_timer(self):
        from andes.models import timer
        self.assertNotIn('options.get("verbose"', self._get_source(timer))

    def test_no_verbose_guard_in_daeint(self):
        from andes.routines import daeint
        self.assertNotIn('options.get("verbose"', self._get_source(daeint))

    def test_no_verbose_guard_in_qndf(self):
        from andes.routines import qndf
        self.assertNotIn('options.get("verbose"', self._get_source(qndf))

    def test_no_verbose_guard_in_timeseries(self):
        from andes.models import timeseries
        self.assertNotIn('options.get("verbose"', self._get_source(timeseries))

    def test_verbose_breakpoint_preserved_in_tds(self):
        """The verbose == 1 breakpoint in tds.py should be preserved."""
        from andes.routines import tds
        import inspect
        source = inspect.getsource(tds)
        self.assertIn("verbose') == 1", source,
                      "The verbose==1 debugger breakpoint should be preserved")


# ---------------------------------------------------------------------------
# supports_color and ColoredFormatter
# ---------------------------------------------------------------------------

class TestSupportsColor(unittest.TestCase):
    """Environment-variable and TTY detection for supports_color()."""

    def _clean_env(self):
        """Remove color env vars so tests are isolated."""
        for key in ('FORCE_COLOR', 'NO_COLOR', 'TERM'):
            os.environ.pop(key, None)

    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in ('FORCE_COLOR', 'NO_COLOR', 'TERM')}
        self._clean_env()

    def tearDown(self):
        self._clean_env()
        for k, v in self._saved.items():
            if v is not None:
                os.environ[k] = v

    def test_force_color_overrides(self):
        os.environ['FORCE_COLOR'] = '1'
        self.assertTrue(supports_color(stream=io.StringIO()))

    def test_force_color_empty_string(self):
        os.environ['FORCE_COLOR'] = ''
        self.assertTrue(supports_color(stream=io.StringIO()))

    def test_no_color_disables(self):
        os.environ['NO_COLOR'] = ''
        self.assertFalse(supports_color())

    def test_non_tty_returns_false(self):
        self.assertFalse(supports_color(stream=io.StringIO()))

    def test_dumb_term_returns_false(self):
        os.environ['TERM'] = 'dumb'
        # Need a stream that reports isatty=True
        tty = io.StringIO()
        tty.isatty = lambda: True
        self.assertFalse(supports_color(stream=tty))


class TestColoredFormatter(unittest.TestCase):
    """ColoredFormatter wraps output with ANSI codes."""

    def test_info_is_colored(self):
        fmt = ColoredFormatter('%(message)s')
        record = logging.LogRecord('t', logging.INFO, '', 0, 'hello', (), None)
        result = fmt.format(record)
        self.assertTrue(result.startswith('\033['))
        self.assertIn('hello', result)
        self.assertTrue(result.endswith('\033[0m'))

    def test_original_record_unchanged(self):
        fmt = ColoredFormatter('%(levelname)s %(message)s')
        record = logging.LogRecord('t', logging.WARNING, '', 0, 'msg', (), None)
        fmt.format(record)
        self.assertEqual(record.levelname, 'WARNING')


if __name__ == '__main__':
    unittest.main()
