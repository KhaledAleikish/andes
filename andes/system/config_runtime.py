"""
System config runtime helpers for System.
"""

#  [ANDES] (C)2015-2026 Hantao Cui
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 3 of the License, or
#  (at your option) any later version.

import configparser
import logging
import os
from collections import OrderedDict

from andes.core import Config
from andes.shared import np
from andes.utils.paths import confirm_overwrite, get_config_path

logger = logging.getLogger(__name__)


class SystemConfigRuntime:
    """
    Manage system-level configuration bootstrapping and persistence.

    Configuration is resolved in four phases, each with higher priority:

    1. ``load_rc()`` — load from ``~/.andes/andes.rc``
    2. ``merge_file_config()`` — merge ``_config`` section from the case file
    3. ``apply_cli_overrides()`` — apply ``config_option`` from the CLI
    4. ``finalize()`` — create ``system.config``, add defaults, validate

    Phases are called in order from :meth:`andes.system.facade.System.__init__`.
    """

    def __init__(self, system):
        self.system = system

    # ------------------------------------------------------------------
    #  Phased config resolution
    # ------------------------------------------------------------------

    def load_rc(self, config_path=None, default_config=False):
        """
        Phase 1: Resolve the rc file path and load it into
        ``system._config_object``.
        """
        system = self.system

        system._config_path = get_config_path()
        if config_path is not None:
            system._config_path = config_path
        if default_config is True:
            system._config_path = None

        system._config_object = self.load_config_rc(system._config_path)

    def merge_file_config(self, files):
        """
        Phase 2: Extract ``_config`` rows from the case file and merge
        them into ``system._config_object``.

        This runs after ``load_rc()`` and before ``apply_cli_overrides()``
        so that file-embedded config overrides andes.rc defaults but is
        itself overridden by CLI ``config_option``.

        Parameters
        ----------
        files : FileMan
            File manager with ``case`` path and ``input_format`` resolved.
        """
        system = self.system

        if files.case is None:
            return

        rows = self._extract_config(files)
        if not rows:
            return

        if system._config_object is None:
            system._config_object = configparser.ConfigParser()

        for row in rows:
            section = str(row.get('section', '')).strip()
            key = str(row.get('key', '')).strip()
            value = str(row.get('value', '')).strip()

            if not section or not key:
                logger.warning("Skipping malformed _config row: %s", row)
                continue

            if not system._config_object.has_section(section):
                system._config_object.add_section(section)

            system._config_object.set(section, key, value)
            logger.debug("File config set: %s.%s=%s", section, key, value)

    def apply_cli_overrides(self):
        """
        Phase 3: Apply ``config_option`` from the command line into
        ``system._config_object``.

        CLI overrides have the highest priority and win over both
        andes.rc and file-embedded config.
        """
        system = self.system
        config_option = system.options.get('config_option', None)
        if config_option is None:
            return

        if len(config_option) == 0:
            return

        if system._config_object is None:
            system._config_object = configparser.ConfigParser()

        for item in config_option:

            # check the validity of the config field
            # each field follows the format `SECTION.FIELD = VALUE`

            if item.count('=') != 1:
                raise ValueError('config_option "{}" must be an assignment expression'.format(item))

            field, value = item.split("=")

            if field.count('.') != 1:
                raise ValueError('config_option left-hand side "{}" must use format SECTION.FIELD'.format(field))

            section, key = field.split(".")

            section = section.strip()
            key = key.strip()
            value = value.strip()

            if not system._config_object.has_section(section):
                system._config_object.add_section(section)

            system._config_object.set(section, key, value)
            logger.debug("CLI config option set: %s.%s=%s", section, key, value)

    def finalize(self, config=None):
        """
        Phase 4: Create ``system.config`` and ``system.runtime`` from
        the resolved ``_config_object``, add defaults, and validate.

        ``system.config`` holds case-relevant settings (freq, mva, etc.)
        and is written to data files.  ``system.runtime`` holds
        machine/environment settings (numba, sparselib, dime, etc.)
        and is persisted only in rc files.
        """
        system = self.system

        # --- Case config (section: "System") ---
        system.config = Config(system.__class__.__name__, dct=config)
        _runtime_hint = 'Use [Runtime] section instead.'
        system.config._deprecated.update({
            'warn_limits': '',
            # Fields moved to [Runtime] in v2.0
            'numba': _runtime_hint, 'numba_parallel': _runtime_hint,
            'numba_nopython': _runtime_hint,
            'yapf_pycode': _runtime_hint, 'save_stats': _runtime_hint,
            'ipadd': _runtime_hint, 'sparselib': _runtime_hint,
            'seed': _runtime_hint, 'np_divide': _runtime_hint,
            'np_invalid': _runtime_hint,
            'dime_enabled': _runtime_hint, 'dime_name': _runtime_hint,
            'dime_address': _runtime_hint,
        })
        system.config.load(system._config_object)
        self._add_case_defaults()
        system.config.check()

        # --- Runtime config (section: "Runtime") ---
        system.runtime = Config('Runtime')
        system.runtime.load(system._config_object)
        self._add_runtime_defaults()
        system.runtime.check()

        self.configure_numpy(
            seed=system.runtime.seed,
            divide=system.runtime.np_divide,
            invalid=system.runtime.np_invalid,
        )

    # ------------------------------------------------------------------
    #  Deprecated alias
    # ------------------------------------------------------------------

    def update_config_object(self):
        """
        .. deprecated:: 2.0
            Use ``apply_cli_overrides()`` instead.
        """
        return self.apply_cli_overrides()

    # ------------------------------------------------------------------
    #  Config extraction from case files
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_config(files):
        """
        Extract ``_config`` rows from the case file.

        Returns a list of dicts with keys ``section``, ``key``, ``value``.
        Returns an empty list for formats that do not support embedded
        config (e.g. PSS/E, MATPOWER) or when no ``_config`` is present.
        """
        from andes.io import input_formats

        case = files.case
        if case is None:
            return []

        # Determine format from extension if not already set
        fmt = files.input_format
        if not fmt:
            ext = os.path.splitext(case)[1].strip('.').lower()
            for key, exts in input_formats.items():
                if ext in exts:
                    fmt = key
                    break

        if fmt == 'xlsx':
            return _extract_config_xlsx(case)
        elif fmt == 'json':
            return _extract_config_json(case)
        else:
            # PSS/E, MATPOWER, etc. — no _config support
            return []

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------

    def _add_case_defaults(self):
        """
        Add default case-relevant config entries (written to data files).
        """
        system = self.system
        system.config.add(OrderedDict((('freq', 60),
                                       ('mva', 100),
                                       ('diag_eps', 1e-8),
                                       ('warn_abnormal', 1),
                                       )))
        system.config.add_extra("_help",
                                freq='base frequency [Hz]',
                                mva='system base MVA',
                                diag_eps='small value for Jacobian diagonals',
                                warn_abnormal='warn initialization out of normal values',
                                )
        system.config.add_extra("_alt",
                                freq="float",
                                mva="float",
                                warn_abnormal=(0, 1),
                                )

    def _add_runtime_defaults(self):
        """
        Add default machine/environment config entries (rc files only).
        """
        system = self.system
        system.runtime.add(OrderedDict((('numba', 0),
                                        ('numba_parallel', 0),
                                        ('numba_nopython', 1),
                                        ('yapf_pycode', 0),
                                        ('save_stats', 0),
                                        ('ipadd', 1),
                                        ('sparselib', 'klu'),
                                        ('seed', 'None'),
                                        ('np_divide', 'warn'),
                                        ('np_invalid', 'warn'),
                                        ('dime_enabled', 0),
                                        ('dime_name', 'andes'),
                                        ('dime_address', 'ipc:///tmp/dime2'),
                                        )))
        system.runtime.add_extra("_help",
                                 numba='use numba for JIT compilation',
                                 numba_parallel='enable parallel for numba.jit',
                                 numba_nopython='nopython mode for numba',
                                 yapf_pycode='format generated code with yapf',
                                 save_stats='store statistics of function calls',
                                 ipadd='use spmatrix.ipadd if available',
                                 sparselib='linear sparse solver name',
                                 seed='seed (or None) for random number generator',
                                 np_divide='treatment for division by zero',
                                 np_invalid='treatment for invalid floating-point ops.',
                                 dime_enabled='enable DiME streaming',
                                 dime_name='DiME client name',
                                 dime_address='DiME server address',
                                 )
        system.runtime.add_extra("_alt",
                                 numba=(0, 1),
                                 numba_parallel=(0, 1),
                                 numba_nopython=(0, 1),
                                 yapf_pycode=(0, 1),
                                 save_stats=(0, 1),
                                 ipadd=(0, 1),
                                 sparselib=("klu", "umfpack", "spsolve", "cupy"),
                                 seed='int or None',
                                 np_divide={'ignore', 'warn', 'raise', 'call', 'print', 'log'},
                                 np_invalid={'ignore', 'warn', 'raise', 'call', 'print', 'log'},
                                 )

    def set_config(self, config=None):
        """
        Set configuration for the System object.

        Config for models are routines are passed directly to their
        constructors.
        """
        system = self.system
        if config is not None:
            if system.__class__.__name__ in config:
                system.config.add(config[system.__class__.__name__])
                logger.debug("Config: set for System")
            if 'Runtime' in config:
                system.runtime.add(config['Runtime'])
                logger.debug("Config: set for Runtime")

    def collect_config(self):
        """
        Collect config data from models into a ``ConfigParser``.

        Returns
        -------
        configparser.ConfigParser
            Sections are class names, values are config dicts.
        """
        system = self.system
        config_dict = configparser.ConfigParser()
        config_dict[system.__class__.__name__] = system.config.as_dict(refresh=True)
        config_dict['Runtime'] = system.runtime.as_dict(refresh=True)

        all_with_config = OrderedDict(list(system.routines.items()) +
                                      list(system.models.items()))

        for name, instance in all_with_config.items():
            cfg = instance.config.as_dict(refresh=True)
            if len(cfg) > 0:
                config_dict[name] = cfg
        return config_dict

    def collect_config_rows(self):
        """
        Collect all config values as a flat list of row dicts for
        serialization to ``_config`` sheets/keys.

        Returns
        -------
        list of dict
            Each dict has keys ``section``, ``key``, ``value``.
        """
        system = self.system
        rows = []

        for key, val in system.config.as_dict(refresh=True).items():
            rows.append({'section': system.__class__.__name__,
                         'key': key, 'value': val})

        for name, routine in system.routines.items():
            cfg = routine.config.as_dict(refresh=True)
            if cfg:
                for key, val in cfg.items():
                    rows.append({'section': name, 'key': key, 'value': val})

        for name, model in system.models.items():
            cfg = model.config.as_dict(refresh=True)
            if cfg:
                for key, val in cfg.items():
                    rows.append({'section': name, 'key': key, 'value': val})

        return rows

    def save_config(self, file_path=None, overwrite=False):
        """
        Save all system, model, and routine configurations to an rc-formatted
        file.

        Parameters
        ----------
        file_path : str, optional
            path to the configuration file default to `~/andes/andes.rc`.
        overwrite : bool, optional
            If file exists, True to overwrite without confirmation. Otherwise
            prompt for confirmation.

        Warnings
        --------
        Saved config is loaded back and populated *at system instance creation
        time*. Configs from the config file takes precedence over default config
        values.
        """
        if file_path is None:
            andes_path = os.path.join(os.path.expanduser('~'), '.andes')
            os.makedirs(andes_path, exist_ok=True)
            file_path = os.path.join(andes_path, 'andes.rc')

        elif os.path.isfile(file_path):
            if not confirm_overwrite(file_path, overwrite=overwrite):
                return

        conf = self.collect_config()
        with open(file_path, 'w') as f:
            conf.write(f)

        logger.info('Config written to "%s"', file_path)
        return file_path

    @staticmethod
    def configure_numpy(seed='None', divide='warn', invalid='warn'):
        """
        Configure NumPy based on Config.
        """

        # set up numpy random seed
        if isinstance(seed, int):
            np.random.seed(seed)
            logger.debug("Random seed set to <%d>.", seed)

        # set levels
        np.seterr(divide=divide,
                  invalid=invalid,
                  )

    @staticmethod
    def load_config_rc(conf_path=None):
        """
        Load config from an rc-formatted file.

        Parameters
        ----------
        conf_path : None or str
            Path to the config file. If is `None`, the function body will not
            run.

        Returns
        -------
        configparse.ConfigParser
        """
        if conf_path is None:
            return

        conf = configparser.ConfigParser()
        conf.read(conf_path)
        logger.info('> Loaded config from file "%s"', conf_path)
        return conf


# ------------------------------------------------------------------
#  Format-specific config extraction (module-level helpers)
# ------------------------------------------------------------------

def _extract_config_xlsx(case):
    """
    Read the ``_config`` sheet from an xlsx file.

    Returns a list of dicts with keys ``section``, ``key``, ``value``.
    """
    try:
        from andes.shared import pd
        df = pd.read_excel(case, sheet_name='_config',
                           index_col=None, engine='openpyxl')
        df.dropna(axis=0, how='all', inplace=True)
        df.dropna(subset=['section', 'key'], inplace=True)
        return df.to_dict(orient='records')
    except (ValueError, KeyError):
        # No _config sheet — expected for older files
        return []
    except Exception as e:
        logger.warning("Could not read _config from xlsx: %s", e)
        return []


def _extract_config_json(case):
    """
    Read the ``_config`` key from a JSON file.

    Returns a list of dicts with keys ``section``, ``key``, ``value``.
    """
    import json as json_mod

    if not isinstance(case, str):
        return []

    try:
        with open(case, 'r') as f:
            data = json_mod.load(f)
        return data.get('_config', [])
    except Exception as e:
        logger.warning("Could not read _config from json: %s", e)
        return []
