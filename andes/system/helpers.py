"""
System-level utility functions and internal helpers.

Active public utilities:
  - ``fix_view_arrays``: Restore NumPy view arrays after deserialization.
  - ``example``: Load a pre-configured IEEE 14-bus example system.
  - ``import_pycode``: Import generated numerical code.

Internal DAE naming helpers (consumed by :mod:`andes.system.facade`):
  - ``_append_model_name``, ``_set_xy_name``, ``_set_hi_name``,
    ``_set_z_name``.

Deprecated backward-compat wrappers (will be removed in v3.0):
  - ``_config_numpy``: use :meth:`SystemConfigRuntime.configure_numpy`.
  - ``load_config_rc``: use :meth:`SystemConfigRuntime.load_config_rc`.
"""

import warnings

from andes.system.config_runtime import SystemConfigRuntime


# ---- Active public utilities -------------------------------------------------

def fix_view_arrays(system, models=None):
    """
    Point NumPy arrays without OWNDATA (termed "view arrays" here) to the source
    array.

    This function properly sets ``v`` and ``e`` arrays of internal variables as
    views of the corresponding DAE arrays.

    Inputs will be refreshed for each model.

    Parameters
    ----------
    system : andes.system.System
        System object to be fixed
    models : OrderedDict, optional
        Subset of models to fix.  Defaults to ``system.models`` (all models).
    """

    if models is None:
        models = system.models

    system.set_var_arrays(models)

    for model in models.values():
        if model.n > 0:
            model.get_inputs(refresh=True)

    return True


def import_pycode(user_pycode_path=None):
    """
    Import generated numerical code (pycode).

    Wrapper around :func:`andes.system.codegen.import_pycode`.
    """
    from andes.system.codegen import import_pycode as _import_pycode
    return _import_pycode(user_pycode_path=user_pycode_path)


def example(setup=True, no_output=True, **kwargs):
    """
    Return an :py:class:`andes.system.System` object for the
    ``ieee14_linetrip.xlsx`` as an example.

    This function is useful when a user wants to quickly get a
    System object for testing.

    Returns
    -------
    System
        An example :py:class:`andes.system.System` object.
    """
    import andes

    return andes.load(andes.get_case("ieee14/ieee14_linetrip.xlsx"),
                      setup=setup, no_output=no_output, **kwargs)


# ---- Internal DAE naming helpers ---------------------------------------------

def _append_model_name(model_name, idx):
    """
    Helper function for appending ``idx`` to model names.
    Removes duplicate model name strings.
    """

    out = ''
    if isinstance(idx, str) and (model_name in idx):
        out = idx
    else:
        out = f'{model_name} {idx}'

    # replaces `_` with space for LaTeX to continue
    out = out.replace('_', ' ')
    return out


def _set_xy_name(mdl, vars_dict, dests):
    """
    Helper function for setting algebraic and state variable names.
    """

    mdl_name = mdl.class_name
    idx = mdl.idx
    for name, item in vars_dict.items():
        for idx_item, addr in zip(idx.v, item.a):
            dests[0][addr] = f'{name} {_append_model_name(mdl_name, idx_item)}'
            dests[1][addr] = rf'${item.tex_name}$ {_append_model_name(mdl_name, idx_item)}'


def _set_hi_name(mdl, vars_dict, dests):
    """
    Helper function for setting names of external equations.
    """

    mdl_name = mdl.class_name
    idx = mdl.idx
    for item in vars_dict.values():
        if len(item.r) != len(idx.v):
            idxall = item.indexer.v
        else:
            idxall = idx.v

        for idx_item, addr in zip(idxall, item.r):
            dests[0][addr] = f'{item.ename} {_append_model_name(mdl_name, idx_item)}'
            dests[1][addr] = rf'${item.tex_ename}$ {_append_model_name(mdl_name, idx_item)}'


def _set_z_name(mdl, dae, dests):
    """
    Helper function for addng and setting discrete flag names.
    """

    for item in mdl.discrete.values():
        if mdl.flags.initialized:
            continue
        mdl_name = mdl.class_name

        for name, tex_name in zip(item.get_names(), item.get_tex_names()):
            for idx_item in mdl.idx.v:
                dests[0].append(f'{name} {_append_model_name(mdl_name, idx_item)}')
                dests[1].append(rf'${item.tex_name}$ {_append_model_name(mdl_name, idx_item)}')
                dae.o += 1


# ---- Deprecated wrappers (to be removed in v3.0) ----------------------------

def _config_numpy(seed='None', divide='warn', invalid='warn'):
    """
    Backward-compatible wrapper to
    :meth:`andes.system.config_runtime.SystemConfigRuntime.configure_numpy`.

    .. deprecated:: 2.0
        Will be removed in v3.0.
    """
    warnings.warn(
        "_config_numpy() is deprecated and will be removed in v3.0. "
        "Use SystemConfigRuntime.configure_numpy() instead.",
        FutureWarning,
        stacklevel=2,
    )
    return SystemConfigRuntime.configure_numpy(seed=seed,
                                               divide=divide,
                                               invalid=invalid)


def load_config_rc(conf_path=None):
    """
    Backward-compatible wrapper to
    :meth:`andes.system.config_runtime.SystemConfigRuntime.load_config_rc`.

    .. deprecated:: 2.0
        Will be removed in v3.0.
    """
    warnings.warn(
        "load_config_rc() is deprecated and will be removed in v3.0. "
        "Use SystemConfigRuntime.load_config_rc() instead.",
        FutureWarning,
        stacklevel=2,
    )
    return SystemConfigRuntime.load_config_rc(conf_path=conf_path)
