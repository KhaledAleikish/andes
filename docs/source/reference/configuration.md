# Configuration

ANDES behavior can be customized through configuration options for the system, routines, and models. This reference documents all available configuration options and how to modify them.

## Configuration Resolution

ANDES resolves configuration from multiple sources using a layered priority system. When the same option is specified in more than one source, the higher-priority source takes precedence. The four sources, listed from lowest to highest priority, are:

1. **Compiled defaults**: Hard-coded values defined in each model, routine, and system configuration class. These values are used when no external source provides an override.

2. **User configuration file** (`andes.rc`): A persistent INI-format file located at `~/.andes/andes.rc`. Settings in this file apply globally to all ANDES sessions on the machine and serve as user-level defaults.

3. **Embedded case configuration** (`_config`): Configuration records embedded within xlsx or json case files. These settings travel with the case file and override any `andes.rc` values for the same option, ensuring reproducibility when cases are shared between users or machines. Note that PSS/E and MATPOWER formats do not support embedded configuration.

4. **Command-line overrides** (`config_option`): Options passed via the `-O` flag on the command line or the `config_option` argument in `andes.load()` / `andes.run()`. These have the highest priority and override all other sources.

The resolution is performed during system initialization in three phases. First, `andes.rc` is loaded into an internal configuration object. Second, if the case file contains a `_config` section, its entries are merged, overwriting any overlapping keys. Third, command-line overrides are applied. The merged configuration object is then distributed to the system, all routines, and all models during their construction.

This layered design allows a single case file to carry its intended simulation settings while still permitting per-user customization through the rc file and per-run adjustments through the command line.

```
Priority (lowest → highest):
compiled defaults  <  andes.rc  <  _config in data file  <  CLI config_option
```

## Configuration Levels

ANDES uses a hierarchical configuration system with multiple levels, each controlling different aspects of the software.

| Level | Scope | Example |
|-------|-------|---------|
| System | Case-relevant global settings | `freq`, `mva` |
| Runtime | Machine/environment settings | `numba`, `sparselib` |
| Routine | Analysis settings | `PFlow.tol`, `TDS.tf` |
| Model | Model-specific | `TGOV1.allow_adjust` |

System, routine, and model configurations are written to both `andes.rc` and `_config` in data files. Runtime configuration is written only to `andes.rc` (under the `[Runtime]` section) and is never embedded in case data files, because machine-specific settings such as JIT compilation or sparse solver selection should not travel with the case.

## Viewing Configuration

### All Configuration

After loading a system, you can inspect the configuration at each level by accessing the `config` attribute of the system, routines, or models.

```python
import andes

ss = andes.load('case.xlsx')

# System config (case-relevant)
print(ss.config)

# Runtime config (machine/environment)
print(ss.runtime)

# Power flow config
print(ss.PFlow.config)

# TDS config
print(ss.TDS.config)

# Model config
print(ss.TGOV1.config)
```

### Specific Options

Individual configuration values can be accessed directly using attribute notation, which is useful when you need to check or use a specific setting in your code.

```python
# Get simulation end time
print(f"End time: {ss.TDS.config.tf}")

# Get convergence tolerance
print(f"Tolerance: {ss.PFlow.config.tol}")
```

## Modifying Configuration

### In Python

The most common approach is to modify configuration after loading the system but before running any analysis routines. This allows settings to be adjusted based on the specific requirements of each study.

```python
ss = andes.load('case.xlsx')

# Modify before running
ss.TDS.config.tf = 20        # Simulation end time
ss.TDS.config.tstep = 0.01   # Time step
ss.PFlow.config.max_iter = 50  # Max iterations
```

### At Load Time

Configuration options can also be passed when loading a system using the `config_option` argument. This is convenient when you want to set options in a single function call.

```python
ss = andes.run('case.xlsx',
               config_option=['TDS.tf=20', 'PFlow.tol=1e-8'])
```

### Command Line

When using the ANDES command-line interface, configuration options are specified using the `-O` flag followed by the option in `Section.option=value` format.

```bash
# Single option
andes run case.xlsx -r tds -O TDS.tf=20

# Multiple options
andes run case.xlsx -r tds -O TDS.tf=20 PFlow.max_iter=50
```

## Configuration File

For persistent settings that apply across all ANDES sessions, you can create a configuration file that is automatically loaded when ANDES starts.

```bash
# Generate default config
andes --save-config

# Location: ~/.andes/andes.rc
```

### Config File Format

The configuration file uses INI format with sections corresponding to the system, runtime, routines, and models. Options are specified as key-value pairs within each section.

```ini
[System]
freq = 60
mva = 100

[Runtime]
numba = 1
sparselib = klu
seed = 0

[PFlow]
tol = 1e-6
max_iter = 25

[TDS]
tf = 20
tstep = 0.0333
fixt = 1
max_iter = 15

[TGOV1]
allow_adjust = 1
```

The `[Runtime]` section contains machine-specific settings (JIT compilation, sparse solver selection, NumPy error behavior) that are not written to case data files. All other sections appear in both `andes.rc` and the `_config` embedded in xlsx/json data files.

## System Options

System-level options control case-relevant global settings. These are written to data files and travel with the case.

| Option | Default | Description |
|--------|---------|-------------|
| `freq` | 60 | System frequency [Hz] |
| `mva` | 100 | System base MVA |
| `diag_eps` | 1e-8 | Diagonal perturbation for Jacobian singularity |
| `warn_abnormal` | 1 | Warn on abnormal values during simulation |

## Runtime Options

Runtime options control machine-specific and environment settings. These are stored only in `andes.rc` under the `[Runtime]` section and are not embedded in case data files.

| Option | Default | Description |
|--------|---------|-------------|
| `numba` | 1 | Enable Numba JIT compilation |
| `sparselib` | klu | Sparse solver library (`klu`, `umfpack`, `spsolve`, `cupy`) |
| `seed` | 0 | Random seed (0 = no seeding) |
| `np_divide` | warn | NumPy divide-by-zero behavior |
| `np_invalid` | warn | NumPy invalid floating-point behavior |
| `dime_enabled` | 0 | Enable DiME streaming |
| `save_stats` | 0 | Save call statistics |

Runtime options can be accessed and modified via `ss.runtime`:

```python
ss.runtime.numba = 1
ss.runtime.sparselib = 'umfpack'
```

## Power Flow Options (PFlow)

Power flow configuration controls the Newton-Raphson iteration for steady-state solution.

| Option | Default | Description |
|--------|---------|-------------|
| `tol` | 1e-6 | Convergence tolerance |
| `max_iter` | 25 | Maximum iterations |

## Time Domain Options (TDS)

Time-domain simulation options control the numerical integration and output behavior.

| Option | Default | Description |
|--------|---------|-------------|
| `tf` | 20 | End time [s] |
| `tstep` | 1/30 | Time step [s] |
| `fixt` | 1 | Fixed time step (1) or adaptive (0) |
| `max_iter` | 15 | Max iterations per step |
| `tol` | 1e-6 | Convergence tolerance |

## Model Limit Options

All models with limiters support automatic limit adjustment. These options control whether and how limits can be adjusted when steady-state values exceed them.

| Option | Default | Description |
|--------|---------|-------------|
| `allow_adjust` | 1 | Allow limit auto-adjustment |
| `adjust_lower` | 0 | Allow lowering lower limits |
| `adjust_upper` | 1 | Allow raising upper limits |

## Sparse Solvers

ANDES supports multiple sparse linear solvers for the Jacobian system. The choice of solver can affect both performance and numerical stability.

```python
# Set solver via runtime config (system-wide)
ss.runtime.sparselib = 'klu'      # Default (fastest)
ss.runtime.sparselib = 'umfpack'  # Alternative
ss.runtime.sparselib = 'spsolve'  # Pure Python (slowest)
```

KLU is generally the fastest solver for power system problems and is recommended for most use cases.

## Numba JIT Compilation

Numba provides just-in-time compilation of numerical functions for improved performance. When enabled, the first execution compiles the functions (which takes additional time), but subsequent executions are significantly faster.

```python
# Enable via runtime config
ss = andes.load('case.xlsx')
ss.runtime.numba = True
```

## Limit Adjustment

Models with limiters can automatically adjust their limits if steady-state values exceed the specified bounds. This feature helps simulations proceed even when data may have inconsistencies, but should be used with caution.

```python
# Disable limit adjustment for a model
ss.TGOV1.config.allow_adjust = 0

# Allow only upper limit adjustment
ss.ESST1A.config.adjust_lower = 0
ss.ESST1A.config.adjust_upper = 1
```

**Warning**: Relying on automatic adjustment can mask underlying data issues. When limits are adjusted, ANDES reports the changes, and you should investigate whether the original parameters need correction.

## Logging

Control the verbosity of ANDES output using the logging configuration. Higher levels produce less output, while lower levels provide more detail for debugging.

```python
# Configure logging level
andes.config_logger(stream_level=20)  # INFO
andes.config_logger(stream_level=30)  # WARNING only
andes.config_logger(stream_level=10)  # DEBUG
```

| Level | Name | Description |
|-------|------|-------------|
| 10 | DEBUG | Detailed debugging info |
| 20 | INFO | Normal output (default) |
| 30 | WARNING | Warnings only |
| 40 | ERROR | Errors only |

Command line:

```bash
andes run case.xlsx --verbose 10  # DEBUG
andes run case.xlsx --verbose 30  # WARNING
```

## See Also

- {doc}`cli` - Command line interface reference
- {doc}`config` - Auto-generated configuration reference
