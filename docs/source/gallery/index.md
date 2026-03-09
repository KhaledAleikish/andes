# Gallery

Application examples showcasing ANDES for power system research.
Each example is a self-contained notebook that you can download and extend for your own studies.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Forced Oscillation Source Localization
:link: forced-oscillation
:link-type: doc
:img-top: images/forced-oscillation.png

Inject a periodic governor disturbance on the WECC 179-bus system with 29 generators.
Locate the source using PMU-observable energy flow and phase analysis, even when
resonance causes non-source generators to oscillate more.

+++
WECC 179-bus | PFlow, TDS, FFT
:::

:::{grid-item-card} Low-Inertia Frequency Stability
:link: low-inertia-freq
:link-type: doc
:img-top: images/low-inertia-freq.png

Scale generator inertia on the IEEE 14-bus system to emulate increasing renewable
penetration. Trip a generator and quantify how RoCoF and frequency nadir degrade as
inertia decreases.

+++
IEEE 14-bus | PFlow, TDS
:::

:::{grid-item-card} Critical Clearing Time Mapping
:link: critical-clearing-time
:link-type: doc
:img-top: images/critical-clearing-time.png

Sweep fault clearing times across all buses in the Kundur two-area system to map the
transient stability boundary. Visualize CCT as a heatmap of rotor angle separation
and identify the most vulnerable buses.

+++
Kundur 2-area | PFlow, TDS, reinit
:::

::::

## Running the Examples

All examples use built-in test cases and can be run directly after installing ANDES:

```bash
pip install andes[dev]    # includes matplotlib, pandas
pip install scikit-learn  # optional, for ML extensions
```

Each notebook is designed as a **template**: after following the walkthrough, look for the
"Extend This Example" section at the end for ideas on adapting it to your own research.

```{toctree}
:maxdepth: 1
:hidden:

forced-oscillation
low-inertia-freq
critical-clearing-time
```
