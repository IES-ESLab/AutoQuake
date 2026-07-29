# AutoQuake

> **AutoQuake: An Automated, All-in-One Solution for Earthquake Catalog Generation.**

AutoQuake connects every step of building an earthquake catalog into a single, configuration-driven pipeline: phase picking (**PhaseNet**) → association (**GaMMA**) → relocation (**H3DD**) → magnitude → first-motion polarity (**DiTingMotion**) → focal mechanism (**GAfocal**). You describe a run in one JSON file and launch it with a single command — no Python required.

---

## Table of Contents

- [1. Install](#1-install)
- [2. Prepare your data](#2-prepare-your-data)
- [3. Configure your run](#3-configure-your-run)
- [4. Run](#4-run)
- [Helper function: prepare the focal input](#helper-function-prepare-the-focal-input)
- [Submodules and External Dependencies](#submodules-and-external-dependencies)
- [License](#license)
- [References](#references)
- [Contact](#contact)

---

## 1. Install

First initialize the bundled submodules, then create the environment. Conda is recommended because the pipeline depends on compiled Fortran binaries and scientific libraries.

```bash
./init_submodules.sh
conda env create -f env.yml
conda activate AutoQuake
```

## 2. Prepare your data

Organize your waveforms by date — each folder name must contain a `YYYYMMDD` token so it can be matched by glob.

For seismometers (SAC):

```text
/data_parent_dir/
 ├── 20240423/             # folder name contains YYYYMMDD
 │   └── waveform.SAC      # waveform data in SAC format
```

## 3. Configure your run

A run is described entirely by a JSON file — point each component at your data paths and tune its parameters. Copy `ParamConfig/params.json` as your starting template and edit the values to match your dataset. Every stage can be switched on or off, so you can run the full chain or only the parts you need.

The file holds a `configs` list, so you can define **multiple runs in a single JSON** — for example one per date, region, or parameter set. They are executed in order, one after another:

```json
{
  "configs": [
    { "name": "run_1", ... },
    { "name": "run_2", ... }
  ]
}
```

## 4. Run

```bash
python predict.py --config params.json
```

Results and logs are written to the result directory set in your config. That's it!

## Helper function: prepare the focal input

The focal stage (**GAfocal**) reads its events from a `dout` file (the `dout_file` field in the `Focal` block of your config). If you are not running the full chain from the start — for example you already have relocation, polarity, and magnitude results and only want to run the focal step — you need to assemble that `dout` yourself.

`autoquake/utils` provides `pol_mag_to_dout` to do exactly this: it merges first-motion polarity (and, optionally, magnitude) into an existing H3DD `dout` and writes a new `dout` in the format the focal stage expects.

```python
from autoquake.utils import pol_mag_to_dout

pol_mag_to_dout(
    ori_dout="path/to/h3dd.dout",       # H3DD relocation output
    df_pol=df_pol,                       # polarity DataFrame (DiTingMotion)
    output_dout="path/to/focal.dout",    # written for the focal stage
    df_mag_event=df_mag_event,           # optional: per-event magnitude
    df_mag_pick=df_mag_pick,             # optional: per-pick magnitude
    df_gamma_event=None                  # optional: If your polarity result carries only the GaMMA `event_index` and no `h3dd_event_index`, pass `df_gamma_event=df_gamma_event` so the events can be mapped — kept for backward compatibility.
)
```

Point the focal stage at the generated file by setting `"dout_file": "path/to/focal.dout"` in your config. If you have no magnitude results, leave `df_mag_event` and `df_mag_pick` as `None` and only the polarity information is written.

---

## Submodules and External Dependencies

This project relies on the following external repositories as submodules:

1. [**EQNet**](https://github.com/IES-ESLab/EQNet): A forked repository contains the PhaseNet, PhaseNet-DAS.
2. [**GaMMA**](https://github.com/IES-ESLab/GaMMA): A forked repository contains the Gaussian Mixture Model for Earthquake Detection and Location.

## License

This project is licensed under the MIT License, which allows for reuse, modification, and distribution with minimal restrictions. See the [LICENSE](./LICENSE) file for the full MIT License text.

### Submodule Licenses

This repository includes third-party submodules and model with their own licensing terms:

1. **GaMMA** - MIT License: Permissive license allowing free use, modification, and distribution.
2. **EQNet** - Academic and Commercial License:
   - This submodule is available for academic and research use only. For commercial use, a separate license is required. Contact the authors for more information.
   - Users must provide proper attribution in any publications resulting from its use.
3. **DiTing-FOCALFLOW** - Announcements:
   - Users are free to make modifications to the programs to meet their particular needs, but are discouraged from distributing modified code to others without notification of the authors. If you find any part of the workflow useful, please cite our work or the corresponding publications of the packages.

   - Dr. Xiao zhuowei designed and trained the DiTingMotion model,Dr. Zhang Miao provided the data download and preprocessing scripts,Zhao Yanna helped on obtaining and using the HASHpy2 code. Questions and comments? Email Ming Zhao (<mn244224@dal.ca>)

Please review each submodule’s `LICENSE` file for detailed terms.

## References

- Yang, H.-Y., Huang, H.-H., Wu, E.-S., Chen, H.-A., Liu, C.-N., Hsu, Y.-F., Liang, W.-T., & Ku, C.-S (2025). [**An ML-Enhanced Earthquake Catalog for the 2024 MW 7.4 Hualien Earthquake Sequence: Insights into Structural Transition from Collision to Subduction in Eastern Taiwan.**]( https://doi.org/10.1029/2025JB032792) Journal of Geophysical Research: Solid Earth, 130, e2025JB032792. https://doi.org/10.1029/2025JB032792.
- Huang, H. H., Wu, Y. M., Song, X., Chang, C. H., Lee, S. J., Chang, T. M., & Hsieh, H. H. (2014). [**Joint Vp and Vs tomography of Taiwan: Implications for subduction-collision orogeny.**](https://www.sciencedirect.com/science/article/pii/S0012821X14000995?via%3Dihub) Earth Planet Science Letters, 392, 177–191. https://doi.org/10.1016/j.epsl.2014.02.026
- Hsu, Y. F., Huang, H. H., Huang, M. H., Tsai, V. C., Chuang, R. Y., Feng, K. F., & Lin, S. H. (2020). [**Evidence for Fluid Migration During the 2016 Meinong, Taiwan, Aftershock Sequence.**](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2020JB019994) Journal of Geophysical Research: Solid Earth. 125. 10.1029/2020JB019994.
- Wu, Y. M., Zhao, L., Chang, C. H., & Hsu, Y. J. (2008). [**Focal-Mechanism Determination in Taiwan by Genetic Algorithm.**](https://pubs.geoscienceworld.org/ssa/bssa/article/98/2/651/350113/Focal-Mechanism-Determination-in-Taiwan-by-Genetic). Bulletin of the Seismological Society of America 2008;; 98 (2): 651–661. doi: https://doi.org/10.1785/0120070115
- Zhao, M., Xiao, Z., Zhang, M., Yang, Y., Tang, L., & Chen, S. (2023). [**DiTingMotion: A deep-learning first-motion-polarity classifier and its application to focal mechanism inversion.**](https://www.frontiersin.org/journals/earth-science/articles/10.3389/feart.2023.1103914/full) Frontiers in Earth Science, 11, 335.https://doi.org/10.3389/feart.2023.1103914
- Zhu, W., & Beroza, G. C. (2019). [**PhaseNet: A Deep-Neural-Network-Based Seismic Arrival Time Picking Method.**](https://academic.oup.com/gji/article/216/1/261/5129142) Geophysical Journal International, Volume 216, Issue 1, January 2019, Pages 261–273, https://doi.org/10.1093/gji/ggy423
- Zhu, W., McBrearty, I. W., Mousavi, S. M., Ellsworth, W. L., & Beroza, G. C. (2022). [**Earthquake Phase Association using a Bayesian Gaussian Mixture Model.**](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021JB023249) Journal of Geophysical Research: Solid Earth, 127, e2021JB023249. https://doi.org/10.1029/2021JB023249
- Zhu, W., Biondi, E., Li, J., Yin, J., Ross, Z. E., & Zhan, Z. (2023). [**Seismic arrival-time picking on distributed acoustic sensing data using semi-supervised learning.**](https://www.nature.com/articles/s41467-023-43355-3) Nat Commun 14, 8192. https://doi.org/10.1038/s41467-023-43355-3

## Contact

If you have any questions or suggestions, feel free to reach out:

- **Email**: [patrick.yang880612@gmail.com](mailto:patrick.yang880612@gmail.com)
- **GitHub**: [@Pamicoding](https://github.com/Pamicoding)
