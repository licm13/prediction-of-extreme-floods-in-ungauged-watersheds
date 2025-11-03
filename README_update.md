# AI Increases Global Access to Reliable Flood Forecasts
# AI 提升全球可靠洪水预报的可及性

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

This repository allows you to recreate the figures and statistics from the following paper:
本代码库允许您复现以下论文中的图表和统计数据：

[Nearing, Grey, et al. "Global prediction of extreme floods in ungauged watersheds." Nature (2024).](<https://www.nature.com/articles/s41586-024-07145-1>)

-----

## Table of Contents / 目录
- [Overview / 概述](#overview--概述)
- [Extending This Repository with New Models / 使用新模型扩展此代码库](#extending-this-repository-with-new-models--使用新模型扩展此代码库)
- [License / 许可证](#license--许可证)
- [System Requirements / 系统要求](#system-requirements--系统要求)
- [Installation / 安装](#installation--安装)
- [Documentation / 文档：复现论文结果的详细步骤](#documentation--文档复现论文结果的详细步骤)
- [Issues / 问题](https://github.com/googlestaging/global_streamflow_model_paper/issues)

-----

## Overview / 概述
The code in this repository is structured so that all analysis can be done with python notebooks in the `~/notebooks` directory. The expected runtime is approxiamtely one day for the full analysis. The steps are as follows:
本代码库的结构使得所有分析都可以通过 `~/notebooks` 目录中的 Python Notebooks 完成。完整分析的预期运行时间约为一天。步骤如下：

1) Download model data, metadata, and pre-calculated metrics from the associated Zenodo repository [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10397664.svg)](https://doi.org/10.5281/zenodo.10397664).
   从关联的 Zenodo 存储库下载模型数据、元数据和预先计算的指标。

2) Download and prepare Global Runoff Data Center (GRDC) streamflow observation data and model simulation data. This step is not necessary if you want to use the pre-calculated statistics included in the Zenodo repository.
   下载并准备全球径流数据中心 (GRDC) 的径流观测数据和模型模拟数据。如果您想使用 Zenodo 存储库中包含的预计算统计数据，则此步骤不是必需的。

3) Run notebooks to calclate metrics. This step is not necessary if you want to use the pre-calculated statistics included in the Zenodo repository.
   运行 Notebooks
   计算指标。如果您想使用 Zenodo 存储库中包含的预计算统计数据，则此步骤不是必需的。

4) Run notebooks to produce figures and analyses.
   运行 Notebooks 生成图表和分析。

Detailed instructions for these three steps are below.
这三个步骤的详细说明如下。

Also included in the `~/notebooks` directory is a subdirectory called 'backend', which contains much of the active (mostly functional) code used by the analysis notebooks. The user should only touch the source code in this directory to change their local working directory, as described in the instructions below.
`~/notebooks` 目录中还包含一个名为 'backend' 的子目录，其中包含分析 Notebooks 使用的大部分活动代码（主要是功能性的）。用户只应触摸此目录中的源代码来更改其本地工作目录，如下文说明所述。

Within the `~/notebooks/backend` source directory is another source directory called `return_period_calculator`. This subdirectory contains python code for fitting return period distributions and estimating return periods. These calculations are based loosely on guidelines outlined in the USGS Bulletin 17c, with some differences related to the statistical tests used for identifying outliers.
`~/notebooks/backend` 源目录中是另一个名为 `return_period_calculator` 的源目录。该子目录包含用于拟合重现期分布和估计重现期的 Python 代码。这些计算大致基于 USGS Bulletin 17c 中概述的指南，但在用于识别异常值的统计检验方面存在一些差异。

-----

## Extending This Repository with New Models / 使用新模型扩展此代码库

This repository is designed for evaluating hydrological models, particularly their ability to predict extreme events (return periods). You can integrate and test your own advanced machine learning models (e.g., Transformers, GNNs, etc.) against the benchmarks in the paper.
该代码库专为评估水文模型而设计，特别是它们预测极端事件（重现期）的能力。您可以集成和测试您自己的先进机器学习模型（例如 Transformer、GNN 等），并与论文中的基准进行对比。

The workflow is as follows:
工作流程如下：

1.  **Develop Your Model (e.g., in `~/models/`)**:
    * Create a new directory `~/models/` to store your model's code (e.g., `my_advanced_model.py`).
    * Your model script should be able to load data using the provided `notebooks.backend.loading_utils`.
    * The model must be trained and, crucially, produce predictions.
    * **[CRITICAL] Output Format**: The prediction function must output an `xarray.Dataset` object (or be saved as NetCDF files) that matches the format of the original Google model data (which can be loaded via `loading_utils.load_google_model_for_one_gauge`). The data must have dimensions (`gauge_id`, `time`, `lead_time`). This ensures compatibility with the existing evaluation notebooks (like `calculate_return_period_metrics.ipynb`).

2.  **Create an Orchestration Notebook (e.g., `~/notebooks/run_my_model.ipynb`)**:
    * We provide a template notebook (see `run_new_model_experiment.ipynb` example below).
    * This notebook handles:
        * Loading gauge lists (e.g., k-fold splits) and data (using `loading_utils`).
        * Importing your model from `~/models/`.
        * Running the training and/or inference loop for all required gauges.
        * Saving the predictions to a new directory (e.g., `~/model_data/my_advanced_model/`).

3.  **Evaluate Your Model (Re-use Existing Notebooks)**:
    * Once your model's predictions are saved in the correct format, you do **not** need to write new evaluation code.
    * Simply run the original evaluation notebooks:
        * `~/notebooks/calculate_standard_hydrograph_metrics.ipynb`
        * `~/notebooks/calculate_return_period_metrics.ipynb`
    * You will need to modify the `base_path` or `EXPERIMENT` variable in these notebooks to point to your new results directory (e.g., `~/model_data/my_advanced_model/`) instead of the original Google model path.
    * The notebooks will automatically calculate all metrics (NSE, KGE, Precision, Recall, F1) for your model, allowing for a direct comparison with the paper's results.

4.  **Update Environment**:
    * Add your new dependencies (e.g., `torch`, `tensorflow`, `pytorch-geometric`) to the `environment.yml` file (or create a new `environment_extended.yml`) and recreate the conda environment.

---

## License / 许可证
This repository is licensed under an Apache 2.0 open source license. Please see the [LICENSE](https://github.com/googlestaging/global_streamflow_model_paper/blob/main/LICENSE) file in the root directory for the full license.

This is not an official Google product.
本代码库根据 Apache 2.0 开源许可证授权。有关完整许可证，请参阅根目录中的 [LICENSE](https://github.com/googlestaging/global_streamflow_model_paper/blob/main/LICENSE) 文件。

这不是 Google 的官方产品。

## System Requirements / 系统要求
This repository should run on any computer and operating system that supports Python version 3. It has been tested on the Debian GNU/Linux 11 (bullseye) operating system. Running the notebooks for calculating metrics requires 128 GB of local memory.
本代码库应可在任何支持 Python 3 版本的计算机和操作系统上运行。它已在 Debian GNU/Linux 11 (bullseye) 操作系统上进行过测试。运行用于计算指标的 Notebooks 需要 128 GB 的本地内存。

## Installation / 安装
No software installation is required beyond Python v3 and Python libraries contained in the environment file. This repository is based on Python notebooks and can be run directly from a local clone:
除了 Python v3 和环境文件中包含的 Python 库外，不需要安装其他软件。本代码库基于 Python Notebooks，可直接从本地克隆运行：

`git clone https://github.com/googlestaging/global_streamflow_model_paper.git`

An [environment file](https://github.com/googlestaging/global_streamflow_model_paper/blob/main/environment.yml) is included for installing the necessary Python dependencies. If you are using Anaconda (miniconda, etc.) you may create this invironment with the following command from inside the `global_streamflow_model_paper` directory that results from cloning this repository:
代码库中包含一个 [environment file](https://github.com/googlestaging/global_streamflow_model_paper/blob/main/environment.yml)，用于安装必需的 Python 依赖项。如果您使用的是 Anaconda（miniconda 等），您可以在克隆本代码库后产生的 `global_streamflow_model_paper` 目录中，使用以下命令创建此环境：

`conda env create -f environment.yml`

(To use new ML models, see the `environment_extended.yml` example and instructions in the "Extending This Repository" section.)
（如需使用新的机器学习模型，请参阅 `environment_extended.yml` 示例和“使用新模型扩展此代码库”部分中的说明。）

## Documentation / 文档：复现论文结果的详细步骤

### Step 0: Define Working Directory in Source Code / 步骤 0：在源代码中定义工作目录
In the file `~/notebooks/backend/data_paths.py` change the local variable `_WORKING_DIR` to your current working directory.
在 `~/notebooks/backend/data_paths.py` 文件中，将局部变量 `_WORKING_DIR` 更改为您当前的工作目录。

### Step 1: Download Model Data / 步骤 1：下载模型数据

You will need to download and unzip/untar the tarballs from the Zenodo repository listed in the Code and Data Availability section of the paper referenced at the top of this README document. The DOI for the zenodo repository is: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10397664.svg)](https://doi.org/10.5281/zenodo.10397664)
您需要从本 README 文档顶部引用的论文的“代码和数据可用性”部分列出的 Zenodo 存储库下载并解压/解包 tarball。Zenodo 存储库的 DOI 是：[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10397664.svg)](https://doi.org/10.5281/zenodo.10397664)

Your working directory should be the directory created by cloning this repository. Unpacking the tarballs from the Zenodo repository will result in the following subdirectories: `~/model_data`, `~/metadata`, and `~/metrics`, and `~/gauge_groups_for_paper`. All of these subdirectories should be placed in the working directory so that the working directory contains `~/notebooks` (and other subdirectories included in this Github repository), as well as `~/model_data` (and all other subdirectories from the Zenodo repository).
您的工作目录应该是克隆本代码库时创建的目录。解包 Zenodo 存储库中的 tarball 将产生以下子目录：`~/model_data`, `~/metadata`, `~/metrics`, 和 `~/gauge_groups_for_paper`。所有这些子目录都应放置在工作目录中，以便工作目录包含 `~/notebooks`（以及本 Github 代码库中包含的其他子目录），以及 `~/model_data`（以及来自 Zenodo 存储库的所有其他子目录）。

The model output data in this repository include reforecasts from the Google model and reanalyses from the GloFAS model. Google model outputs are in units [mm/day] and GloFAS outputs are in units [m3/s]. Modlel outputs are daily and timestamps are right-labeled, meaning that model ouputs labeled, .e.g., 01/01/2020 correspond to streamflow predictions for the day of 12/31/2019.
本代码库中的模型输出数据包括来自 Google 模型的再预报和来自 GloFAS 模型的再分析。Google 模型输出的单位是 [mm/day]，GloFAS 输出的单位是 [m3/s]。模型输出是每日的，时间戳是右标记的，这意味着标记为（例如）01/01/2020 的模型输出对应于 12/31/2019 这一天的径流预测。

### (Not Required) Step 2: Download GRDC Streamflow Observation Data / （非必需）步骤 2：下载 GRDC 径流观测数据
Due to licensing restrictions, we are not allowed to share streamflow observation data from the Global Runoff Data Center (GRDC). Using the [GRDC Data Portal](https://portal.grdc.bafg.de/applications/public.html?publicuser=PublicUser), download GRDC data for all stations that are listed in the `~/gauge_groups/dual_lstm/grdc_filtered.txt` file. Download these as daily NetCDF files. This requires registering with the GRDC. You will likely have to download these data in multiple batches, resulting in multiple NetCDF files. If that is the case, name each of the NetCDF files uniqely and put them into a single directory somewhere on your local machine. Point to that directory using the `GRDC_DATA_DOWNLOAD_DIRECTORY` variable in the `~/notebooks/backend/data_paths.py` file, and then run the `~/notebooks/concatenate_grdc_downloads.ipynb` notebook to concatenate the download files into one netcdf file.
由于许可限制，我们不允许共享来自全球径流数据中心 (GRDC) 的径流观测数据。请使用 [GRDC 数据门户](https://portal.grdc.bafg.de/applications/public.html?publicuser=PublicUser)，下载 `~/gauge_groups/dual_lstm/grdc_filtered.txt` 文件中列出的所有站点的数据。将它们下载为每日 NetCDF 文件。这需要注册 GRDC。您可能需要分批下载这些数据，从而产生多个 NetCDF 文件。如果是这种情况，请唯一命名每个 NetCDF 文件，并将它们放在本地计算机上的某个目录中。使用 `~/notebooks/backend/data_paths.py` 文件中的 `GRDC_DATA_DOWNLOAD_DIRECTORY` 变量指向该目录，然后运行 `~/notebooks/concatenate_grdc_downloads.ipynb` notebook 将下载的文件连接成一个 netcdf 文件。

GRDC Data Portal: https://portal.grdc.bafg.de/applications/public.html?publicuser=PublicUser

### (Not Required) Understand Cross Validation Gauge Splits / （非必需）了解交叉验证站点划分
Groups of streamflow gauges that were used for cross validation studies are contained in the directory `~/gauge_groups_for_paper` (from the Zenodo repository). Code that shows how these cross validation splits were constructed is contained in the `~/notebooks/create_ungauged_experiments_gauge_groups.ipynb` notebook. This notebook will produce two products:
用于交叉验证研究的径流站点组包含在 `~/gauge_groups_for_paper` 目录中（来自 Zenodo 存储库）。`~/notebooks/create_ungauged_experiments_gauge_groups.ipynb` notebook 中包含了展示这些交叉验证划分是如何构建的代码。这个 notebook 将产生两个产品：

1) Gauge groups as text files for various types of cross validation splits, which are stored in `~/gauge_groups` directory.
   用于各种交叉验证划分的站点组（以文本文件形式），存储在 `~/gauge_groups` 目录中。
2) Maps of the locations of gauges in each cross validation split.
   每个交叉验证划分中站点位置的地图。

You have the option to create gauge splits with GRDC gauges and Caravan gauges (either or both combined).
您可以选择使用 GRDC 站点和 Caravan 站点（单独或两者结合）创建站点划分。

Note that if you run this notebook it will overwrite any existing gauge groups with new ones. These new gauge groups will not be the same as the ones used in the paper, since at least some of these gauge groups were created with a random number generator (i.e., the k-fold cross validation splits and the hydrologically-separated gauge splits). Using gauge groups that you create yourself instead of the ones that are in the `~/gauge_groups_for_paper` subdirectory will result in inaccurate statistics. Doing so will cause the AI model to appear better than it really is since results will be pulled from gauges that were not withheld during training. This notebook is included in this repository only so that you can see how the gauge groups were created.
请注意，如果运行此 notebook，它将用新的站点组覆盖任何现有的站点组。这些新的站点组将与论文中使用的站点组不同，因为至少有部分站点组是使用随机数生成器创建的（即 k-fold 交叉验证划分和水文分离的站点划分）。使用您自己创建的站点组，而不是 `~/gauge_groups_for_paper` 子目录中的站点组，将导致统计数据不准确。这样做会使 AI 模型看起来比实际更好，因为结果将来自训练期间未被保留的站点。本代码库中包含此 notebook 仅为了让您了解站点组是如何创建的。

### (Not Required) Step 3: Calculate Metrics / （非必需）步骤 3：计算指标
Once you have the GRDC netCDF file created, run the `/notebooks/calculate_standard_hydrograph_metrics.ipynb` notebook to calculate a set of standard hydrological skill metrics on modeled hydrographs. This notebook produces plots that are in the paper’s Supplementary Material.
创建 GRDC netCDF 文件后，运行 `/notebooks/calculate_standard_hydrograph_metrics.ipynb` notebook 来计算建模水文过程线的一组标准水文技巧指标。这个 notebook 产生的图表位于论文的补充材料中。

Next, run the `~/notebooks/calculate_return_period_metrics.ipynb` notebook to calculate precision and recall metrics on different magnitude extreme events from modeled hydrographs.
接下来，运行 `~/notebooks/calculate_return_period_metrics.ipynb` notebook，计算建模水文过程线中不同量级极端事件的精确率和召回率指标。

### Step 4: Create Figures and Results from Paper / 步骤 4：创建论文中的图表和结果
Run the various `figure_*.ipynb` notebooks to create figures from the paper. These figures are saved in both PNG and vector graphics formats in the directory `~/results_figures`
运行各种 `figure_*.ipynb` notebooks 来创建论文中的图表。这些图表以 PNG 和矢量图形格式保存在 `~/results_figures` 目录中。