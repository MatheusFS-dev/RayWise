![header](https://capsule-render.vercel.app/api?height=190&type=blur&color=4ea7f7&section=header&text=RayWise&fontColor=f8f8f2&fontSize=40)

<p align="center">
<a href="https://github.com/DenverCoder1/readme-typing-svg"><img src="https://readme-typing-svg.herokuapp.com?font=Time+New+Roman&color=%234ea7f7&size=25&center=true&vCenter=true&width=700&height=30&lines=RayWise:+NAS+for+mmWave+Beam+Prediction"></a>
</p>

<p align="center">
  <img src="https://img.shields.io/github/license/MatheusFS-dev/RayWise" alt="GitHub License">
  <img src="https://img.shields.io/github/issues/MatheusFS-dev/RayWise" alt="GitHub Issues">
  <img src="https://img.shields.io/github/forks/MatheusFS-dev/RayWise" alt="GitHub Forks">
  <img src="https://img.shields.io/github/stars/MatheusFS-dev/RayWise" alt="GitHub Stars">
</p>

<p align="center">
  <a href="#">
      <img src="https://api.visitorbadge.io/api/VisitorHit?user=MatheusFS-dev&repo=RayWise&countColor=%23007FFF" />
   </a>
</p>

RayWise is a research repository focused on mmWave beam prediction with CNN1D, NAS, ensembles, and knowledge distillation experiments over Raymobtime-style inputs.

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Project Overview](#project-overview)
  - [What Is In This Repository](#what-is-in-this-repository)
  - [Repository Layout](#repository-layout)
  - [Known Historical Errors](#known-historical-errors)
  - [Araras (Private Utility Toolkit)](#araras-private-utility-toolkit)
- [Installation Instructions](#installation-instructions)
  - [Prerequisites](#prerequisites)
  - [Steps](#steps)
- [Usage](#usage)
  - [Run NAS v7 (Best Architecture Search)](#run-nas-v7-best-architecture-search)
  - [Run KD With The Best Architecture](#run-kd-with-the-best-architecture)
  - [Where Outputs Are Stored](#where-outputs-are-stored)
- [Documentation Map](#documentation-map)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)
- [References](#references)

## Project Overview

### What Is In This Repository

This repository contains many experiment branches and historical versions. Public users should focus on:

- `src/architectures/cnn1d/nas/v7/nas_v7_seeds_val_acc.py`: best NAS setup found in this project.
- `src/architectures/cnn1d/ensemble/nas_mo_kd_cnn1d.py`: knowledge distillation pipeline that uses the best architecture family.
- `weights/`: best saved models found during experiments.
- `results/`: experiment outputs and summaries.

Most remaining files in `src/architectures/`, `legacy/`, and multiple versioned folders are exploratory tests and intermediate research iterations.

### Repository Layout

- `src/`: main codebase (data loaders, architecture scripts, and training logic).
- `src/architectures/cnn1d/nas/`: NAS experiments and versioned architecture searches.
- `src/architectures/cnn1d/ensemble/`: ensemble and knowledge distillation experiments.
- `docs/`: technical notes, setup references, and methodology documentation.
- `results/`: exported experiment metrics, reports, and analysis artifacts.
- `weights/`: best model checkpoints/artifacts found during experiments.
- `legacy/`: historical notebooks and previous experiment versions kept for traceability.
- `tests/`: helper scripts and environment checks used during development.

### Known Historical Errors

Some older scripts and notebooks include historical mistakes that were documented during development. Newer versions are generally the ones to follow.

- See `docs/ErrorsCaller.md` for a public-facing warning and review checklist.
- Recommendation: verify dataset paths, callback wiring, and tensor shape assumptions before reusing old experiment files.

## Installation Instructions

### Prerequisites

- Python 3.10+ recommended
- Git
- (Optional) CUDA-compatible GPU for training

### Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/MatheusFS-dev/RayWise.git
   cd RayWise
   ```

2. Create and activate a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r REQUIREMENTS.txt
   ```

4. Ensure datasets are available under:

   - `src/data/s008`
   - `src/data/s009`

## Usage

### Run NAS v7 (Best Architecture Search)

```bash
python src/architectures/cnn1d/nas/v7/nas_v7_seeds_val_acc.py
```

This is the best architecture search setup reported in this repository.

### Run KD With The Best Architecture

```bash
python src/architectures/cnn1d/ensemble/nas_mo_kd_cnn1d.py
```

This script is the multi-objective NAS + knowledge distillation implementation based on the best architecture family.

Note:

- In `src/architectures/cnn1d/ensemble/nas_mo_kd_cnn1d.py`, update `TEACHER_MODELS_DIR` to your local teacher checkpoints path before running.

### Where Outputs Are Stored

- Intermediate run artifacts: local `runs/` folders created by scripts.
- Curated experiment outputs: `results/`.
- Best model artifacts: `weights/`.

## Documentation Map

Main documentation is under `docs/`.

- `docs/README.md`: map of documentation files and when to use each one.
- `docs/howto.md`: environment setup tips (Git, venv/conda, Docker, Jupyter workflows).
- `docs/ErrorsCaller.md`: warning for historical errors in old files.
- `docs/references/references.md`: bibliography and external references.
- `docs/related_works/related_works.md`: related work notes.

## Contributing

> [!IMPORTANT]
> First read `CONTRIBUTING.md`.

1. Fork the project.
2. Create a feature branch.
3. Commit your changes.
4. Push to your fork.
5. Open a Pull Request.

## License

This project is licensed under the terms described in `LICENSE.txt`.

## Author

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/MatheusFS-dev" title="Matheus Ferreira">
        <img src="https://avatars.githubusercontent.com/u/99222557" width="100px;" alt="Matheus Ferreira on GitHub"/><br>
        <sub>
          <b>Matheus Ferreira</b>
        </sub>
      </a>
    </td>
  </tr>
</table>

## References

- Raymobtime-related publications and dataset resources referenced in `docs/references/references.md`
- Additional literature review in `docs/related_works/related_works.md`
