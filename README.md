# GRAVA Common

Shared Python contracts and coordinate utilities for the GRAVA framework.
`grava_common.assets.SceneAsset` defines the versioned GR-NavSim release index.
The coordinate converter preserves the existing right-positive lateral and
forward-positive longitudinal model frame. The legacy `nous` wire label is
retained for data compatibility; it does not refer to an internal service.

Install with `pip install .`. Use this version together with grava-train and
grava-sim-engine 0.1.0. Full setup and data preparation instructions live in
[grava-train](https://github.com/AhernResearch/grava-train).

Original workspace directories are not required at runtime. See PROVENANCE.md
for the source snapshot used to initialize this independent repository.
