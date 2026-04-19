# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Phase 0.5: Repository bootstrap. Initial commit, gitignore, gitattributes,
  editorconfig, CHANGELOG skeleton, directory scaffold (`docs/`, `project-a/`,
  `project-b/`, `bench/`, `scripts/`).
- Phase 1: Hardware + driver preflight. `scripts/preflight.fish` verifies
  kernel ≥ 6.14, NPU PCI ID, `/dev/accel/accel0`, amdxdna dmesg, RADV
  Vulkan, group membership, memlock, IOMMU, XRT enumeration, `/dev/video0`.
  `docs/hardware.md` records the tested L16 Gen 2 stack (kernel 6.19.10,
  amdxdna 0.6.0, NPU firmware 1.5.2.380, XRT 2.21.75, Mesa 26.0.4).

### Changed

### Fixed

[Unreleased]: https://example.invalid/compare/HEAD
