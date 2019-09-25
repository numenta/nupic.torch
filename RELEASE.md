
# Release Process

1. Create a Release PR that includes:
  - Complete release notes in `CHANGELOG.md`
  - Update version currently in `src/torch/__init__.py` as `nupic.torch.__version__`
1. Create a new Github "Release" at https://github.com/numenta/nupic.torch/releases/new
  - Along with the creation of the release, there is an option to create a git tag with the release. Name it "X.Y.Z" and point it to the commit SHA for the merged PR described above.
  - Release title should be "X.Y.Z".
  - Release description should be the latest changelog.
  - Confirm deployment to [PYPI](https://pypi.org/project/nupic.torch/).
1. Announce at https://discourse.numenta.org/c/engineering/machine-learning.
1. Update master to developer version `X.Y.Z.dev0` in a new PR for continuing development.
