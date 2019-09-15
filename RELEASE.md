
# Release (that doesn't quite work yet)

- Update version available via `nupic.torch.__version__`
- `git tag $VERSION`
- `git push --tags upstream`
- CircleCI does the rest, new update should be available via `pip install nupic.torch`
- Update version in `nupic.torch.__version__` to `X.Y.Z.dev0`
