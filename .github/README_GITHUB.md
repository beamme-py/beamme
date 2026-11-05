# How to create a new release of BeamMe

1. **Make sure your local repository is up to date and that all remote tags are available:**
   ```bash
   git pull
   git fetch --tags
   ```

2. **Inspect existing tags to avoid version conflicts:**
   ```bash
   git tag
   ```

3. **Choose a new version tag in the form `vX.Y.Z` (for example, `v1.2.4`).**

4. **Identify the commit you want to release and create a tag for it:**
   ```bash
   git tag vX.Y.Z <commit-hash>
   ```
   Optionally, create an *annotated* tag with a release message:
   ```bash
   git tag -a vX.Y.Z <commit-hash> -m "Release vX.Y.Z with new support for ABC"
   ```

5. **Push the new tag to GitHub to trigger the release workflow**
   (verify whether your remote is `origin` or `upstream`):
   ```bash
   git push upstream vX.Y.Z
   ```

6. **The automated CI workflow will build and publish all wheels to PyPI.**

7. **Optionally, create a GitHub Release from the new tag using the GitHub web interface**
   (for example, to add release notes or highlight changes).


# Protected workflow changes

Contributions are usually made from a fork, and the test workflows run automatically for the pull request.

The only exception is a change to [`workflows/testing_protected.yml`](workflows/testing_protected.yml), because that workflow requires access to repository secrets (to activate Cubit Coreform). It must be made on a branch in the main repository whose name starts with `trusted-ci/`, for example `trusted-ci/update-protected-workflow`. Open the pull request from that branch so GitHub can run the updated workflow with the required Cubit Coreform secrets.
