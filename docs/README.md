# API Documentation

This directory contains detailed API documentation for the `qiskit-qm-provider`. It is set up to be published as a **GitHub Pages** site (Jekyll with the [Just the Docs](https://just-the-docs.github.io/just-the-docs/) theme).

## Contents

- [Providers](providers.md): Documentation for `QMProvider`, `QmSaasProvider`, and `IQCCProvider`.
- [Backend](backend.md): Documentation for `QMBackend` and utility functions like `add_basic_macros` and `get_measurement_outcomes`.
- [Primitives](primitives.md): Documentation for `QMEstimatorV2` and `QMSamplerV2`.
- [Parameter Table](parameter_table.md): Documentation for `ParameterTable` and `Parameter` classes.

## Publishing the documentation site (GitHub Pages)

1. In your GitHub repo, go to **Settings → Pages**.
2. Under **Build and deployment**, set **Source** to **Deploy from a branch**.
3. Choose the branch (e.g. `main`) and set the folder to **/ (root)** or, if your docs live only in `docs/`, set it to **/docs**.
   - **Recommended:** set **Branch** to `main` (or your default branch) and **Folder** to **/docs**. GitHub will build the site from the `docs/` directory using Jekyll.
4. Save. After a minute or two, the site will be available at `https://<username>.github.io/qiskit-qm-provider/` (or your repo’s custom domain if configured).

The `docs/` folder contains `_config.yml` (Jekyll + Just the Docs) and `index.md` as the home page; the rest of the `.md` files are rendered as linked pages with a sidebar.
