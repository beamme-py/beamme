# The BeamMe website

The BeamMe website is based on the Sphinx PyData Theme (https://pydata-sphinx-theme.readthedocs.io)

## Setup

Ensure that the necessary packages for building the website are present. You can simply install them with

```
pip install -r website/requirements.txt
```

## Building the website locally

In the source directory of BeamMe simply execute

```bash
python website/docs/prepare_docs.py
```

to prepare the documents for the website build.

Then build the website with

```bash
sphinx-build -b html website/docs/source website/docs/build
```

Afterwards you can open `website/docs/build/index.html` to view the local build of the website.

## Important information

The contents of the main landing page reflect the contents of the `README.md` at the top level.

Therefore, to change any contents on the main landing page **do not** touch anything within the website directory!
