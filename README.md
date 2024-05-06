# RAG Search Comparison Tool

This workbook allows for quick analysis of and RAG model system against a provided corpus of documents and base models

## How to use

- Set up python venv and install requirements
- Make a copy of `example.ipynb`. Rename to something relevant for the use case being tested for
- Take document corpus and place in `./data` directory. Currently only Markdown is support so convert all documents to markdown
- Updating settings in the notebook for with the dataset and models to be used along with any other settings to update

** Note: notebook can support either downloading datasets from [HuggingFace](huggingface.io) or using local models. If placing models in repo, please use the `./models` directory **

- Run all cells in the notebook. This process will typically take more the 10 minutes on commodity hardware
