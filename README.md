# FIAM-hack
## Installation

> Rye is not a necesary dependency but does make it **much** simpler. You can still create a virtual environment and install using the `requirements.lock` file

This project uses [Rye](https://rye-up.com/) for dependency management. Follow these steps to set up the project:

Instructions for installation on windows are on the Rye site

1. Install Rye if you haven't already (this is for MacOS):
   ```
   curl -sSf https://rye.astral.sh/get | bash

   ```

2. Clone the repository (or git pull if you've already cloned):

3. Install dependencies using Rye:
   ```
   rye sync
   ```

This will create a virtual environment and install all the required dependencies specified in the `pyproject.toml` file.

4. Activate the virtual environment (from main dir):
   ```
   . .venv/bin/activate
   ```


