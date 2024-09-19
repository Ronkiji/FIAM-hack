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

Mac OS:
4. Activate the virtual environment (from main dir):
   ```
   . .venv/bin/activate
   ```

Windows OS:
4. Open Windows PowerShell (Admin):
   Press
   ```
   Ctr + X
   ```
   Then select Windows PowerShell (Admin)

5.  Check if you have permission to run scripts:
   ```
   Get-ExecutionPolicy
   ```

   Then press enter

If you get the output
   ```
   RemoteSigned
   ```
skip the next step

6. Allow system to run scripts:
   ```
   Set-ExecutionPolicy RemoteSigned
   ```

   Then press enter and type Y to confirm change

7. Activate the virtual environment (from main dir):
   ```
   .\.venv\Scripts\activate
   ```