"""
Plugin Example
========================
This is an example of the Plugin.

Explain what your example simulation does...

Provide instructions on how to run your plugin, eg:
    - Clone this repository: git clone {link}
    - Create a virtual environment: python -m venv venv
    - Activate the environment: source venv/bin/activate
    - Install the module: pip install .
    - Etc...

"""

import pathlib
import shutil
import uuid
from simvue_template.plugin import YourPlugin

# Define a function for running your example
def example(offline=False) -> None:
    
    # Delete old copies of results, if they exist:
    if pathlib.Path(__file__).parent.joinpath("results").exists():
        shutil.rmtree(pathlib.Path(__file__).parent.joinpath("results"))

    # Initialise the FDSRun class as a context manager
    with YourPlugin(mode="offline" if offline else "online") as run:
        # Initialise the run
        run.init(name="simulation_example")
        
        # Do some simulations...

        # Make sure you return the run ID, for use in the integration tests
        return run.id

if __name__ == "__main__":
    example()

