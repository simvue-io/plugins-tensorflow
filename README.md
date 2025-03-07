# Simvue Connectors - Template

<br/>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/simvue-io/.github/blob/5eb8cfd2edd3269259eccd508029f269d993282f/simvue-white.png" />
    <source media="(prefers-color-scheme: light)" srcset="https://github.com/simvue-io/.github/blob/5eb8cfd2edd3269259eccd508029f269d993282f/simvue-black.png" />
    <img alt="Simvue" src="https://github.com/simvue-io/.github/blob/5eb8cfd2edd3269259eccd508029f269d993282f/simvue-black.png" width="500">
  </picture>
</p>

<p align="center">
This is a template repository which allows you to quickly create new Plugins which provide Simvue tracking and monitoring functionality to Python-based simulations.
</p>

<div align="center">
<a href="https://github.com/simvue-io/client/blob/main/LICENSE" target="_blank"><img src="https://img.shields.io/github/license/simvue-io/client"/></a>
<img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue">
</div>

<h3 align="center">
 <a href="https://simvue.io"><b>Website</b></a>
  •
  <a href="https://docs.simvue.io"><b>Documentation</b></a>
</h3>

## How to use this template

### Naming your plugin
First, make a name for your new plugin. Typically, the module name is of the form `simvue-{software_name}`, and the connector class itself is of the form `{SoftwareName}Run`. Update the `pyproject.toml` file with the name of your module, and also update the directory currently called `simvue_template` with your module name.

### Creating the code
Ideally, the plugin class which you want users to interact with should be made in the `plugin.py` file inside your module, with any extra functionality which it needs to work (but you don't want inside the class itself) put in files inside the `extras` directory. However since there is no set format for plugins (unlike the more rigid structure of connectors), this may change depending on your needs. Make sure you document thoroughly in the README and then examples how you intend for your plugin to be used. Check out any of our premade plugins for ideas:

* [TensorFlow](https://github.com/simvue-io/plugins-tensorflow)

Also look at the `CONTRIBUTING.md` file for expected coding standards.


### Writing examples
In the `examples` directory, please provide at least one example of your plugin being used to track your simulation software. Create this example inside a function so that it can be used in the integration tests.

### Writing tests
You should create two types of tests:

* Unit tests: Check each element of your plugin independently, such as file parsers and callbacks, each method etc. These should use Pytest.
* Integration tests: These check the end-to-end functionality of your plugin when used with the actual simulation software. You should parametrize the test to include offline mode, as well as online. You can use the example(s) which you created earlier as the basis for these tests.

### CI Workflows
Inside the `.github` directory, there are a number of workflows already created. You should edit these to work for your plugin. They include:

* `test_macos`, `test_ubuntu`, `test_windows`: These run the unit and integration tests, should not need to be altered
* `deploy`: Automates deployment to test-PyPI and PyPI for tagged releases (see below). You need to update the module names in this file - see the curly brackets.

### Deployment
When you are happy with your plugin and are ready to deploy it to PyPI for the first time, you need to do the following:

* Install `poetry` and `twine` if you haven't already: `pip install poetry twine`
* Check your `pyproject.toml` file is valid by running `poetry check`
* Install your module: `poetry install`
* Build the distribution: `poetry build`
* Go to `test.pypi.org`, create an account, and get a token
* Upload your package with Twine: `twine upload -r testpypi dist/*`
* Enter the token when prompted
* Go to `https://test.pypi.org/project/{your-package-name}`, check it has been published
* Click 'Manage Project'
* If you wish to enable automatic deployments, click 'Publishing' -> 'Add a new publisher' and fill in the details for your repository, setting Workflow name to `deploy.yaml` and Environment name to `test_pypi`

If this was all successful, repeat with the real PyPI instance at `pypi.org`, using `twine upload dist/*`, and setting the Environment name in the publisher settings to `pypi`.

From now on, you can do deployments automatically. Simply:

* Update the `pyproject.toml` with a new version number, eg `v1.0.1`
* Update the CHANGELOG to reflect your newest changes
* Tag a branch with a semantic version number, eg `git tag v1.0.1`
* Push the tag: `git push origin v1.0.1`

This should automatically start the deployment workflow - check that it completes successfully on the Github UI.

### Updating the README
When finished, delete all of the information above under the 'How to use this template' heading. Then update the information below to be relevant for your plugin:

## Implementation
{List here how your Plugin works, and the things about the simulation it tracks by default.}

## Installation
To install and use this plugin, first create a virtual environment:
```
python -m venv venv
```
Then activate it:
```
source venv/bin/activate
```
And then use pip to install this module:
```
pip install {your_module_name_here}
```

## Configuration
The service URL and token can be defined as environment variables:
```sh
export SIMVUE_URL=...
export SIMVUE_TOKEN=...
```
or a file `simvue.toml` can be created containing:
```toml
[server]
url = "..."
token = "..."
```
The exact contents of both of the above options can be obtained directly by clicking the **Create new run** button on the web UI. Note that the environment variables have preference over the config file.

## Usage example
{Give an example of how to use your plugin, with details such as the actual simulation being run abstracted away to make it as generic as possible.}

## License

Released under the terms of the [Apache 2](https://github.com/simvue-io/client/blob/main/LICENSE) license.
