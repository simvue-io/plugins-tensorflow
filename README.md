# Simvue Plugins - TensorFlow

<br/>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/simvue-io/.github/blob/5eb8cfd2edd3269259eccd508029f269d993282f/simvue-white.png" />
    <source media="(prefers-color-scheme: light)" srcset="https://github.com/simvue-io/.github/blob/5eb8cfd2edd3269259eccd508029f269d993282f/simvue-black.png" />
    <img alt="Simvue" src="https://github.com/simvue-io/.github/blob/5eb8cfd2edd3269259eccd508029f269d993282f/simvue-black.png" width="500">
  </picture>
</p>

<p align="center">
This plugin allows you to easily add Simvue tracking and monitoring functionality to the training and testing of ML models built using TensorFlow.
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

## Implementation
This package provides a custom `TensorVue` callback, which inherits from TensorFlow's `Callback` class. This will do the following when training, testing or validating a model:

* Uploads the Python script creating the model as a Code Artifact
* Uploads the model config as an Input Artifact
* Uploads parameters about the model as Metadata
* Uploads the Training Accuracy and Loss after each batch to an Epoch runUploads the Training and Validation Accuracy and Loss after each Epoch to the Simulation run
* Uploads model checkpoints after each Epoch to the corresponding Epoch run as Output Artifacts(if enabled by the user)
* Uploads the final model to the Simulation run as an Output Artifact

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
pip install simvue-tensorflow
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

```python
import tensorflow as tf
from tensorflow import keras
import numpy
import matplotlib.pyplot as plt

# Firstly we import our Tensorflow integration:
import simvue_tensorflow.plugin as sv_tf

# Load the training and test data
(img_train, label_train), (img_test, label_test) = keras.datasets.fashion_mnist.load_data()

# Normalize pixel values between 0 and 1
img_train = img_train.astype('float32') / 255.0
img_test = img_test.astype('float32') / 255.0

# Create a basic model
model = keras.Sequential()

model.add(keras.layers.Flatten(input_shape=(28, 28)))
model.add(keras.layers.Dense(32, activation='relu'))
model.add(keras.layers.Dense(10))

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

# At the most basic level, all we need to do is initialize our callback, providing a run name
tensorvue = sv_tf.TensorVue("recognising_clothes_basic")

# Train the model.
model.fit(
    img_train,
    label_train,
    epochs=5,
    validation_split=0.2,
    # Add the tensorvue class as a callback
    callbacks=[tensorvue,]
)

# That's it! Check your Simvue dashboard and you should see:
#    - A 'simulation' run, which summarises the overall training performance
#    - A number of 'epoch' runs, which show the training performed in each epoch

# You can also use the TensorVue callback to record results from model.evaluate
# Above we do it all in one step during the fitting, but you can also do it afterwards:
results = model.evaluate(
    img_test,
    label_test,
    # Add the tensorvue class as a callback
    callbacks=[tensorvue,]
)
```

## License

Released under the terms of the [Apache 2](https://github.com/simvue-io/client/blob/main/LICENSE) license.
