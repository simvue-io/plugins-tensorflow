"""Tensorflow Integration.

Generic callback class which can be used in any Tensorflow Keras CNN to automatically add Simvue tracking and monitoring.
"""

import inspect
import pathlib
import typing

import simvue
from simvue.api.objects import (
    EventsAlert,
    MetricsRangeAlert,
    MetricsThresholdAlert,
    UserAlert,
)
from tensorflow.keras.callbacks import Callback

import simvue_tensorflow.extras.operators as operators
from simvue_tensorflow.extras.create_alerts import create_alerts


class TensorVue(Callback):
    """Tensorflow Callback class for adding Simvue integration."""

    def __init__(
        self,
        run_name: typing.Optional[str] = None,
        run_folder: typing.Optional[str] = None,
        run_description: typing.Optional[str] = None,
        run_tags: typing.Optional[list[str]] = None,
        run_metadata: typing.Optional[dict] = None,
        run_mode: typing.Literal["online", "offline"] = "online",
        alert_definitions: typing.Optional[
            dict[str, dict[str, typing.Union[str, int, float]]]
        ] = None,
        manifest_alerts: typing.Optional[list[str]] = None,
        simulation_alerts: typing.Optional[list[str]] = None,
        epoch_alerts: typing.Optional[list[str]] = None,
        evaluation_alerts: typing.Optional[list[str]] = None,
        start_alerts_from_epoch: int = 0,
        script_filepath: str = inspect.stack()[-1].filename,
        model_checkpoint_filepath: typing.Optional[str] = None,
        model_final_filepath: str = "/tmp/simvue/final_model.keras",
        evaluation_parameter: str = None,
        evaluation_target: float = None,
        evaluation_condition: operators.Operator = None,
        create_epoch_runs: typing.Optional[bool] = True,
        optimisation_framework: bool = False,
        simulation_run: typing.Optional[simvue.Run] = None,
        evaluation_run: typing.Optional[simvue.Run] = None,
    ):
        """Tensorflow Callback class for adding Simvue integration.

        Parameters
        ----------
        run_name : typing.Optional[str], optional
            Name of the Simvue run, must be provided when not using the Optimisation framework, by default None
            If using the optimisation framework, any name provided here will be overriden by the name from the Workspace
        run_folder : typing.Optional[str], optional
            Name of the folder to store the run in, by default None
            If not specified and not using the optimisation framework, will create a folder with the same name as the run
            If using the optimisation framework, any folder provided here will be overriden by the folder from the Workspace
        run_description : typing.Optional[str], optional
           Description of the run, by default None
            If using the optimisation framework, any description provided here will be overriden by the description from the Workspace
        run_tags : typing.Optional[list[str]], optional
            Tags associated with the run, by default None
        run_metadata: typing.Optional[dict], optional
            Metadata associated with this run, by default None
        run_mode : typing.Literal["online", "offline"]
            Whether Simvue should run in Online or Offline mode, by default Online
        alert_definitions : typing.Optional[dict[str, dict[str, typing.Union[str, int, float]]]], optional
            Definitions of any alerts to add to the run, by default None
        manifest_alerts : typing.Optional[list[str]], optional
            Which of the alerts defined above to add to the manifest run, by default None
        simulation_alerts : typing.Optional[list[str]], optional
            Which of the alerts defined above to add to the simulation run, by default None
        epoch_alerts : typing.Optional[list[str]], optional
            Which of the alerts defined above to add to the epoch runs, by default None
        evaluation_alerts : typing.Optional[list[str]], optional
            Which of the alerts defined above to add to the evaluation runs, by default None
        start_alerts_from_epoch : int, optional
            The number of the epoch which you would like to begin setting alerts for, by default 0
        script_filepath : str, optional
            Path of the file to upload as Code to the simulation run, by default inspect.stack()[-1].filename
        model_checkpoint_filepath : typing.Optional[str], optional
            If using the ModelCheckpoint callback, the path where the checkpoint files are saved after each epoch, by default None
        model_final_filepath : str, optional
            The location where the final model should be stored after training is complete, by default "/tmp/simvue/final_model.keras"
        evaluation_parameter: str, optional
            The parameter to check the value of after each Epoch, eitheer accuracy, loss, val_accuracy, or val_loss
        evaluation_target: float, optional
            The target value of the parameter, which will cause the training to stop if satisfied
        evaluation_condition: operators.Operator, optional
            How you wish to compare the latest value of the parameter to the target value
        create_epoch_runs: typing.Optional[bool], optional
            Whether to create runs for the training data for each Epoch individually, by default True
        optimisation_framework : bool, optional
            Whether to use the Simvue ML Optimisation framework, by default False
        simulation_run : typing.Optional[simvue.Run], optional
            If using the ML Opt framework and this callback is being called within the simulation function,
            the 'data' run which has been created by the framework for this trial, by default None
        evaluation_run : typing.Optional[simvue.Run], optional
            If using the ML Opt framework and this callback is being called within the evaluation function,
            the 'eval' run which has been created by the framework for this trial, by default None

        Raises
        ------
        ValueError
            Raised if the ML Optimisation framework is not enabled and no run name was provided
        KeyError
            Raised if attempted to add an alert to a run which was not defined

        """
        if not optimisation_framework and not run_name:
            raise ValueError("Must provide a run name!")
        self.run_name = run_name
        self.run_folder = run_folder or f"/{self.run_name}"
        self.run_description = (
            run_description
            or f"Tracking and monitoring of the training and evaluation of {self.run_name} Tensorflow algorithm."
        )
        self.run_tags = run_tags or []
        self.run_metadata = run_metadata or {}
        self.run_mode = run_mode
        self.alert_definitions = alert_definitions
        self.script_filepath = script_filepath
        self.model_checkpoint_filepath = model_checkpoint_filepath
        self.model_final_filepath = model_final_filepath
        self.evaluation_parameter = evaluation_parameter
        self.evaluation_condition = evaluation_condition
        self.evaluation_target = evaluation_target
        self.create_epoch_runs = create_epoch_runs
        self.optimisation_framework = optimisation_framework
        self.simulation_run = simulation_run
        self.eval_run = evaluation_run

        # Create alerts in a disabled run up front, to validate that they have been defined accurately
        if alert_definitions:
            with simvue.Run(mode="disabled") as temp_run:
                temp_run._user_config.run.mode == self.run_mode
                [
                    create_alerts(alert_name, alert_definition, temp_run)
                    for alert_name, alert_definition in alert_definitions.items()
                ]

        self.manifest_alerts = manifest_alerts or []
        self.simulation_alerts = simulation_alerts or []
        self.epoch_alerts = epoch_alerts or []
        self.evaluation_alerts = evaluation_alerts or []
        self.start_alerts_from_epoch = start_alerts_from_epoch

        for alert_name in (
            self.simulation_alerts
            + self.epoch_alerts
            + self.evaluation_alerts
            + self.manifest_alerts
        ):
            if alert_name not in self.alert_definitions.keys():
                raise KeyError(
                    f"Alert name {alert_name} not present in alert definitions."
                )

        super().__init__()

    def create_manifest_run(self) -> simvue.Run:
        """Create a Manifest run with user defined inputs.

        Returns
        -------
        simvue.Run
            The manifest run

        """
        manifest_run = simvue.Run(mode=self.run_mode)
        manifest_run.init(
            name=f"{self.run_name}_manifest",
            tags=self.run_tags
            + [
                "manifest",
            ],
            folder=self.run_folder,
            description=self.run_description,
            metadata=self.run_metadata,
        )
        [
            create_alerts(alert_name, self.alert_definitions[alert_name], manifest_run)
            for alert_name in self.manifest_alerts
        ]

        if self.script_filepath:
            manifest_run.save_file(
                file_path=self.script_filepath,
                category="code",
            )
        return manifest_run

    def on_train_begin(self, logs: dict):
        """Upload relevant information to Simvue at the start of the training session.

        Parameters
        ----------
        logs : dict
            Currently no data is passed into this argument by Tensorflow.

        Raises
        ------
        RuntimeError
            Raised if the optimisation framework is enabled, but no simulation run has been initialised.

        """
        if not self.optimisation_framework:
            self.simulation_run = simvue.Run(mode=self.run_mode)
            self.simulation_run.init(
                name=self.run_name + "_simulation",
                description=self.run_description,
                folder=self.run_folder,
                tags=self.run_tags
                + [
                    "simulation",
                    "training",
                ],
                metadata=self.run_metadata,
            )
        elif not self.simulation_run:
            raise RuntimeError(
                "Simulation run must be provided when using the Optimisation framework."
            )

        else:
            self.run_name = self.simulation_run._name
            self.run_folder = (
                self.simulation_run._data["folder"]
                + f"/trial_{self.run_name.split('_')[-1]}"
            )
            self.run_tags = self.simulation_run._data["tags"]
            self.simulation_run.update_tags(
                self.run_tags
                + [
                    "training",
                ],
            )

        self.simulation_run.update_metadata(self.params)

        [
            create_alerts(
                alert_name, self.alert_definitions[alert_name], self.simulation_run
            )
            for alert_name in self.simulation_alerts
        ]

        if self.script_filepath:
            self.simulation_run.save_file(
                file_path=self.script_filepath,
                category="code",
            )
        model_config = self.model.get_config()
        self.simulation_run.save_object(
            obj=model_config,
            category="input",
            name="model_config",
        )

    def on_train_end(self, logs: dict):
        """Upload relevant information to Simvue at the end of the training session.

        Parameters
        ----------
        logs : dict
            The output from the final call of on_epoch_end

        """
        if self.model_final_filepath:
            if not pathlib.Path(self.model_final_filepath).exists():
                print(
                    "Directory to store final model in was not found - creating directory..."
                )
                pathlib.Path(self.model_final_filepath).parent.mkdir(exist_ok=True)
            self.model.save(self.model_final_filepath)
            self.simulation_run.save_file(
                file_path=self.model_final_filepath,
                category="output",
                name="final_model.keras",
            )
        if not self.optimisation_framework:
            self.simulation_run.close()

        self.simulation_run = None

    def on_epoch_begin(self, epoch: int, logs: dict) -> None:
        """Upload relevant information to Simvue at the start of a new epoch.

        Parameters
        ----------
        epoch : int
            The epoch currently being trained
        logs : dict
            Currently no data is passed into this argument by Tensorflow.

        Returns
        -------
        None
            If the user does not want Epoch runs, exit this method after logging an Event

        """
        self.simulation_run.log_event(f"Starting Epoch {epoch+1}:")

        if not self.create_epoch_runs:
            return

        self.epoch_run = simvue.Run(mode=self.run_mode)
        self.epoch_run.init(
            name=self.run_name + f"_epoch_{epoch+1}",
            folder=self.run_folder,
            description=f"Tracking the training performed during Epoch {epoch+1}.",
            tags=self.run_tags + ["epoch", "training"],
            metadata=self.run_metadata,
        )

        if self.epoch_alerts and epoch + 1 >= self.start_alerts_from_epoch:
            [
                create_alerts(
                    alert_name, self.alert_definitions[alert_name], self.epoch_run
                )
                for alert_name in self.epoch_alerts
            ]

        if epoch > 0:
            self.epoch_run.log_event("Accuracy and Loss values before epoch training:")
            self.epoch_run.log_event(f"Accuracy: {self.accuracy}, Loss: {self.loss}")
            if self.val_accuracy and self.val_loss:
                self.epoch_run.log_event(
                    f"Validation Accuracy: {self.val_accuracy}, Validation Loss: {self.val_loss}"
                )
        self.epoch_run.log_event("Beginning training...")

    def on_epoch_end(self, epoch: int, logs: dict):
        """Upload relevant information to Simvue at the end of an epoch.

        Parameters
        ----------
        epoch : int
            The epoch currently being trained
        logs : dict
            Metrics for this training (and validation) epoch, such as accuracy and loss

        Raises
        ------
        FileNotFoundError
            Raised if a model checkpoint file location has been specified, but no file can be found
        RuntimeError
            Raised if an evalation parameter has been specified for early stopping, but this cannot be found in the logs

        """
        available_metrics = (
            ["accuracy", "loss", "val_accuracy", "val_loss"]
            if logs.get("val_accuracy") and logs.get("val_loss")
            else ["accuracy", "loss"]
        )
        runs_to_update = (
            (self.epoch_run, self.simulation_run)
            if self.create_epoch_runs
            else (self.simulation_run,)
        )

        for run in runs_to_update:
            run.log_event(f"Epoch {epoch+1} training complete!")
            run.log_event("Accuracy and Loss values after epoch training:")
            run.log_event(f"Accuracy: {logs.get('accuracy')}, Loss: {logs.get('loss')}")
            if logs.get("val_accuracy") and logs.get("val_loss"):
                run.log_event(
                    f"Validation Accuracy: {logs.get('val_accuracy')}, Validation Loss: {logs.get('val_loss')}"
                )

        if epoch > 0 and self.create_epoch_runs:
            self.epoch_run.log_event(
                "Improvements in Accuracy and Loss after epoch training:"
            )

        for metric in available_metrics:
            value = logs.get(metric)
            self.simulation_run.log_metrics({metric: value}, step=epoch + 1)
            if not self.create_epoch_runs:
                continue
            elif epoch > 0:
                change: float = value - getattr(self, metric)
                if (metric in ["accuracy", "val_accuracy"] and change > 0) or (
                    metric in ["loss", "val_loss"] and change < 0
                ):
                    improved: bool = True
                else:
                    improved = False

                self.epoch_run.log_event(
                    f"Improved {metric}: {improved}. Change in {metric}: {change}"
                )
            self.epoch_run.update_metadata({f"final_{metric}": value})

        self.accuracy = logs.get("accuracy")
        self.loss = logs.get("loss")
        self.val_accuracy = logs.get("val_accuracy")
        self.val_loss = logs.get("val_loss")

        if self.create_epoch_runs:
            if self.model_checkpoint_filepath:
                if not pathlib.Path(self.model_checkpoint_filepath).exists():
                    raise FileNotFoundError(
                        f"Model checkpoint has not been created at {self.model_checkpoint_filepath}. Have you enabled the ModelCheckpoint callback? "
                    )
                self.epoch_run.save_file(
                    self.model_checkpoint_filepath, category="output"
                )

            self.epoch_run.close()
        if all(
            (
                self.evaluation_condition,
                self.evaluation_parameter,
                self.evaluation_target,
            )
        ):
            if not logs.get(self.evaluation_parameter):
                raise RuntimeError("Evaluation parameter not found in log file!")
            terminate: bool = operators.OPERATORS[self.evaluation_condition](
                logs.get(self.evaluation_parameter), self.evaluation_target
            )
            if terminate:
                self.model.stop_training = True
                termination_message = f"Training terminating early on epoch {epoch+1} - {self.evaluation_parameter} = {logs.get(self.evaluation_parameter)} which is {self.evaluation_condition} the target of {self.evaluation_target}."
                self.simulation_run.log_event(termination_message)
                print(termination_message)

    def on_train_batch_begin(self, batch: int, logs: dict) -> None:
        """Upload relevant information to Simvue at the start of a new training batch.

        Parameters
        ----------
        batch : int
            The batch being trained
        logs : dict
            Currently no data is passed into this argument by Tensorflow.

        Returns
        -------
        None
            If the user does not want Epoch runs, exit the method as there is nothing to log

        """
        # Print progress in 10% increments, to prevent message spam
        if not self.create_epoch_runs:
            return
        if int((batch) / (self.params.get("steps") / 10)) != int(
            (batch + 1) / (self.params.get("steps") / 10)
        ):
            self.epoch_run.log_event(
                f"Training is {10* int((batch) / (self.params.get('steps') / 10))}% complete."
            )

    def on_train_batch_end(self, batch: int, logs: dict) -> None:
        """Upload relevant information to Simvue at the end of a training batch.

        Parameters
        ----------
        batch : int
            The batch being trained
        logs : dict
            Aggregated metrics for this training up to this batch, such as accuracy and loss

        Returns
        -------
        None
            If the user does not want Epoch runs, exit the method as there is nothing to log

        """
        if not self.create_epoch_runs:
            return
        self.epoch_run.log_metrics(
            {
                "accuracy": logs.get("accuracy"),
                "loss": logs.get("loss"),
            }
        )

    def on_test_begin(self, logs: dict):
        """Upload relevant information to Simvue at the start of validation or evaluation.

        Parameters
        ----------
        logs : dict
            Currently no data is passed into this argument by Tensorflow.

        Raises
        ------
        RuntimeError
            Raised if the optimisation framework is enabled, but no evaluation run has been passed in.

        """
        if self.simulation_run:  # This is here because these can be called during training if validation set provided
            if self.create_epoch_runs:
                self.epoch_run.log_event("Validating results...")
        else:
            if not self.optimisation_framework:
                self.eval_run = simvue.Run(mode=self.run_mode)
                self.eval_run.init(
                    name=self.run_name + "_evaluation",
                    folder=self.run_folder,
                    description="Tracking the evaluation performed on the final model.",
                    tags=self.run_tags + ["evaluation"],
                    metadata=self.run_metadata,
                )
            elif not self.eval_run:
                raise RuntimeError(
                    "Evaluation run must be provided when using the Optimisation framework."
                )
            else:
                self.eval_run.update_tags(
                    self.eval_run._data["tags"]
                    + [
                        "evaluation",
                    ]
                )
            [
                create_alerts(
                    alert_name, self.alert_definitions[alert_name], self.eval_run
                )
                for alert_name in self.evaluation_alerts
            ]

            if self.script_filepath:
                self.eval_run.save_file(
                    file_path=self.script_filepath,
                    category="code",
                )
            model_config = self.model.get_config()
            self.eval_run.save_object(
                obj=model_config,
                category="input",
                name="model_config",
            )

    def on_test_end(self, logs: dict):
        """Upload relevant information to Simvue at the end of validation or evaluation.

        Parameters
        ----------
        logs : dict
            Aggregated accuracy/loss metrics for the test, output from the final call of on_test_batch_end

        """
        if not self.simulation_run:
            self.eval_run.log_event("Accuracy and Loss values after evaluation:")
            self.eval_run.log_event(
                f"Accuracy: {logs.get('accuracy')}, Loss: {logs.get('loss')}"
            )
            self.eval_run.update_metadata(
                {"final_accuracy": logs.get("accuracy"), "final_loss": logs.get("loss")}
            )
            if not self.optimisation_framework:
                self.eval_run.close()

    def on_test_batch_begin(self, batch: int, logs: dict):
        """Upload relevant information to Simvue at the start of a validation or evaluation batch.

        Parameters
        ----------
        batch : int
            The batch being trained
        logs : dict
            Currently no data is passed into this argument by Tensorflow.

        """
        if not self.simulation_run:
            if int((batch) / (self.params.get("steps") / 10)) != int(
                (batch + 1) / (self.params.get("steps") / 10)
            ):
                self.eval_run.log_event(
                    f"Evaluation is {10* int((batch) / (self.params.get('steps') / 10))}% complete."
                )

    def on_test_batch_end(self, batch: int, logs: dict):
        """Upload relevant information to Simvue at the end of a validation or evaluation batch.

        Parameters
        ----------
        batch : int
            The batch being trained
        logs : dict
            Aggregated metrics for this evaluation up to this batch, such as accuracy and loss

        """
        if self.simulation_run:
            if self.create_epoch_runs:
                self.epoch_run.log_metrics(
                    {
                        "val_accuracy": logs.get("accuracy"),
                        "val_loss": logs.get("loss"),
                    },
                    step=batch,
                )
        else:
            self.eval_run.log_metrics(
                {
                    "accuracy": logs.get("accuracy"),
                    "loss": logs.get("loss"),
                }
            )
