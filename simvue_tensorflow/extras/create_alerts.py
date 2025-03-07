"""Create Alerts.

Function for creating alerts based on definitions provided by user

"""

import typing

import simvue


def create_alerts(
    alert_name: str, alert_definition: dict[str, typing.Any], run: simvue.Run
) -> None:
    """Create alerts from their definitions provided in to TensorVue.

    Parameters
    ----------
    alert_name : str
        Name of the alert to create
    alert_definition : dict[str, typing.Any]
        Definition of the alert, passed into the relevant Run method as kwargs
    run : simvue.Run
        The run to add the alerts to

    Raises
    ------
    RuntimeError
        Raised if a valid source could not be deduced from alert definition

    """
    alert_definition = alert_definition.copy()
    _source = alert_definition.pop("source")
    if _source == "events":
        _alert_id = run.create_event_alert(
            name=alert_name,
            **alert_definition,
        )
    elif _source == "metrics" and alert_definition.get("threshold"):
        _alert_id = run.create_metric_threshold_alert(
            name=alert_name,
            **alert_definition,
        )
    elif _source == "metrics":
        _alert_id = run.create_metric_range_alert(
            name=alert_name,
            **alert_definition,
        )
    elif _source == "user":
        _alert_id = run.create_user_alert(
            name=alert_name,
            **alert_definition,
        )
    else:
        raise RuntimeError(f"{alert_name} has unknown source type '{_source}'")
