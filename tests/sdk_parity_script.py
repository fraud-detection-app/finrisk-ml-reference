"""Dev-only. Run by tests/test_definition.py in a subprocess when the SageMaker Python SDK v3 (`sagemaker` + `sagemaker-mlops`) is installed.

Reads our finrisk-train definition (JSON, path in argv[1]), rebuilds its SHAPE with the SDK -- the same parameters, the same step names,
types, dependencies and property files, the SDK's own ConditionStep / FailStep / JsonGet / Join -- asks the SDK for ITS definition, and
prints both so the test can compare. Exit 77 = the SDK is not importable here (the test then skips).
"""

import json
import sys

try:
    from sagemaker.core.workflow.conditions import ConditionGreaterThanOrEqualTo
    from sagemaker.core.workflow.functions import Join, JsonGet
    from sagemaker.core.workflow.parameters import ParameterFloat, ParameterInteger, ParameterString
    from sagemaker.core.workflow.properties import PropertyFile
    from sagemaker.mlops.workflow.condition_step import ConditionStep
    from sagemaker.mlops.workflow.fail_step import FailStep
    from sagemaker.mlops.workflow.pipeline import Pipeline
    from sagemaker.mlops.workflow.steps import Step, StepTypeEnum
except Exception as exc:  # noqa: BLE001
    print(f"sdk not importable: {type(exc).__name__}: {exc}", file=sys.stderr)
    sys.exit(77)

ours = json.load(open(sys.argv[1]))
PARAMS = {"String": ParameterString, "Integer": ParameterInteger, "Float": ParameterFloat}
parameters = {p["Name"]: PARAMS[p["Type"]](p["Name"], default_value=p["DefaultValue"]) for p in ours["Parameters"]}
TYPES = {"Processing": StepTypeEnum.PROCESSING, "Tuning": StepTypeEnum.TUNING, "RegisterModel": StepTypeEnum.REGISTER_MODEL}


class Raw(Step):
    """A step whose Arguments are given as they are (what D3 says a step is)."""

    def __init__(self, name, step_type, arguments, depends_on=None):
        super().__init__(name=name, step_type=step_type, depends_on=depends_on)
        self._arguments = arguments

    @property
    def arguments(self):
        return self._arguments

    @property
    def properties(self):
        return None


def property_file(step_name, files, name):
    f = next(f for f in files[step_name] if f["PropertyFileName"] == name)
    return PropertyFile(name=f["PropertyFileName"], output_name=f["OutputName"], path=f["FilePath"])


files = {s["Name"]: s.get("PropertyFiles", []) for s in ours["Steps"]}


def sdk_value(node):
    """Our {"Std:JsonGet"}/{"Get": Parameters.X} nodes -> SDK objects."""
    if "Std:JsonGet" in node:
        pf_path = node["Std:JsonGet"]["PropertyFile"]["Get"].split(".")
        return JsonGet(step_name=pf_path[1], property_file=property_file(pf_path[1], files, pf_path[3]), json_path=node["Std:JsonGet"]["Path"])
    return parameters[node["Get"].split(".")[1]]


def build(step):
    if step["Type"] == "Condition":
        a = step["Arguments"]
        conditions = [ConditionGreaterThanOrEqualTo(left=sdk_value(c["LeftValue"]), right=sdk_value(c["RightValue"])) for c in a["Conditions"]]
        return ConditionStep(step["Name"], depends_on=step.get("DependsOn"), conditions=conditions,
                             if_steps=[build(s) for s in a["IfSteps"]], else_steps=[build(s) for s in a["ElseSteps"]])
    if step["Type"] == "Fail":
        message = step["Arguments"]["ErrorMessage"]
        pf = message["Std:JsonGet"]["PropertyFile"]["Get"].split(".")
        return FailStep(step["Name"], error_message=sdk_value(message))
    raw = Raw(step["Name"], TYPES[step["Type"]], step["Arguments"], step.get("DependsOn"))
    raw.property_files = [PropertyFile(name=f["PropertyFileName"], output_name=f["OutputName"], path=f["FilePath"]) for f in step.get("PropertyFiles", [])]
    return raw


pipeline = Pipeline(name="finrisk-train", parameters=list(parameters.values()), steps=[build(s) for s in ours["Steps"]])
print(pipeline.definition())
