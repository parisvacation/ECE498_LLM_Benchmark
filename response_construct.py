from pydantic import BaseModel, Field
from src.task_report import EngineeringReport


class ConfigFile(BaseModel):
    theta: float = Field(description="The value of theta")
    tau: float = Field(description="The value of tau")
    num: list[float] = Field(description="The numerator of the transfer function of the controller")
    den: list[float] = Field(description="The denominator of the transfer function of the controller")


# Define your desired output structure
class Response_structure(BaseModel):
    task_report: EngineeringReport
    config: ConfigFile