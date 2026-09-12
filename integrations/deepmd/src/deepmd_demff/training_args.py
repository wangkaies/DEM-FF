from dargs import Argument
from deepmd.utils.argcheck import model_args_plugin


@model_args_plugin.register("demff")
def demff_model_args():
    return Argument("demff", dict, [
        Argument("foundation", str, doc="Trusted .demff manifest for the original complete MACE-LES architecture and weights."),
        Argument("precision", str, optional=True, default="float64"),
    ])
