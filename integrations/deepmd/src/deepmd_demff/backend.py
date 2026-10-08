"""Lightweight registration; avoid importing torch during CLI discovery."""

from importlib.util import find_spec

from deepmd.backend.backend import Backend


@Backend.register("demff")
class DEMFFBackend(Backend):
    name = "demff"
    suffixes = [".demff"]
    features = Backend.Feature.DEEP_EVAL

    def is_available(self):
        return all(find_spec(x) is not None for x in ("torch", "mace", "les", "ase"))

    @property
    def deep_eval(self):
        from .infer import DEMFFEval
        return DEMFFEval

    @property
    def entry_point_hook(self):
        raise NotImplementedError("The .demff backend handles inference. Train with dp --pt train and model.type=demff; dp freeze is not supported.")

    @property
    def neighbor_stat(self):
        raise NotImplementedError("DEM-FF builds its complete periodic graph with MACE")

    @property
    def serialize_hook(self):
        raise NotImplementedError("Original MACE-LES checkpoint is not a native DP model")

    @property
    def deserialize_hook(self):
        raise NotImplementedError("Use demff-deepmd register to create a .demff manifest")
