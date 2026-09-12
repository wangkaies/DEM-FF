"""Register an actual model in DeePMD's native PyTorch trainer."""

def load():
    from . import training_model, training_args  # noqa: F401


# DeePMD loads the entry-point object; registration must occur at import time.
load()
