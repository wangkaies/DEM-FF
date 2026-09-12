"""Small, portable manifest referring to an unchanged original checkpoint."""

import hashlib
import json
import os
from pathlib import Path


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def register(model, output):
    model, output = Path(model).resolve(strict=True), Path(output).resolve()
    if output.suffix != ".demff":
        raise ValueError("Manifest filename must end in .demff")
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        reference = os.path.relpath(model, output.parent)
    except ValueError:  # different Windows drives
        reference = str(model)
    data = {"format": "deepmd-demff-v1", "model_path": reference,
            "sha256": digest(model), "fparam_units": "eV",
            "fparam_name": "electronic_temperature_kBT",
            "periodicity": "3d", "atomic_outputs": False}
    with output.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2)
        stream.write("\n")
    return output


def load(path):
    path = Path(path).resolve(strict=True)
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("format") != "deepmd-demff-v1" or data.get("fparam_units") != "eV":
        raise ValueError("Unsupported DEM-FF manifest format or electronic-temperature units")
    if data.get("periodicity") != "3d" or data.get("atomic_outputs") is not False:
        raise ValueError("Only 3D-periodic total-system inference is supported")
    model = (path.parent / data["model_path"]).resolve(strict=True)
    if digest(model) != data.get("sha256"):
        raise ValueError("DEM-FF checkpoint SHA256 mismatch; original weights changed")
    return data, model
