#!/usr/bin/env python3

import os
from config import ArgClass
from model_utils import ModelLoader

backbone = "stgcn2" # infogcn2, msg3d, stgcn2
dataset = "ntu" # ntu, ntu120, ucf101
out_dir = f"./org/model_outlines/{backbone}"

os.makedirs(out_dir, exist_ok=True)


def dump_models_to_org(models: dict, out_dir="./org/model_outlines"):
    """
    models: {"some_name": model, ...} — filenames are derived from the keys.
    Produces one .org file per model with a headline per named module,
    nested by depth (e.g. "l4.gcn.conv" -> depth 3), so you can fold/unfold
    in Emacs org-mode to explore the architecture.
    """
    paths = {}
    for name, model in models.items():
        lines = []
        for mod_name, module in model.named_modules():
            label = mod_name or "(root)"
            lines.append(f"{label} /{type(module).__name__}/")

        path = os.path.join(out_dir, f"{name}.org")
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
        paths[name] = path

    return paths

def main():
    # Define the arguments
    arg_base = ArgClass(f"config/{backbone}/{dataset}/base.yaml")
    arg_poseoff = ArgClass(f"config/{backbone}/{dataset}/cnn.yaml")

    # Load the models
    modelLoader_base = ModelLoader(arg_base)
    modelLoader_poseoff = ModelLoader(arg_poseoff)
    model_base = modelLoader_base.model
    model_poseoff = modelLoader_poseoff.model

    # Create model dictionary
    models = {f"{ backbone }_base": model_base, f"{ backbone }_poseoff": model_poseoff}

    # Write the org files
    dump_models_to_org(models, out_dir)


if __name__ == '__main__':
    main()
