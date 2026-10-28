import importlib
import inspect


def load_model(
    model_path: str,
    device: str = "cuda",
    module_name="",
    package_name="..models.modules",
    **kwargs,
):
    if not model_path.endswith((".pt", ".pth")):
        raise ValueError(f"Expected a PyTorch checkpoint, got: {model_path}")

    module = getattr(importlib.import_module(package_name, __package__), module_name)
    if "device" in inspect.signature(module).parameters:
        kwargs["device"] = device
    model = module(**kwargs)
    model.load_model(model_path).to(device)
    model.device = device
    return model


