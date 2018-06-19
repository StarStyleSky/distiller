import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import distiller
from models import ALL_MODEL_NAMES, create_model

def setup_test(arch, dataset):
    model = create_model(False, dataset, arch, parallel=False)
    assert model is not None

    # Create the masks
    zeros_mask_dict = {}
    for name, param in model.named_parameters():
        masker = distiller.ParameterMasker(name)
        zeros_mask_dict[name] = masker
    return model, zeros_mask_dict

def find_module_by_name(model, module_to_find):
    for name, m in model.named_modules():
        if name == module_to_find:
            return m
    return None
