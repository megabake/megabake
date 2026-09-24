import torch

from megabake.v3.frontend import collect_facts, normalize_fx, recognize
from megabake.v3.frontend.inventory import build_inventory, inventory_json


def test_inventory_reports_linear_tiny_shape_and_unknown_physical_bytes():
    program = normalize_fx(torch.export.export(torch.nn.Linear(33, 17), (torch.ones(1, 33),)), input_spec={})
    inventory = build_inventory(recognize(program, facts=collect_facts(program)))
    linear = inventory["operations"][0]
    assert (linear["attributes"]["M"], linear["attributes"]["N"], linear["attributes"]["K"]) == (1, 17, 33)
    assert inventory["bytes"]["physical_dram"] is None
    assert inventory["bytes"]["parameter_storage"] == (17 * 33 + 17) * 4
    assert linear["weights"] and inventory_json(recognize(program, facts=collect_facts(program))) == inventory_json(recognize(program, facts=collect_facts(program)))
