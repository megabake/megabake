import torch

from megabake.v3.frontend.capture import FrontendError, UnsupportedGraphError, capture_graph_module


def test_graphmodule_capture_requires_explicit_examples_and_keeps_nested_output():
    graph = torch.fx.symbolic_trace(lambda x, y: {"sum": x + y, "pair": (x, y)})
    with __import__("pytest").raises(FrontendError):
        capture_graph_module(graph, input_spec={})
    program = capture_graph_module(graph, (torch.ones(2), torch.ones(2)), input_spec={"args": 2})
    assert program.run_reference(torch.ones(2), torch.ones(2))["sum"].tolist() == [2.0, 2.0]


def test_graphmodule_capture_rejects_unknown_python_callback_before_export():
    def callback(x):
        return x + 1
    graph = torch.fx.symbolic_trace(callback)
    # symbolic_trace emits a built-in add for this body; explicitly insert a
    # custom callback node to exercise the frontend's pre-export guard.
    fx_graph = torch.fx.Graph()
    x = fx_graph.placeholder("x")
    result = fx_graph.call_function(callback, (x,))
    fx_graph.output(result)
    custom = torch.fx.GraphModule({}, fx_graph)
    with __import__("pytest").raises(UnsupportedGraphError):
        capture_graph_module(custom, (torch.ones(2),), input_spec={})
