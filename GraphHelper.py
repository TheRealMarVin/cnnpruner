import numpy as np
import torch
import re


def get_childs(graph_edges, parent_name):
    if parent_name in graph_edges:
        return graph_edges[parent_name]
    return None


def get_parents(graph_edges, child_name):
    result = []
    for parent_name, values in graph_edges.items():
        for name, _ in values:
            if name == child_name:
                result.append(parent_name)
                break
    return result


def generate_graph(model, args):
    graph = {}

    # Run the Pytorch graph to get a trace and generate a graph from it
    trace, out = torch.jit.get_trace_graph(model, args)
    torch.onnx._optimize_trace(trace, torch.onnx.OperatorExportTypes.ONNX)
    torch_graph = trace.graph()

    model_name = model._get_name()
    root = None
    for torch_node in torch_graph.nodes():
        inputs = [o.unique() for o in torch_node.inputs()]
        outputs = [o.unique() for o in torch_node.outputs()]

        # Get output shape
        shape = get_shape(torch_node)

        # Add edges
        curr_name = reformat_path(model_name, torch_node.scopeName())
        sub_layers = []
        for target_torch_node in torch_graph.nodes():
            target_inputs = [i.unique() for i in target_torch_node.inputs()]
            target_output = [o.unique() for o in target_torch_node.outputs()]
            target_name = reformat_path(model_name, target_torch_node.scopeName())

            # if "downsample" in target_name or "downsample" in curr_name:
            intersect = set(outputs) & set(target_inputs)
            # print("Line i{} - o{} + &{}: \n\tcurr: {} \n\tnext: {}".format(target_inputs, outputs, intersect, curr_name, target_name))
            if intersect and shape is not None:
                if curr_name == "":
                    curr_name = str(inputs)
                if target_name == "":
                    target_name = str(target_inputs)
                if root is None:
                    root = curr_name #TODO this may be absolutely wrong
                # print("Line {} - {} + {}: \n\tcurr: {} \n\tnext: {}".format(target_inputs, outputs, intersect, curr_name, target_name))
                print("Line: \n\tcurr: {} \n\tnext: {}\n\tshape:{}".format(curr_name, target_name, shape))
                sub_layers.append((target_name, shape))

        if len(sub_layers) > 0:
            graph[curr_name] = sub_layers
    return graph, root


def reformat_path(model_name, entry):
    if len(entry) == 0:
        return ""
    result = entry
    if entry.startswith(model_name):
        result = entry[len(model_name):]

    if len(result) > 0:
        result_as_array = result.split("/")
        names = []
        for val in result_as_array:
            m = re.search(r"(?<=\[)(?=\w*\])\w*", val)
            if m:
                names.append(m.group(0))

        if len(names) > 0:
            result = ".".join(names)
    return result


def get_shape(torch_node):
    # TODO get this method on github, but the regex is not efficient at all
    m = re.match(r".*Float\(([\d\s\,]+)\).*", str(next(torch_node.outputs())))
    if m:
        shape = m.group(1)
        shape = shape.split(",")
        shape = tuple(map(int, shape))
    else:
        shape = None
    return shape

