import torch
import re

def generate_graph(model, args):
    edges = {}

    # Run the Pytorch graph to get a trace and generate a graph from it
    trace, out = torch.jit.get_trace_graph(model, args)
    torch.onnx._optimize_trace(trace, torch.onnx.OperatorExportTypes.ONNX)
    torch_graph = trace.graph()

    model_name = model._get_name()


    # def add_edge_by_id(vid1, vid2, label=None):
    #     edges.append((vid1, vid2, label))

    root = None
    for torch_node in torch_graph.nodes():
        outputs = [o.unique() for o in torch_node.outputs()]

        # Get output shape
        shape = get_shape(torch_node)

        # Add edges
        curr_name = refornat_path(model_name, torch_node.scopeName())
        sub_layers = []
        for target_torch_node in torch_graph.nodes():
            target_name = refornat_path(model_name, target_torch_node.scopeName())
            target_inputs = [i.unique() for i in target_torch_node.inputs()]
            if set(outputs) & set(target_inputs):
                if root is None:
                    root = curr_name #TODO this may be absolutely wrong
                if len(curr_name) > 0 and len(target_name) > 0:
                    print("Line {}: \n\tcurr: {} \n\tnext: {}".format(target_inputs, curr_name, target_name))
                    #add_edge_by_id(curr_name, target_name, shape)
                    sub_layers.append((target_name, shape))

        if len(sub_layers) > 0:
            edges[curr_name] = sub_layers
    return edges, root


def refornat_path(model_name, entry):
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

