import numpy as np
import torch
import re


def get_childs(graph_edges, parent_name):
    res = []
    if parent_name in graph_edges:
        for name, _ in graph_edges[parent_name]:
            res.append(name)
    return res


# def get_parents(graph_edges, child_name):
#     result = []
#     for parent_name, values in graph_edges.items():
#         for name, _ in values:
#             if name == child_name:
#                 if name in get_childs(graph_edges, parent_name):
#                     result.append(parent_name)
#                     break
#     return result

# def get__input_connection_count_per_entry(graph_edges, node, res):
#     if node in graph_edges:
#         for name, _ in graph_edges[node]:
#             if name in res.keys():
#                 res[name] += 1
#             else:
#                 res[name] = 1
#             get__input_connection_count_per_entry(graph_edges, name, res)

# def generate_graph_old(model, args):
#     graph = {}
#
#     # Run the Pytorch graph to get a trace and generate a graph from it
#     trace, out = torch.jit.get_trace_graph(model, args)
#     torch.onnx._optimize_trace(trace, torch.onnx.OperatorExportTypes.ONNX)
#     torch_graph = trace.graph()
#
#     model_name = model._get_name()
#     root = None
#     for torch_node in torch_graph.nodes():
#         inputs = [o.unique() for o in torch_node.inputs()]
#         outputs = [o.unique() for o in torch_node.outputs()]
#
#         # Get output shape
#         shape = get_shape(torch_node)
#
#         # Add edges
#         curr_name = reformat_path(model_name, torch_node.scopeName())
#         sub_layers = []
#         for target_torch_node in torch_graph.nodes():
#             target_inputs = [i.unique() for i in target_torch_node.inputs()]
#             target_outputs = [o.unique() for o in target_torch_node.outputs()]
#             target_name = reformat_path(model_name, target_torch_node.scopeName())
#
#             intersect = set(outputs) & set(target_inputs)
#             if intersect and shape is not None:
#                 if curr_name == "":
#                     curr_name = str(inputs)
#                 if target_name == "":
#                     target_name = str(target_inputs)
#                 if root is None:
#                     root = curr_name #TODO this may be absolutely wrong
#                 # print("Line: \n\tcurr: {} \n\tnext: {}\n\tshape:{}".format(curr_name, target_name, shape))
#                 print("Line: intersect:{} ci{} co{} ti{} to{}\n\tcurr: {} \n\tnext: {}\n\tshape:{}".format(intersect, inputs, outputs, target_inputs, target_outputs, curr_name, target_name, shape))
#                 sub_layers.append((target_name, shape))
#
#         if len(sub_layers) > 0:
#             graph[curr_name] = sub_layers
#     return graph, root

def get__input_connection_count_per_entry(graph_edges, node, res):
    if node in graph_edges:
        for name in graph_edges[node].split(","):
            if name in res.keys():
                res[name] += 1
                break
            else:
                res[name] = 1
            get__input_connection_count_per_entry(graph_edges, name, res)

def generate_graph(model, args):
    execution_graph = {}
    # execution_shapes = {}
    id_name_dict = {}

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
        # sub_layers = []
        for target_torch_node in torch_graph.nodes():
            target_inputs = [i.unique() for i in target_torch_node.inputs()]
            target_outputs = [o.unique() for o in target_torch_node.outputs()]
            # target_name = reformat_path(model_name, target_torch_node.scopeName())

            intersect = set(outputs) & set(target_inputs)
            if intersect and shape is not None:
                if curr_name == "":
                    curr_name = ",".join(str(x) for x in inputs)
                # if target_name == "":
                #     target_name = str(target_inputs)
                intersect_as_string = ",".join(str(x) for x in intersect)
                if root is None:
                    root = intersect_as_string
                # print("Line: \n\tcurr: {} \n\tnext: {}\n\tshape:{}".format(curr_name, target_name, shape))
                # print("Line: intersect:{} ci{} co{} ti{} to{}\n\tcurr: {} \n\tshape:{}".format(intersect, inputs, outputs, target_inputs, target_outputs, curr_name, shape))
                if intersect_as_string in execution_graph:
                    execution_graph[intersect_as_string].extend(np.array(target_outputs))
                else:
                    execution_graph[intersect_as_string] = target_outputs
                id_name_dict[intersect_as_string] = curr_name

    execution_graph = clean_execution_graph(execution_graph)
    return execution_graph, id_name_dict, root

def clean_execution_graph(execution_graph):
    cleaned_graph = {}
    for k, v in execution_graph.items():
        for i in v:
            if str(i) in execution_graph:
                if k not in cleaned_graph:
                    cleaned_graph[k] = [i]
                else:
                    cleaned_graph[k].append(i)

    for k, v in cleaned_graph.items():
        cleaned_graph[k] = ",".join(str(x) for x in cleaned_graph[k])
    return cleaned_graph

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

