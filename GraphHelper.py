import numpy as np
import torch
import re


def get_childs(graph_edges, name):
    res = []
    if name in graph_edges:
        for name, _ in graph_edges[name]:
            res.append(name)
    return res


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
    execution_shapes = {}
    id_name_dict = {}

    # Run the Pytorch graph to get a trace and generate a graph from it
    trace, out = torch.jit.get_trace_graph(model, args)
    torch.onnx._optimize_trace(trace, torch.onnx.OperatorExportTypes.ONNX)
    torch_graph = trace.graph()

    model_name = model._get_name()
    root = None

    max_counter = 0
    for _ in torch_graph.nodes():
        max_counter += 1

    for index, torch_node in enumerate(torch_graph.nodes()):
        inputs = [o.unique() for o in torch_node.inputs()]
        outputs = [o.unique() for o in torch_node.outputs()]

        shape = get_shape(torch_node)
        curr_name = reformat_path(model_name, torch_node.scopeName())
        op = torch_node.kind()

        if index == max_counter - 1:
            ",".join(str(x) for x in outputs)
            execution_graph[intersect_as_string] = []
            if curr_name in id_name_dict.values() and op == "onnx::Gemm":
                new_name = try_correct_broken_name(op, shape, curr_name, model)
                if new_name is not None:
                    curr_name = new_name
            # print("TATA: ", curr_name)
            id_name_dict[intersect_as_string] = curr_name
            execution_shapes[intersect_as_string] = shape
            break

        # print("curr_name: ", curr_name, " \top: ", op)
        for target_torch_node in torch_graph.nodes():
            target_inputs = [i.unique() for i in target_torch_node.inputs()]
            target_outputs = [o.unique() for o in target_torch_node.outputs()]

            intersect = set(outputs) & set(target_inputs)

            # print("Line: \n\tcurr: {} \n\tnext: {}\n\tshape:{}".format(curr_name, str(target_outputs), shape))
            if intersect:
                if curr_name == "":
                    curr_name = ",".join(str(x) for x in inputs)

                intersect_as_string = ",".join(str(x) for x in intersect)
                if root is None:
                    root = intersect_as_string
                # print("Line: \n\tcurr: {} \n\tnext: {}\n\tshape:{}".format(curr_name, target_name, shape))
                # print("Line: intersect:{} ci{} co{} ti{} to{}\n\tcurr: {} \n\tshape:{}".format(intersect, inputs, outputs, target_inputs, target_outputs, curr_name, shape))
                if intersect_as_string in execution_graph:
                    execution_graph[intersect_as_string].extend(np.array(target_outputs))
                else:
                    execution_graph[intersect_as_string] = target_outputs

                if curr_name in id_name_dict.values() and op == "onnx::Gemm":
                    new_name = try_correct_broken_name(op, shape, curr_name, model)
                    if new_name is not None:
                        curr_name = new_name
                id_name_dict[intersect_as_string] = curr_name
                execution_shapes[intersect_as_string] = shape

    execution_graph = clean_execution_graph(execution_graph, execution_shapes, id_name_dict)
    return execution_graph, id_name_dict, root


def try_correct_broken_name(onnx_kind, shape, name, module):
    # this is to fix issues in alexnet that seems to be related to some node combination

    splitted_name = name.split(".")
    desired_path = None
    if len(splitted_name) > 1:
        desired_path = splitted_name[0]
        name = ".".join(splitted_name[1:])
    elif len(splitted_name) == 1:
        name = splitted_name[0]
    else:
        return name

    res = None
    pick_next = False
    for sub_layer, sub_module in module._modules.items():
        if pick_next:
            return sub_layer
        if sub_layer == name:
            pick_next = True
        elif sub_layer == desired_path and \
                sub_module is not None and \
                len(sub_module._modules.items()) > 0:
            res = try_correct_broken_name(onnx_kind, shape, name, sub_module)
            if res is not None:
                res = sub_layer + "." + res
                break
    return res


def get_node_in_model(module, full_name):
    res = None
    splitted_name = full_name.split(".")
    desired_path = None
    if len(splitted_name) > 1:
        desired_path = splitted_name[0]
        name = ".".join(splitted_name[1:])
    elif len(splitted_name) == 1:
        name = splitted_name[0]
    else:
        return res

    for sub_layer, sub_module in module._modules.items():
        if sub_layer == name:
            res = sub_module
            break
        elif sub_layer == desired_path and \
                sub_module is not None and \
                len(sub_module._modules.items()) > 0:
            res = get_node_in_model(sub_module, name)
            if res is not None:
                break

    return res


# TODO could be more streamlined
def clean_execution_graph(execution_graph, execution_shapes, id_name_dict):
    cleaned_graph = {}
    for k, v in execution_graph.items():
        cleaned_graph[k] = []

    for k, v in execution_graph.items():
        temp_list = []
        for i in v:
            if str(i) in execution_graph:
                temp_list.append(i)

        cleaned_graph[k] = ",".join(str(x) for x in temp_list)
    # for k, v in cleaned_graph.items():
    #     cleaned_graph[k] = ",".join(str(x) for x in cleaned_graph[k])

    to_delete = []
    for k, v in cleaned_graph.items():
        if v in execution_shapes:
            while execution_shapes[v] is None:
                to_delete.append(v)
                v = cleaned_graph[v]
            cleaned_graph[k] = v

    for key in to_delete:
        cleaned_graph.pop(key, None)
        id_name_dict.pop(key, None)

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

