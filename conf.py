def gen_statement(path, data, features):

    nodes_on_path = path.split("+")
    statement = ""

    for node_num, node in enumerate(nodes_on_path):

        if node_num > 0:

            prev_node = nodes_on_path[node_num - 1]

            feature = data['lnr']['tree_']['nodes'][int(node)-1]['split_mixed']['parallel_split']['feature']
            threshold = data['lnr']['tree_']['nodes'][int(node)-1]['split_mixed']['parallel_split']['threshold']

            if int(node) == int(prev_node) + 1:
                equality = "<"

            else:
                equality = ">="

            if feature != 0:
                statement += f"(features['{features.columns[feature-1]}'] {equality} {threshold})"

            else:
                statement += f"({0} {equality} {threshold})"

            if int(node) != max([int(i) for i in nodes_on_path]):
                statement += " & "

    return statement

def gen_paths(data):

    paths = []

    for child in [1]:

        node_1 = data['lnr']['tree_']['nodes'][child-1]
        node_id1 = node_1['id']

        children = sorted([node_1['upper_child'], node_1['lower_child']])

        if children[0] == -2:
            id = f"{node_id1}"
            paths.append(id)
            break

        for child in children:
            node_2 = data['lnr']['tree_']['nodes'][child-1]
            node_id2 = node_2['id']
            children = sorted([node_2['upper_child'], node_2['lower_child']])

            if children[0] == -2:
                id = f"{node_id1}+" + f"{node_id2}"
                paths.append(id)

            for child in children:

                node_3 = data['lnr']['tree_']['nodes'][child-1]
                node_id3 = node_3['id']
                children = sorted([node_3['upper_child'], node_3['lower_child']])

                if children[0] == -2:
                    id = f"{node_id1}+" + f"{node_id2}+" + f"{node_id3}"
                    paths.append(id)

                for child in children:

                    node_4 = data['lnr']['tree_']['nodes'][child-1]
                    node_id4 = node_4['id']
                    children = sorted([node_4['upper_child'], node_4['lower_child']])

                    if children[0] == -2:
                        id = f"{node_id1}+" + f"{node_id2}+" + f"{node_id3}+" + f"{node_id4}"
                        paths.append(id)

    return paths


def gen_subpaths(paths):

    sub_paths = []

    for path in paths:

        blank_path = ""
        path_list = path.split("+")

        for i in path_list:

            path_to_append = blank_path + f"{i}"
            blank_path += f"{i} + "

            sub_paths.append(path_to_append)

    return [i for i in sub_paths if i != '1']

def gen_features(sub_paths, data, features, orig_feature_size):
    for path in sub_paths:
        statement = gen_statement(path, data, features)
        index_feature = features.loc[eval(statement)].index

        new_feature = [1 if i in index_feature else 0 for i in range(len(features))]

        if sum(new_feature) >= 1:
            features[statement] = new_feature
    return features.iloc[:, orig_feature_size:]

def do_everything(jason, features, orig_features_size):
    paths = gen_paths(jason)
    sub_paths = gen_subpaths(paths)
    rule_features = gen_features(sub_paths, jason, features, orig_features_size)
    return rule_features