import re
from glyles import Glycan, convert, convert_generator
from IPython.display import Image
from glyles.glycans.mono.monomer import Monomer

# ********************************************
def replace_wildcards(input_string):
    """
    input_string:  glycan string in its IUPAC condensed notation
    replace bonds like (?1-?) -> (a1-999)
    replace bonds like (a1-?) -> (a1-998)
    returns full glycan string with updated wildcards bond configuration"""

    # Replace patterns like (?1-?), (?2-?), (?4-?) with (a1-999)
    input_string = re.sub(r'\(\?(\d+)-\?\)', '(a1-999)', input_string)

    # Replace patterns like (a1-?), (b2-?), etc with (a1-998), (b2-998), etc
    input_string = re.sub(r'\((\w)(\d+)-\?\)', lambda m: f'({m.group(1)}{m.group(2)}-998)', input_string)

    return input_string

# ***********************************************

def readable_glycans(glycan):
    """
    glycan: glycan string in it IUPAC condensed notation
    returns glycans which are readable by Glyles package"""

    global current_index
    #i = 0
    try:
        glycan = Glycan(glycan, tree_only= True)

        Image(glycan.save_dot("viz_toy.dot").create_png())   

        if current_index % 1000 == 0:
            print(current_index)    
        return True

    except Exception as e:        
        print(f"error processing this glycan '{glycan}: {e}")
        if current_index % 1000 == 0:
            print(current_index) 
        return False
    
    finally: current_index += 1


# *********************************************


def fun1(parsed_tree, node, predecessor, info_dic, monomer_list):
     """
     function that update initialized dictionary
     parsed_tree = graph/tree of glycan
     node: whose children are extracted
     predecessor: parent nodes of node "node"
     info_dic: dictionary that being updated (initialized as dic)
     monomer_list: list of monomer in the glycan in their IUPAC notation
     returns: updated data structure of a glycan at particular depth in its tree"""
     
     #target = graph.nodes[node]['label']
     target = Monomer(monomer_list[node]).get_name(full=True)
     sources = predecessor
     for i in sources:
          visit = set()
          visit.add(i)
          key_set = set()
          if len(sources) > 1:
               for j in range(len(sources)):
                   
                    if sources[j] not in visit:
                         edges = [(u, v, d) for u, v, d in parsed_tree.edges(data=True) if u == node and v == sources[j]]
                         key_set.add((Monomer(monomer_list[sources[j]]).get_name(full=True), list(edges[0][2].values())[0]))

               s = frozenset(key_set)
               edges = [(u, v, d) for u, v, d in parsed_tree.edges(data=True) if u == node and v == i]
               if (Monomer(monomer_list[i]).get_name(full=True), target, s) not in info_dic:
                info_dic[(Monomer(monomer_list[i]).get_name(full=True), target, s)] = {}
                info_dic[(Monomer(monomer_list[i]).get_name(full=True), target, s)][list(edges[0][2].values())[0]] = 1

               else:
                if list(edges[0][2].values())[0] in info_dic[(Monomer(monomer_list[i]).get_name(full=True), target, s)]:
                     info_dic[(Monomer(monomer_list[i]).get_name(full=True), target, s)][list(edges[0][2].values())[0]] += 1

                else:
                     info_dic[(Monomer(monomer_list[i]).get_name(full=True), target, s)][list(edges[0][2].values())[0]] = 1

          elif len(sources) == 1:
              s = frozenset()
              edges = [(u, v, d) for u, v, d in parsed_tree.edges(data=True) if u == node and v == i]
              if (Monomer(monomer_list[i]).get_name(full=True), target, s) not in info_dic:
                info_dic[(Monomer(monomer_list[i]).get_name(full=True), target, s)] = {}                
                info_dic[(Monomer(monomer_list[i]).get_name(full=True), target, s)][list(edges[0][2].values())[0]] = 1

              else:
                if list(edges[0][2].values())[0] in info_dic[(Monomer(monomer_list[i]).get_name(full=True), target, s)]:
                     info_dic[(Monomer(monomer_list[i]).get_name(full=True), target, s)][list(edges[0][2].values())[0]] += 1

                else:
                     info_dic[(Monomer(monomer_list[i]).get_name(full=True), target, s)][list(edges[0][2].values())[0]] = 1

          else:
              print("This node does not have any predecessor")
             
     return info_dic


# ***********************************************
def dfs_traversal(graph, node, visited, result, dic, monomer_list):    
    """
    dfs: depth first search for the tree
    graph: graph/tree of the glycan
    node: node whose children are extracted
    visited: set of node which are already visited
    result: returned result of dfs function when running iteratively
    dic: initialized data structure as a dictionary
    monomer_list: list of monomer in the glycan in their IUPAC notation
    returns full data structure of glycans"""

    if node not in visited:
        visited.add(node)
        predecessors = list(graph.successors(node))
        """
        Note: glycan tree has directed edges going from leaves till root
        However, glyles package read them as a standard tree (edges dicrection from root to leaves)
        therefore successor of a node in standard tree sense is actually a predecessor in molecule sense"""
        full_struct = fun1(graph, node, predecessors, dic, monomer_list)
        
        for predecessor in predecessors:
            edge_label = graph.get_edge_data(predecessor, node) #.get('label', '')
            dfs_traversal(graph, predecessor, visited, result, dic, monomer_list)
            result.append((node, graph.nodes[node].get('label', ''), edge_label))

    return full_struct


# ********************************************

def new_wildcard_dict(data_structure, p_threshold = 0.75, n_threshold = 10):
    """
    data_structure: data structure for glycans
    p_threshold: probability of occuring of a known bond configuration (default value 0.75)
    n_threshold: frequency of occurence of a known bond configuration (default value 10)
    returns: updated data structure with all wildcards replaced"""
    to_remove = []
    for k, v in data_structure.items():
        total = sum(v1 for k1, v1 in v.items() if "999" not in k1 and "998" not in k1 and len(k1.split('-')[0].strip('(')) > 1)

        if total < n_threshold:
            to_remove.append(k)

        else:
            prob_list = []
            for k1, v1 in v.items():

                if '999' not in k1 and '998' not in k1 and len(k1.split('-')[0].strip('(')) > 1:
                    prob_list.append(v1 / total)

            prob = max(prob_list)#[-1]
            if prob < p_threshold:
                to_remove.append(k)

    for key in to_remove:
        del data_structure[key]

    #print(data_structure)

    to_remove = []
    for k, v in data_structure.items():
        prob_list = []
        for k1, v1 in v.items():
            if "998" not in k1:
                pass
            elif "998" in k1:
                bond_configuration = k1.split('-')[0].strip('(')
                bond_configuration_values = list(v2 for k2, v2 in v.items() if k2.split('-')[0].strip('(') == bond_configuration and "998" not in k2 and "999" not in k2)
                #print(bond_configuration_values)
                max_value = max(bond_configuration_values)
                total = sum(bond_configuration_values)
                #print(max_value,total)
                if total < n_threshold:
                    to_remove.append(k)
                else:
                    bond_configuration_values_prob = [x / total for x in bond_configuration_values]
                    if max(bond_configuration_values_prob) < p_threshold:
                        to_remove.append(k)

    for key in to_remove:
        del data_structure[key]
            



    return data_structure


# ***************************************************

def remove_wildcards_from_datastructure(data_structure):
    """
    this function replaces wildcrds with known based p_threshold and n_threshold
    data_structure: updated data structure"""

    for key, value in data_structure.items(): 
        to_remove = []
        # key: ('Fuc', 'Gal', frozenset())
        # value: {'(a1-2)': 29, '(a1-998)': 5, '(a1-3)': 1}
        max_value1 = max(v1 for k1, v1 in value.items() if "999" not in k1 and "998" not in k1)
        #key_with_max_value1 = max(value, key=value.get)
        key_with_max_value1 = [k1 for k1, v1 in value.items() if v1 == max_value1]
        for key1, value1 in value.items():
            # key1: '(a1-2)'
            #  value1: 29
            if "999" in key1:
                value[key_with_max_value1[0]] += value1
                to_remove.append(key1)
                #del value[key1]

            elif "998" in key1:
                bond_configuration = key1.split('-')[0].strip('(')
                max_value2 = max(v1 for k1, v1 in value.items() if k1.split('-')[0].strip('(') == bond_configuration and "998" not in k1 and "999" not in k1)
                key_with_max_value2 = [k1 for k1, v1 in value.items() if v1 == max_value2]
                value[key_with_max_value2[0]] += value1
                to_remove.append(key1)

        #print(data_structure)
        #print(to_remove)

        for key2 in to_remove:
        #    print(key2)
            del value[key2]

    return data_structure



# ****************************************
# this function updates the data structure of glycan strings one by one.
# inputs are:
# 1. glycan string 
# 2. tree of this glycan string
# 3. the whole  data structure
# 4. index of the glycan string in the main data
def update_glycans(tree, glycan_string, data_structure, index, monomer_list):
    """
    this function updates individual data structures of glycans
    tree: tree of a glycan string
    glycan_string: glycan string in its IUPAC notation
    data_structure: the main data structure
    index: index of glycan in dataframe
    monomer_list: list of monomers"""    
    
    result = []
    visited = set()

    roots = [node for node in tree.nodes  if tree.in_degree(node) == 0]
    root = roots[0]
    result = []
    visited = set()
    dic = {}
    #print("info for known glycans")
    #if gly.find("(a1-999)") == -1:
    info = dfs_traversal(tree, root, visited, result, dic, monomer_list) 
    #print(info)
    unkown_keys_info = set()
    for key_info, value_info in info.items():
        if "999" not in str(key_info) and "998" not in str(key_info):
            if "999" in str(value_info) or "998" in str(value_info):
                unkown_keys_info.add(key_info)


    keys_data_structure = set(data_structure.keys())
    keys_not_in_data_structure = unkown_keys_info - keys_data_structure
    #print(keys_not_in_data_structure)
    if keys_not_in_data_structure:
        return 0


    # info is a dictionary
    # data_structure is a dictionary
    for key1, value1 in info.items():

        if "999" in str(key1) or "998" in str(key1):
            #print("we could not find it in the wild cards dictionary")
            continue

        total_bond_sum_before = sum(value1.values())
        
        # value1 is from individual glycan {'(b1-4)': 3, '(b1-998)': 1, '(a1-999)': 1}
       
        has_not_998_999 = list(k for k in value1.keys() if "999" in k or "998" in k)
        if not has_not_998_999:
            #print("we will pass")
            
            continue
        
        else:
            # value is from main data_structure
            value = data_structure[key1]
            mod_dic = {}

            for k, v in value1.items():
                
                if "999" not in k and "998" not in k and k not in mod_dic:
                    mod_dic[k] = v
                    continue

                elif "999" not in k and "998" not in k and k in mod_dic:
                    mod_dic[k] += v
                    continue

                    
                if "999" in k:
                    key_with_max_value = max(value, key=value.get)
                    
                elif "998" in k:
                    #print("998 is in there")
                    #to_remove.append(k)
                    bond_configuration = k.split('-')[0].strip('(')
                    filtered_keys = [keys for keys in value.keys() if bond_configuration in keys]
                    key_with_max_value = max(filtered_keys, key = lambda k: value[k])

                if key_with_max_value not in mod_dic:
                    mod_dic[key_with_max_value] = v
                else:
                    mod_dic[key_with_max_value] += v
                #print(mod_dic)                

            info[key1] = mod_dic
            #print(value1)
            #print(mod_dic)
            total_bond_sum_after = sum(mod_dic.values())
            if total_bond_sum_before != total_bond_sum_after:
                print("there no consistency")
                return 2
        
    return info


# *********************************************

def update_graph(graph, gly_data_structure, visited, node, monomer_list):
    """
    this function updates graph of glycan string
    graph: tree/graph (graph or tree are interchangable) of glycan string
    gly_data_structure: data structure of glycan string
    visited: nodes that are visited
    node: parent node (initially root node)
    monomer_list: list of monomers
    returns: fully updated tree"""
    
    if node not in visited:
        #print('not visited')
        visited.add(node)
        #print(visited)
        predecessors = list(graph.successors(node))
        #print(predecessors)
        
        g = graph_bond_update(graph, node, predecessors, gly_data_structure, monomer_list)
        

        for predecessor in predecessors:
            #print(predecessor)
            if g == 0:
                #print("g is zero")
                return 0
            else:
                g = update_graph(g, gly_data_structure, visited, predecessor, monomer_list)

    return g


# ***********************************************

def graph_bond_update(graph, node, predecessor, gly_data_structure, monomer_list):
    """
    this function updates bond configurations (wildcard edges) between nodes
    graph, node, gly_data_structure, monomer: defined above
    predecessor: parents of node "node" (since edges in our tree edges are directed from downside up)
    it updates an edge and returns updated tree"""

    if not len(predecessor):
        #print("no predecessor for ", node)
        return graph 
    
    target = Monomer(monomer_list[node]).get_name(full=True)
    sources = predecessor.copy()
    #print(target)
    #print(sources)

    
    wild_cards_sources = list(i for i in ((sources)) if "999" in list([(u, v, d) for u, v, d in graph.edges(data=True) if u == node and v == i][0][2].values())[0] or 
                              "998" in list([(u, v, d) for u, v, d in graph.edges(data=True) if u == node and v == i][0][2].values())[0])
    

    if len(wild_cards_sources) > 1:
        #print("not resolvable", node)
        return 0
    
    elif len(wild_cards_sources) == 0:
        #print("no wildcard", node)
        return graph
    
    else:
        
        #print("wildcard and resolvable", node)
        sources.remove(wild_cards_sources[0])
        #print(sources)        
        s = set()
        for i in sources:
            s.add((Monomer(monomer_list[i]).get_name(full=True), list([(u, v, d) for u, v, d in graph.edges(data=True) if u == node and v == i][0][2].values())[0]))
        #print(s)
        s = frozenset(s)
        #print(s)
        key = (Monomer(monomer_list[wild_cards_sources[0]]).get_name(full=True), target, s)
        
        if key in gly_data_structure:
            #print("yes, key is in the data structure")
            value = gly_data_structure[key]
            
            if "999" in list([(u, v, d) for u, v, d in graph.edges(data=True) if u == node and v == wild_cards_sources[0]][0][2].values())[0]:
                
                max_key = max(value, key=value.get)                

            elif "998" in list([(u, v, d) for u, v, d in graph.edges(data=True) if u == node and v == wild_cards_sources[0]][0][2].values())[0]:
                    #print("wildcard type is 998")
                    bond_configuration = list([(u, v, d) for u, v, d in graph.edges(data=True) if u == node and v == wild_cards_sources[0]][0][2].values())[0].split('-')[0].strip('(')
                    #print(bond_configuration)
                    max_value2 = max(list(v1 for k1, v1 in value.items() if k1.split('-')[0].strip('(') == bond_configuration and "998" not in k1 and "999" not in k1))
                    #print(max_value2)
                    max_key = [k1 for k1, v1 in value.items() if v1 == max_value2]
                    max_key = max_key[0]
                    #print(max_key)
            edge_attrs = graph.get_edge_data(node, wild_cards_sources[0])
            
            edge_attrs['type'] = max_key
    
            return graph

# ****************************************

# this function is dfs on the tree of a glycan string
# it reads each node and their respective edge labels and forming a string from the tree
def dfs_traversal_for_string(graph, node, visited, string, monomer_list):
    """
    this function scan the glycan string. Scanning is done from left to right
    NOTE: most right monomer is the root node and has predecessors as we go down because our tree edges are directed from downside up
    graph, node, visited, monomer_list: defined above
    string: glycan string in IUPAC condensed notation
    returns: updated glycan string without wildcards"""
    
    if node not in visited:
        #print('not visited')
        visited.add(node)
        #print(visited)
        predecessors = list(graph.successors(node))
        #print(predecessors)
        string = update_string(graph, node, predecessors, string, monomer_list)
        #print("updated string is: ", string)
        

        for predecessor in predecessors:
            #print(string)
            string = list([(u, v, d) for u, v, d in graph.edges(data=True) if u == node and v == predecessor][0][2].values())[0] + string
            #print(string)
            string = Monomer(monomer_list[predecessor]).get_name(full=True) + string
            #print(string)
            string = dfs_traversal_for_string(graph, predecessor, visited, string, monomer_list)

    return string

# ********************************************
def update_string(graph, node, predecessor, string, monomer_list):
    """
    this function update the string while scanning
    graph, node, monomer_list: defined above
    predecessor: parents of node "node" 
    string: glycan string
    returns: updated substring of glycan string"""

    if not len(predecessor):
        #print("this node does not have any predecessor")
        position_open_bracket = string.find("]")
        position_close_bracket = string.find("[")
        #print(position_open_bracket, position_close_bracket)
        #print(type(position_close_bracket))

        if position_open_bracket == -1 and position_close_bracket == -1:
            #print("there is no open or close bracket")
            return string
        
        elif position_open_bracket > 0 and position_close_bracket > 0:
            #print("there are both brackets")
            if position_open_bracket < position_close_bracket:
                string = "[" + string
                return string
            else: 
                open_positions = [index for index, char in enumerate(string) if char == "]"]
                close_position = [index for index, char in enumerate(string) if char == "["]
                if len(open_positions) > len(close_position):
                    string = "[" + string
                return string

                #return string

        elif position_open_bracket > 0 and position_close_bracket == -1:
            #print("only open bracket")
            string = "[" + string

            return string
    
    target = Monomer(monomer_list[node]).get_name(full=True)
    sources = predecessor.copy()
    #print(target)
    #print(sources)

    if len(sources) == 1:
        #string = list([(u, v, d) for u, v, d in graph.edges(data=True) if u == node and v == sources[0]][0][2].values())[0] + string
        #string = Monomer(monomer_list[sources[0]]).get_name(full=True) + string
        return string
    
    elif len(sources) > 1:
        open_bracket = "]"
        string = open_bracket + string
        return string
    

# *************************************
def iupac_to_smiles(iupac_string, index):
    """
    this function converts glycan string from IUPAC -> smiles
    iupac_string: glycan string
    index: index of glycan in dataframe (just to keep the track of code)
    returns: smiles string of iupac_string"""
    #print('yes this function is running')
    glycan = Glycan(iupac_string)
    conversion_dict = {iupac_string : glycan.get_smiles()}
    if index % 1000 == 0:
        print(index)
    return conversion_dict.get(iupac_string, 'Unknown')

