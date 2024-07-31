"""
# Ride_hailing match
This is the file for finding the best matches for a given matching pool,
the matching pool should include information of unique number of riders 
and drivers, and their distance. This file can also condider its weight 
and avoid the cases of disallowed matches.

# Import this file to use:
from ride_hailing_match import Match

# To calculate best matches:
match = Match()
matched_pairs = match.match(pool, method = 'Munkres')

# Hints:
pool must be a list, here is an example,
pool = [
    [1.0, 1.0, 192.64], 
    [2.0, 17.0, 322.83], 
    [3.0, 8.0, 463.25], 
    [5.0, 2.0, 602.88], 
    [8.0, 15.0, 989.76], 
    [9.0, 0.0, 490.83], 
    [10.0, 19.0, 125.03], 
    [11.0, 5.0, 284.32], 
    [12.0, 7.0, 552.82], 
    [14.0, 18.0, 412.42]
    ]
"""


import numpy as np

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from scipy.sparse.csgraph import maximum_bipartite_matching

class KMNode(object):
    """Convert nodes to KN nodes format
    
     Give attributes to KN nodes, those attributes includes information about
     whther the node is an exception, matched or visted.

    Attributes:
     id - an int, the index id of this node.
     exception - a int, 0 for not excepted.
     match - an int, the values is the id of the matched node with this node, None for not matched.
     visit - a bool, give information for augmenting path in Hungarian Algorithm.
    """
    def __init__(self, id, exception=0, match=None, visit=False) -> None:
        self.id = id
        self.exception = exception
        self.match = match
        self.visit = visit
        pass
        

class KuhnMunkres(object):
    """Hungarian Algorithm considering edge weights
    
     Calculate highest summed weights of all the matches, the algorithm is to 
     add edges with negative infinite edges to convert this problem to a full 
     Hungarian Match Problem. If want to calculate lowest summed weight, input 
     weights can be inverted.

    Attributes:
     matrix - this is a n-dimension list contains link weights.
     x_nodes, y_nodes - each is am instance of KMNode class, contains information for Hungarian Algorithm.
     minz - an int, the values is the id of the matched node with this node, None for not matched.
     x_length, y_length - according to the input, defining the scale of the weight matrix.
     index_x, index_y - put riders on the left-hand side in the bipartite graph.
     zero_threshold - a small value below which to end the iteration of finding the argumenting path.
    """
    def __init__(self) -> None:
        self.matrix = None
        self.x_nodes = []
        self.y_nodes = []
        self.minz = float('inf') # in case of disallowed link
        self.x_length = 0
        self.y_length = 0
        self.index_x = 0
        self.index_y = 1
        self.zero_threshold = 0.00000001
        pass

    def set_matrix(self, x_y_values: list) -> None:
        """
        Build the link weight matrix.
        
        Parameters:
        x_y_values - give the weight value for allowed links, the rest will be marked as zero (disallowed).
        """
        xs = set()
        ys = set()
        for x, y, value in x_y_values:
            xs.add(x)
            ys.add(y)

        # select the riders to be fully matched
        if len(xs) < len(ys):
            self.index_x = 0
            self.index_y = 1
        else:
            self.index_x = 1
            self.index_y = 0
            xs, ys = ys, xs

        x_dic = {x: i for i, x in enumerate(xs)}
        y_dic = {y: j for j, y in enumerate(ys)}
        self.x_nodes = [KMNode(x) for x in xs]
        self.y_nodes = [KMNode(y) for y in ys]
        self.x_length = len(xs)
        self.y_length = len(ys)

        self.matrix = np.zeros((self.x_length, self.y_length))
        for row in x_y_values:
            x = row[self.index_x]
            y = row[self.index_y]
            value = row[2]
            x_index = x_dic[x]
            y_index = y_dic[y]
            self.matrix[x_index, y_index] = 1/value

        for i in range(self.x_length):
            self.x_nodes[i].exception = max(self.matrix[i, :])

        pass

    def km(self) -> None:
        """
        Kuhn-Munkres Algorithm, convert problem to Hungarian Algorithm.
        Excuting Depth First Search to find argumenting path.
        """
        for i in range(self.x_length):
            while True:
                self.minz = float('inf')
                self.set_false(self.x_nodes)
                self.set_false(self.y_nodes)

                if self.dfs(i):
                    break

                self.change_exception(self.x_nodes, -self.minz)
                self.change_exception(self.y_nodes, self.minz)

        pass

    def dfs(self, i: int) -> bool:
        """
        Depth First Search to find possible argumenting path.

        Parameters:
         i - the number of the unique driver
        
        Returns:
         An boolen is returned, True if stop-iteration criteria is reached.
        """
        x_node = self.x_nodes[i]
        x_node.visit = True
        for j in range(self.y_length):
            y_node = self.y_nodes[j]
            if not y_node.visit:
                t = x_node.exception + y_node.exception - self.matrix[i][j]
                if abs(t) < self.zero_threshold:
                    y_node.visit = True
                    if y_node.match is None or self.dfs(y_node.match):
                        x_node.match = j
                        y_node.match = i
                        return True
                else:
                    if t >= self.zero_threshold:
                        self.minz = min(self.minz, t)
        return False

    def set_false(self, nodes: object) -> None:
        """
        At the begining of km algorithm, initialize visit information for nodes.

        Parameters:
         nodes - a set of nodes with attributes for visited or not.
        """
        for node in nodes:
            node.visit = False

    def change_exception(self, nodes: object, change: float) -> None:
        """
        Mark exceptions for disallowed links.

        Parameters:
         nodes - a set of nodes with some attributes.
         change - indicating how to mark those disallowed link, default to negative infinate.
        """
        for node in nodes:
            if node.visit:
                node.exception += change

    def get_connect_result(self) -> list:
        """
        Get the matched pairs and their weights.

        Returns:
         A list with the result format as [[x_node_id, y_node_id, link_weight], [x_node_id, y_node_id, link_weight]...]
        """
        ret = []
        for i in range(self.x_length):
            x_node = self.x_nodes[i]
            j = x_node.match
            y_node = self.y_nodes[j]
            x_id = x_node.id
            y_id = y_node.id
            value = self.matrix[i][j]

            if self.index_x == 1 and self.index_y == 0:
                x_id, y_id = y_id, x_id
            ret.append([x_id, y_id, value])

        return ret

    def get_max_value_result(self) -> None:
        """
        Get the summed weight of all the matched pairs.

        Returns:
         A value of summed link weights.
        """
        ret = 0
        for i in range(self.x_length):
            j = self.x_nodes[i].match
            ret += self.matrix[i][j]

        return ret


def run_kuhn_munkres(x_y_values: list) -> list:
    """
    This is the main functional method of Kuhn-Munkres Algorithm codes.
    Use this function to calculate (solve) optimised mathes for a bipartite 
    graph match problem considering link weights.

    Parameters:
     x_y_values - give the weight value for allowed links, the rest will be marked as zero (disallowed).

    Returns:
     A list with the result format as [[x_node_id, y_node_id, link_weight], [x_node_id, y_node_id, link_weight]...]
    """
    process = KuhnMunkres()
    process.set_matrix(x_y_values)
    process.km()

    return process.get_connect_result()

class Match:
    """Match riders and drivers with their distance
    
    This is the main functional class of this file, three match method can be used 
    to find matches between riders and drivers, considering distance between them.

    Attributes:
     matrix - this is a n-dimension list contains link weights.
     matrix_csr - link weight matrix in scipy csr type.
     very_large_num, very_small_num - dummy values to mark disallowed links.
    """
    def __init__(self) -> None:
        self.matrix = None
        self.matrix_csr = None
        self.very_large_num = 100000
        self.very_small_num = 0
        
        pass

    def set_matrix(self, x_y_values: list, is_get_max: bool = False) -> None:
        xs = set()
        ys = set()
        for x, y, value in x_y_values:
            xs.add(x)
            ys.add(y)
        
        if len(xs) == 0  or len(ys) == 0:
            return None

        x_length = int(max(xs) + 1)
        y_length = int(max(ys) + 1)
        if not is_get_max:
            self.matrix = np.ones((x_length, y_length)) * self.very_large_num
        else:
            self.matrix = np.ones((x_length, y_length)) * self.very_small_num
        for row in x_y_values:
            x = int(row[0])
            y = int(row[1])
            value = row[2]
            self.matrix[x][y] = value
            
        self.matrix_csr = csr_matrix(self.matrix)

        return self.matrix_csr
    
    def get_result_value(self, graph: list, raw: list, is_munkres: bool = False) -> list:
        result = []
        if is_munkres:
            for row in raw:
                if row[2] != 0:
                    sub = [row[0], row[1], 1/row[2]]
                    result.extend([sub])
        else:
            result = []
            for i in range(len(raw)):
                if graph[i][raw[i]] == self.very_large_num or raw[i] == -1:
                    continue
                sub = [i, raw[i], graph[i][raw[i]]]
                result.extend([sub])
        return result
    
    def match(self, match_pool: list, method: str = 'Max') -> list:
        """
        This is the main functional method of this file, three match method can be used 
        to find matches between riders and drivers, considering distance between them.

        Parameters:
         match_pool - give the weight value for allowed links.
         method - defines the method or basis (optimising direction) to find mathes, can be:
            'Max': Max bipartite matching
            'Min': Min weight full bipartite matching
            'Munkres': Min weight bipartite matching with disallowed links
        
        Returns:
         A list with the result format as [[x_node_id, y_node_id, link_weight], [x_node_id, y_node_id, link_weight]...]
        """

        if method == 'Munkres':
            raw_result = run_kuhn_munkres(match_pool)
            result = self.get_result_value(self.matrix, raw_result, True)
        elif method == 'Max':
            graph = self.set_matrix(match_pool, True)
            if graph == None:
                return []
            raw_result = maximum_bipartite_matching(graph, perm_type='column')
            result = self.get_result_value(self.matrix, raw_result)
        elif method == 'Min':
            graph = self.set_matrix(match_pool, False)
            if graph == None:
                return []
            raw_result_row, raw_result_col = min_weight_full_bipartite_matching(graph)
            result = self.get_result_value(self.matrix, raw_result_col)

        return result



