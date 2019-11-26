import numpy as np
from queue import PriorityQueue
import sys

class Graph:

    @staticmethod
    def get_nearest_vertice(start, vertices, distances, exclude = set()):
        distance = sys.maxsize
        i = -1

        for city in vertices:
            dist = distances[start, city]

            if dist < distance:
                distance = dist
                i = city

        if distance == sys.maxsize:
            distance = 0

        return i, distance

    @staticmethod
    def estimate_shortest_path(vertices, start, distances):
        distance = 0
        current = start
        path = [current]

        while vertices:
            i, dist = Graph.get_nearest_vertice(current, vertices, distances)
            distance += dist
            current = i
            path.append(current)
            vertices.remove(i)

        return path, distance

    @staticmethod
    def find(parent, i): 
        return i if parent[i] == i else Graph.find(parent, parent[i])

    @staticmethod
    def concat(parent, rank, x, y): 
        parent_x = Graph.find(parent, x) 
        parent_y = Graph.find(parent, y) 
  
        if rank[parent_x] == rank[parent_y]:
            parent[parent_y] = parent_x 
            rank[parent_x] += 1
        elif rank[parent_x] < rank[parent_y]: 
            parent[parent_x] = parent_y
        elif rank[parent_x] > rank[parent_y]: 
            parent[parent_y] = parent_x 
        
    @staticmethod
    def get_min_spanning_tree_size(vertices, distances):
        verts = {}
        edges = []
        
        for vert in vertices:
            if vert not in verts:
                verts[vert] = len(verts)

        for i in vertices:
            for j in vertices:
                if i > j:
                    edges.append((distances[i, j], verts[i], verts[j]))
        
        min_spanning_tree = Graph.get_min_spanning_tree(len(verts), edges)

        sum = 0
        for cost, v1, v2 in min_spanning_tree:
            sum += cost

        return sum

    @staticmethod
    def get_min_spanning_tree(vertices_count, edges): 
        edges.sort()
        result = []
        edge_i = 0 
        spanning_tree_i = 0
  
        parents = []
        rank = [] 
  
        for node in range(vertices_count): 
            parents.append(node) 
            rank.append(0) 

        while spanning_tree_i < vertices_count -1: 
            cost, v1, v2 =  edges[edge_i] 
            parent_1 = Graph.find(parents, v1) 
            parent_2 = Graph.find(parents, v2) 
  
            if parent_1 != parent_2: 
                spanning_tree_i += 1     
                result.append(edges[edge_i]) 
                Graph.concat(parents, rank, parent_1, parent_2)    

            edge_i += 1        

        return result
                

class Path:

    @staticmethod
    def get_cost(path, distances):
        fitness = 0

        for i in range(len(path) - 1):
            fitness += distances[path[i], path[i + 1]]

        return fitness

    @staticmethod
    def get_action_cost(action, distances):
        return distances[action]

    @staticmethod
    def get_unvisited(state, distances):
        unvisited = []

        for i in range(distances.shape[0]):
            if i not in state:
                unvisited.append(i)

        return unvisited

class State:

    @staticmethod
    def equals(state1, state2, only_position = False):
        return state1[0] == state2[0] and state1[1] == state2[1] and (only_position or state1[2] == state2[2])

    @staticmethod
    def to_string(state):
        return ",".join(map(str, state))

    @staticmethod
    def apply_action(state, action):
        newState = state.copy()
        newState.append(action[1])
        return newState

class Node:

    ID = 0
    STATE = 1
    PARENT = 2
    ACTION = 3
    PATH_EVAL = 4
    PATH_COST = 5

    @staticmethod
    def create(state, distances, parent = None, action = None):
        id = State.to_string(state)
        unvisited = Path.get_unvisited(state, distances)
        path_eval = Graph.get_min_spanning_tree_size(unvisited, distances)
        city, dist1 = Graph.get_nearest_vertice(0, unvisited, distances)
        city, dist2 = Graph.get_nearest_vertice(state[-1], unvisited, distances)
        path_eval += dist1 + dist2
        path_cost = 0

        if parent:
            path_cost = parent[Node.PATH_COST] + distances[action]
        
        return (id, state, parent, action, path_eval, path_cost)

    @staticmethod
    def add(node, queue):
        queue.put((node[Node.PATH_COST] + node[Node.PATH_EVAL], node))

    @staticmethod
    def get(queue):
        return queue.get()[1]

    @staticmethod
    def get_successors(node, distances, explored):
        if len(node[Node.STATE]) == distances.shape[0]:
            action = ((node[Node.STATE])[-1], 0)
            newState = State.apply_action(node[Node.STATE], action)
            successor = Node.create(newState, distances, node, action)
            yield successor
        else:
            for i in range(distances.shape[0]):
                if i not in node[Node.STATE]:
                    action = ((node[Node.STATE])[-1], i)
                    newState = State.apply_action(node[Node.STATE], action)
                    successor = Node.create(newState, distances, node, action)

                    if successor[Node.ID] not in explored:
                        yield successor

    @staticmethod
    def get_path(node):
        actions = []
        current = node

        while current[Node.ACTION] != None:
            actions.insert(0, current[Node.ACTION])
            current = current[Node.PARENT]

        return actions


class PathFinder:

    def get_path(self, distances):
        self.explored = set()
        self.fringe = PriorityQueue()
        init = [0]
        Node.add(Node.create(init, distances), self.fringe)

        while not self.fringe.empty():
            node = Node.get(self.fringe)
            self.explored.add(node[Node.ID])

            for successor in Node.get_successors(node, distances, self.explored):
                if len(successor[Node.STATE]) == distances.shape[0] + 1:
                    return successor[Node.STATE], successor[Node.PATH_COST]

                Node.add(successor, self.fringe)