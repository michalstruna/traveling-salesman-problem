import numpy as np
from queue import PriorityQueue
import sys

def find(vertice, parent):
    if parent[vertice] != vertice:
        parent[vertice] = find(parent[vertice], parent)
    return parent[vertice]

def union(vertice1, vertice2, rank, parent):
    root1 = find(vertice1, parent)
    root2 = find(vertice2, parent)
    if root1 != root2:
        if rank[root1] > rank[root2]:
            parent[root2] = root1
    else:
	    parent[root1] = root2
    if rank[root1] == rank[root2]: rank[root2] += 1

class Path:

    @staticmethod
    def get_cost(path, distances):
        fitness = 0

        for i in range(len(path) - 1):
            fitness += Path.get_action_cost((path[i], path[i + 1]), distances)

        return fitness

    @staticmethod
    def get_action_cost(action, distances):
        return distances[action[0], action[1]] if action[0] < action[1] else distances[action[1], action[0]]

    @staticmethod
    def get_min_spanning_tree_size(start, vertices, distances):
        edges = Path.get_min_spanning_tree(vertices, distances)

        size = 0

        for edge in edges:
            size += edge[0]

        i, dist = Path.get_nearest(start, vertices, distances)

        if i != -1:
            size += dist

            if len(vertices) > 1:
                vertices.remove(i)
                
            i, dist = Path.get_nearest(0, vertices, distances)
            size += dist

        return size

    @staticmethod
    def get_min_spanning_tree(vertices, distances):
        parent = {}
        rank = {}
        edges = []

        for i in vertices:
            for j in vertices:
                if i > j:
                    action = (i, j)
                    edges.append((Path.get_action_cost(action, distances), action))

        edges.sort()

        for vertice in vertices:
            parent[vertice] = vertice
            rank[vertice] = 0

        min_spanning_tree = set()

        for edge in edges:
            cost, edge_vertices = edge

            if find(edge_vertices[0], parent) != find(edge_vertices[1], parent):
                union(edge_vertices[0], edge_vertices[1], rank, parent)
                min_spanning_tree.add(edge)

        return min_spanning_tree

    @staticmethod
    def get_nearest(start, targets, distances):
        distance = sys.maxsize
        i = -1

        for city in targets:
            dist = Path.get_action_cost((start, city), distances)

            if dist < distance:
                distance = dist
                i = city

        if distance == sys.maxsize:
            distance = 0

        return i, distance

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
    def tmp(vertices, start, end, distances):
        distance = 0
        s = len(vertices)
        current, distance = Path.get_nearest(start, vertices, distances)

        while vertices:
            i, dist = Path.get_nearest(current, vertices, distances)
            distance += dist
            current = i
            vertices.remove(i)

        distance += Path.get_action_cost((current, 0), distances)
        return distance / 1.2

    @staticmethod
    def tmp2(cities, distances):
        #while True:
        #    pass

        sum = 0
        count = 0

        for i in cities:
            sum += np.sum(distances[i, :]) + np.sum(distances[:, i])
            count += distances.shape[0] + 1

        return (sum / count if count > 0 else 0)

    @staticmethod
    def create(state, distances, parent = None, action = None):
        id = State.to_string(state)
        current = 0 if not action else action[1]
        unvisited = Path.get_unvisited(state, distances)
        nearest, nearest_dist = Path.get_nearest(current, unvisited, distances)
        path_eval = Path.get_min_spanning_tree_size(current, unvisited, distances)
        path_cost = 0

        if parent:
            path_cost = Path.get_cost(state, distances)#parent[Node.PATH_COST] + Path.get_action_cost(action, distances)
        
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

    def get_expanded_states_count(self):
        return len(self.explored)

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

                    actions = []
                    current = successor

                    while current[Node.ACTION] != None:
                        print(current[Node.PATH_COST], current[Node.PATH_EVAL], current[Node.PATH_COST] + current[Node.PATH_EVAL], current[Node.ACTION], Path.get_action_cost(current[Node.ACTION], distances))
                        current = current[Node.PARENT]

                    print(current[Node.PATH_COST], current[Node.PATH_EVAL], current[Node.PATH_COST] + current[Node.PATH_EVAL])

                    return successor[Node.STATE], successor[Node.PATH_COST]

                Node.add(successor, self.fringe)