import numpy as np
import heapq
from dataclasses import dataclass
from atomic_task import AtomicTask, Room
from edge_task import DFA_Edge, torch, sympy
from ltl_util import formula_to_dfa
from boolean_task import GoalOrientedQLearning

@dataclass
class StateNode:
    """Represents a state with current position in search"""
    state: int
    coord: tuple[float, ...]
    total_cost: float
    path: list[tuple[int, DFA_Edge]]  # [(state, edge_index), ...]
    
    def __lt__(self, other):
        return self.total_cost < other.total_cost
    
    def get_path(self):
        # state: (edge_index, next_state)
        return [(self.path[i][0], self.path[i+1][0], self.path[i+1][1]) for i in range(len(self.path)-1)]

class DFA_Task:
    """
    Dynamic DFA Dijkstra planning algorithm that finds shortest path to accepting states
    considering location-dependent edge costs.
    """
    
    def __init__(self, formula: str, atomic_tasks: dict[str, AtomicTask], name="code_input_task"):
        self.name = name
        self.formula = formula
        self.atomic_tasks = atomic_tasks
        
        dfa, mona = formula_to_dfa(formula, name)
        self.dfa_matrix = dfa[1]
        self.n_states = len(self.dfa_matrix)
        self.accepting_states = [s-1 for s in dfa[0]['accepting_states']]
        self.rejecting_states = set()
        self.policy = self.policy_matrix()
        self.dfa_state = 0

    def __repr__(self):
        dfa_map = {}
        for i, row in enumerate(self.dfa_matrix):
            for j, choices in enumerate(row):
                if choices == (sympy.false,):
                    continue
                dfa_map[(i, j)] = f"Nondeterministic {choices}" if len(choices) > 1 else choices[0]
        return str(dfa_map)
    
    def policy_matrix(self):
        policy = []
        for i, row in enumerate(self.dfa_matrix):
            p_row = []
            for j, choices in enumerate(row):
                if sympy.false in choices:
                    edges = [DFA_Edge(formula) for formula in choices if formula != sympy.false]
                elif sympy.true in choices:
                    edges = [DFA_Edge(sympy.true)]
                else:
                    edges = [DFA_Edge(formula) for formula in choices]
                p_row.append(edges)
            policy.append(p_row)
        return policy 
        
    def find_shortest_path(self, start_state: int, start_coord: tuple, 
                           qmodel: GoalOrientedQLearning, room: Room, coordinate_tolerance: float = 1e-6):
        """
        Find shortest path from start state to any accepting state using Dijkstra's algorithm.
        
        Args:
            start_state: Starting state number
            start_coord: Starting physical coordinate
            coordinate_tolerance: Tolerance for coordinate comparison to avoid infinite loops
            
        Returns:
            Dictionary containing:
            - 'path': List of (state, coordinate) tuples representing the path
            - 'total_cost': Total cost of the path
            - 'final_state': The accepting state reached
            - 'final_coord': Final coordinate reached
            Returns None if no path exists
        """
        # Priority queue: (cost, StateNode)
        pq = [StateNode(start_state, start_coord, 0.0, [(start_state, None)])]
        heapq.heapify(pq)
        
        # Visited states with coordinates to avoid cycles
        # Key: (state, rounded_coord_tuple), Value: minimum_cost_reached
        visited = {}
        
        def coord_key(coord: tuple[float, ...]) -> tuple[int, ...]:
            """Create a discrete key from coordinates for cycle detection"""
            return tuple(round(c / coordinate_tolerance) for c in coord)
        
        while pq:
            current = heapq.heappop(pq)
            
            # Check if we reached an accepting state
            if current.state in self.accepting_states:
                return {
                    'path': current.get_path(),
                    'total_cost': current.total_cost,
                    'final_state': current.state,
                    'final_coord': current.coord
                }
            
            # Create key for visited tracking
            state_coord_key = (current.state, coord_key(current.coord))
            
            # Skip if we've visited this state-coordinate with lower cost
            if state_coord_key in visited and visited[state_coord_key] <= current.total_cost:
                continue
            
            visited[state_coord_key] = current.total_cost
            
            # Explore all outgoing edges from current state
            for next_state in range(self.n_states):
                if next_state == current.state:
                    continue    # Skip self-loops

                edges: list[DFA_Edge] = self.policy[current.state][next_state]
                if len(edges) == 0:
                    continue    # Skip edges with no viable routes
                
                for ei, edge in enumerate(edges):
                    # Get cost and next coordinate from edge's estimate function
                    sequence, trace = edge.policy_composition(qmodel, self.atomic_tasks, current.coord)
                    edge_cost = sum(len(seg)-1 for seg in trace)
                    
                    # Skip if cost is invalid
                    if edge_cost < 0 or np.isinf(edge_cost) or np.isnan(edge_cost):
                        continue
                    
                    new_total_cost = current.total_cost + edge_cost

                    next_coord = trace[-1][-1].loc
                    new_path = current.path + [(next_state, edge)]
                    # Create new state node
                    next_node = StateNode(
                        state=next_state,
                        coord=next_coord,
                        total_cost=new_total_cost,
                        path=new_path
                    )
                    
                    # Check if this state-coordinate combination is worth exploring
                    next_key = (next_state, coord_key(next_coord))
                    if next_key not in visited or visited[next_key] > new_total_cost:
                        heapq.heappush(pq, next_node)
        
        # No path found
        return None
        
if __name__ == "__main__":
    room = Room(10, 10)
    task = DFA_Task("G(p1 U p2)", room, None, {"p1": "p1", "p2": "p2"})
    print("Shortest paths from each state to accepting states:")
    print(task.shortest_paths)
    print("\nPaths from state 0 to accepting states:")
    print(task.policy_composition(0))
    print("\nUnreachable states:")
    print(task.get_unreachable_states())