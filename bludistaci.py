import math
from queue import PriorityQueue
import numpy as np
import pygame
import threading
import sys

# List for robot colors
robot_colors = [
    ("Blue", (0, 0, 255)),
    ("Yellow", (255, 255, 0)),
    ("Green", (0, 255, 0)),
    ("Purple", (128, 0, 128)),
    ("Cyan", (0, 255, 255)),
    ("Magenta", (255, 0, 255)),
    ("Dark Green", (0, 128, 0)),
    ("Olive", (128, 128, 0)),
    ("Teal", (0, 128, 128)),
    ("Maroon", (128, 0, 0)),
    ("Navy", (0, 0, 128)),
    ("Gray", (128, 128, 128))
]

# Function to load maze from source
def load_maze(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    maze = []
    reading_maze = True
    robot_starts = []
    bludistaci = []
    pickup_point = None
    
    for line in lines:
        line = line.strip()
        if line.startswith('robot_starts'):
            reading_maze = False
            _, starts = line.split('=', 1)
            starts = starts.strip().strip('[]')
            robot_starts = [tuple(map(int, coord.strip().strip('()').split(','))) for coord in starts.split('), (')]
        elif line.startswith('bludistaci'):
            _, bludis = line.split('=', 1)
            bludis = bludis.strip().strip('[]')
            bludistaci = [tuple(map(int, coord.strip().strip('()').split(','))) for coord in bludis.split('), (')]
        elif line.startswith('pickup_point'):
            _, pickup = line.split('=', 1)
            pickup = pickup.strip().strip('()')
            pickup_point = tuple(map(int, pickup.split(',')))
        elif reading_maze:
            if 'X' in line or ' ' in line:
                maze_row = [1 if char == 'X' else 0 for char in line]
                maze.append(maze_row)
    
    maze_array = np.array(maze)
    return maze_array, robot_starts, bludistaci, pickup_point

# Function to print the initial state of maze
def print_maze(screen, maze, robot_starts, robot_color, bludistaci, pickup_point):
    screen.fill((255, 255, 255))
    block_size = 20
    for y in range(maze.shape[0]):
        for x in range(maze.shape[1]):
            rect = pygame.Rect(x * block_size, y * block_size, block_size, block_size)
            if maze[y, x] == 1:
                pygame.draw.rect(screen, (0, 0, 0), rect)
            else:
                pygame.draw.rect(screen, (255, 255, 255), rect)
                pygame.draw.rect(screen, (200, 200, 200), rect, 1)
    
    for pos in robot_starts:
        name, color = robot_color[pos]
        pygame.draw.rect(screen, color, (pos[0] * block_size, pos[1] * block_size, block_size, block_size))

    for pos in bludistaci:
        pygame.draw.rect(screen, (255, 0, 0), (pos[0] * block_size, pos[1] * block_size, block_size, block_size))
    
    pygame.draw.rect(screen, (255, 165, 0), (pickup_point[0] * block_size, pickup_point[1] * block_size, block_size, block_size))
    pygame.display.flip()

# Function to make color lighter for open nodes
def light_color(color, amount=0.7):
    r, g, b = color
    r = int(r + (255 - r) * amount)
    g = int(g + (255 - g) * amount)
    b = int(b + (255 - b) * amount)
    return (r, g, b)

screen_lock = threading.Lock()

# Function to display maze during a run 
def show_maze(screen, start, goal, color, goal_color=None, path=None, open_nodes=None):
    block_size = 20
    with screen_lock:
        pygame.draw.rect(screen, color, (start[0] * block_size, start[1] * block_size, block_size, block_size))
        if goal_color != None:
            pygame.draw.rect(screen, goal_color, (goal[0] * block_size, goal[1] * block_size, block_size, block_size))
        else: 
            pygame.draw.rect(screen, (255, 0, 0), (goal[0] * block_size, goal[1] * block_size, block_size, block_size))

        if open_nodes:
            for node in open_nodes:
                if node == goal:
                    continue
                open_color = light_color(color)
                pygame.draw.rect(screen, open_color, (node[0] * block_size, node[1] * block_size, block_size, block_size))

        if path:
            for node in path:
                pygame.draw.rect(screen, color, (node[0] * block_size, node[1] * block_size, block_size, block_size))

        pygame.display.flip()

# Heuristic for astar algorithm
# Type 1 = Euclidean distance (default)
# Type 2 = Manhattan distance
# Type 3 = Chebyshev distance
def heuristic(a, b, type=None):
    if type == 1:
        return math.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)
    elif type == 2:
        return abs(b[0] - a[0]) + abs(b[1] - a[1])
    elif type == 3:
        return max(abs(b[0] - a[0]), abs(b[1] - a[1]))
    else:
        return math.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)

# Function that performs astar algorithm
def astar(maze, start, goal, color, goal_color=None, pickup_point=None, visualize=False, screen=None, type=None):
    movements = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    pq = PriorityQueue()
    pq.put((0, start))
    came_from = {}
    g_score = {(i, j): float('inf') for i in range(len(maze[0])) for j in range(len(maze))}
    g_score[start] = 0
    f_score = {(i, j): float('inf') for i in range(len(maze[0])) for j in range(len(maze))}
    f_score[start] = heuristic(start, goal, type) 
    expanded_nodes = 0
    open_nodes = []
    while not pq.empty():
        current = pq.get()[1]
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            if pickup_point:
                pickup_path, pickup_cost, pickup_expanded_nodes = astar(maze, goal, pickup_point, color, goal_color=(255, 165, 0), pickup_point=None, visualize=True, screen=screen, type=type)  
                total_path = path + pickup_path[1:]
                total_cost = g_score[goal] + pickup_cost
                total_expanded_nodes = expanded_nodes + pickup_expanded_nodes
                if visualize:
                    show_maze(screen, start, goal, color, goal_color=goal_color, path=total_path, open_nodes=open_nodes)    
                return total_path, total_cost, total_expanded_nodes
            return path, g_score[goal], expanded_nodes
        expanded_nodes += 1
        for move in movements:
            neighbor = (current[0] + move[0], current[1] + move[1])
            if 0 < neighbor[0] < len(maze[0]) and 0 < neighbor[1] < len(maze) and maze[neighbor[1]][neighbor[0]] != 1:
                new_g_score = g_score[current] + 1
                if new_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = new_g_score
                    f_score[neighbor] = new_g_score + heuristic(neighbor, goal, type)
                    pq.put((f_score[neighbor], neighbor))
                    open_nodes.append(neighbor)
        if visualize:
            show_maze(screen, start, goal, color, goal_color=goal_color, open_nodes=open_nodes)
    return None, float('inf'), expanded_nodes

# Function for threads to run astar algorithm
def run_astar(maze, start, goal, robot_color, used_robots, pickup_point=None, visualize=False, screen=None, type=None):
    name, color = robot_color
    path, cost, expanded_nodes = astar(maze, start, goal, color, goal_color=None, pickup_point=pickup_point, visualize=visualize, screen=screen, type=type)
    print('\n')
    print(f'{name} Robot')
    print("Path:", path)
    print("Expanded Nodes:", expanded_nodes)
    print("Cost:", cost)
    used_robots.remove(start)
    
# Main part of program
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Chybí test file")
        sys.exit(1)
    
    file = sys.argv[1]
    maze, robot_starts, bludistaci, pickup_point = load_maze(file)

    type = None
    if len(sys.argv) == 3:
        type = int(sys.argv[2])

    heuristics = []
    for robot_start in robot_starts:
        for bludistak in bludistaci:
            h = heuristic(robot_start, bludistak, type)
            heuristics.append((h, robot_start, bludistak))
    heuristics.sort()

    pygame.init()
    screen = pygame.display.set_mode((maze.shape[1] * 20, maze.shape[0] * 20))
    pygame.display.set_caption("Bludišťáci")
    
    robot_color = {robot_starts[i]: robot_colors[i % len(robot_colors)] for i in range(len(robot_starts))}
    print_maze(screen, maze, robot_starts, robot_color, bludistaci, pickup_point)

    used_robots = set()
    used_bludistaci = set()
    threads = []
    while len(bludistaci) != len(used_bludistaci):
        for h, robot_start, bludistak in heuristics:
            if robot_start not in used_robots and bludistak not in used_bludistaci:
                thread = threading.Thread(target=run_astar, args=(maze, robot_start, bludistak, robot_color[robot_start], used_robots, pickup_point, True, screen, type))
                threads.append(thread)
                thread.start()
                used_robots.add(robot_start)
                used_bludistaci.add(bludistak)
                heuristics.remove((h, robot_start, bludistak))

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
    pygame.quit()
    
    for thread in threads:
        thread.join()