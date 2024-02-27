# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 11:54:50 2023

@author: urmii
"""

def find_nearest_meeting_cell(N, edges, src, dest):
    # Create a list to store the distance from source and destination cells to each cell
    dist_src = [-1] * N
    dist_dest = [-1] * N

    # Initialize the distance of source and destination cells to themselves as 0
    dist_src[src] = 0
    dist_dest[dest] = 0

    # Create a queue to perform breadth-first search (BFS) from both source and destination cells
    queue_src = [src]
    queue_dest = [dest]

    # Perform BFS from source cell
    while queue_src:
        current = queue_src.pop(0)
        for neighbor in edges[current]:
            if dist_src[neighbor] == -1:
                dist_src[neighbor] = dist_src[current] + 1
                queue_src.append(neighbor)

    # Perform BFS from destination cell
    while queue_dest:
        current = queue_dest.pop(0)
        for neighbor in edges[current]:
            if dist_dest[neighbor] == -1:
                dist_dest[neighbor] = dist_dest[current] + 1
                queue_dest.append(neighbor)

    # Find the nearest meeting cell by comparing distances
    nearest_meeting_cell = -1
    min_distance = float('inf')
    for i in range(N):
        if dist_src[i] != -1 and dist_dest[i] != -1:
            total_distance = dist_src[i] + dist_dest[i]
            if total_distance < min_distance:
                min_distance = total_distance
                nearest_meeting_cell = i

    return nearest_meeting_cell

# Sample input
N = 23
edge_str = "4 4 1 4 13 8 8 8 0 8 14 9 15 11 -1 10 15 22 22 22 22 22 21"
edges = [[] for _ in range(N)]
edge_list = edge_str.split()
for i in range(N):
    if edge_list[i] != '-1':
        edges[i] = list(map(int, edge_list[i]))
src = 4
dest = 8

nearest_meeting_cell = find_nearest_meeting_cell(N, edges, src, dest)
print(nearest_meeting_cell)
