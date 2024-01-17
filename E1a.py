def find_nearest_node():
    min_distance = float('inf')
    min_key = ''
    for visited_city in tour:
        for remaining_city in remaining_cities:
            if distance_matrix[visited_city][remaining_city] < min_distance:
                min_key = remaining_city
                min_distance = distance_matrix[visited_city][remaining_city]
    return (min_key, min_distance)

def find_lowest_cost_insertion():
    min_cost = float('inf')
    min_index = 0
    for i in range(len(tour)-1):
        cost = distance_matrix[tour[i]][min_key] + distance_matrix[tour[i + 1]][min_key] - distance_matrix[tour[i]][tour[i + 1]]
        if cost < min_cost:
            min_cost = cost
            min_index = i + 1
    return (min_index, min_cost)

distance_matrix={
'Aarhus': {'Aarhus': 0,'Copenhagen': 300,'Odense': 150,'Aalborg': 200,'Esbjerg': 250},
'Copenhagen': {'Aarhus': 300,'Copenhagen': 0,'Odense': 100,'Aalborg': 400,'Esbjerg': 350},
'Odense': {'Aarhus': 150,'Copenhagen': 100,'Odense': 0,'Aalborg': 300,'Esbjerg': 200},
'Aalborg': {'Aarhus': 200,'Copenhagen': 400,'Odense': 300,'Aalborg': 0,'Esbjerg': 250},
'Esbjerg': {'Aarhus': 250,'Copenhagen': 350,'Odense': 200,'Aalborg': 250,'Esbjerg': 0},
}

tour = ['Copenhagen', 'Copenhagen']
remaining_cities = set(distance_matrix.keys())
remaining_cities.remove('Copenhagen')

i = 0
while remaining_cities:
    i += 1
    min_key, min_distance = find_nearest_node()
    min_index, min_cost = find_lowest_cost_insertion()
    tour.insert(min_index, min_key)
    remaining_cities.remove(min_key)
    print('Iteration:', i)
    print(min_key, min_distance)
    print(tour)
    print(min_index, min_cost)

print('Final tour is:', tour)

