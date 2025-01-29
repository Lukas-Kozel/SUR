from class_object import Class
import numpy as np
import matplotlib.pyplot as plt

def compute_euclidean_distance(v1, v2):
    return np.power(np.linalg.norm(v1 - v2),2)

def compute_distance_to_each_point(reference, center_points)->np.array:
    distance_vector = []
    for index, x in enumerate(center_points):   
            distance_vector.append(compute_euclidean_distance(x,reference))
    return np.array(distance_vector)


def find_closest_center(distance_vector):
    min_index = np.argmin(distance_vector)
    return min_index


def classification(point_to_classify, center_points, class1, class2, class3):
    distance_vector = compute_distance_to_each_point(point_to_classify,center_points)
    index_of_closest_center_point = find_closest_center(distance_vector)

    if index_of_closest_center_point == 0:
        class1.append_classification(point_to_classify)
    if index_of_closest_center_point == 1:
        class2.append_classification(point_to_classify)
    if index_of_closest_center_point == 2:
        class3.append_classification(point_to_classify)


#return true if the algorithm should contiune otherwise return false
def compute_new_center_points(class1,class2,class3)->bool:
    class1_prev_center_point = class1.get_center_point()
    class2_prev_center_point = class2.get_center_point()
    class3_prev_center_point = class3.get_center_point()
    class1.compute_center_point()
    class2.compute_center_point()
    class3.compute_center_point()

    return not (np.array_equal(class1_prev_center_point, class1.get_center_point()) and 
                np.array_equal(class2_prev_center_point, class2.get_center_point()) and 
                np.array_equal(class3_prev_center_point, class3.get_center_point()))

def compute_whole_criterial_function(class1, class2, class3)->float:
    class1.compute_criterial_function()
    class2.compute_criterial_function()
    class3.compute_criterial_function()
    return class1.get_criterial_function()+class2.get_criterial_function()+class3.get_criterial_function()

def plot_classes(class1, class2, class3, iteration):
    plt.figure()
    plt.scatter(*zip(*class1.get_classified_points()), color='red', label='Class 1')
    plt.scatter(*zip(*class2.get_classified_points()), color='blue', label='Class 2')
    plt.scatter(*zip(*class3.get_classified_points()), color='green', label='Class 3')
    plt.scatter(*class1.get_center_point(), color='red', marker='x', s=100, label='Center 1')
    plt.scatter(*class2.get_center_point(), color='blue', marker='x', s=100, label='Center 2')
    plt.scatter(*class3.get_center_point(), color='green', marker='x', s=100, label='Center 3')
    plt.title(f'Iteration {iteration}')
    plt.legend()
    plt.show()

def plot_criterial_function(history_of_criterial_function):
    plt.figure()
    plt.plot(range(1, len(history_of_criterial_function) + 1), history_of_criterial_function, marker='o', linestyle='-')
    plt.xlabel("Iteration")
    plt.ylabel("Criterial Function Value")
    plt.title("Convergence of Criterial Function")
    plt.grid()
    plt.show()

def main():
    training_set = np.array([[4, -7], [-1, -3], [-3,4], [-4,6],[2,-3], [4,-6], [-3,1], [-1,2], [6,-9], [-1,-1]])
    center_points = np.array([training_set[0], training_set[5], training_set[8]])
    class1 = Class(1)
    class1.set_center_point(center_points[0])
    class2 = Class(2)
    class2.set_center_point(center_points[1])
    class3 = Class(3)
    class3.set_center_point(center_points[2])
    history_of_criterial_function = []
    iteration = 0
    while True:
        iteration += 1
        center_points = np.array([class1.get_center_point(),class2.get_center_point(),class3.get_center_point()])
        
        class1.set_of_points = []
        class2.set_of_points = []
        class3.set_of_points = []
        
        for vect in training_set:
            classification(vect,center_points, class1, class2, class3)
        
        print(f'Iteration {iteration}:')
        print(f'Class 1: {class1.get_classified_points()}')
        print(f'Class 2: {class2.get_classified_points()}')
        print(f'Class 3: {class3.get_classified_points()}')
        
        plot_classes(class1, class2, class3, iteration)
        current_criterial_value = compute_whole_criterial_function(class1, class2, class3)
        history_of_criterial_function.append(current_criterial_value)
        if not compute_new_center_points(class1, class2, class3):
            break

    plot_classes(class1, class2, class3, "final division")
    plot_criterial_function(history_of_criterial_function)

if __name__ == "__main__":
    main()