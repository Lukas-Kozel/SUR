from class_object import Class
import numpy as np
import matplotlib.pyplot as plt

def compute_euclidean_distance(v1, v2):
    return np.power(np.linalg.norm(v1 - v2),2)

def compute_distance_to_each_point(reference, center_points)->np.array:
    distance_vector = []
    for _, x in enumerate(center_points):   
            distance_vector.append(compute_euclidean_distance(x,reference))
    return np.array(distance_vector)


def find_closest_center(distance_vector):
    min_index = np.argmin(distance_vector)
    return min_index


def classification(point_to_classify, center_points, class_list):
    distance_vector = compute_distance_to_each_point(point_to_classify,center_points)
    index_of_closest_center_point = find_closest_center(distance_vector)

    class_list[index_of_closest_center_point].append_classification(point_to_classify)


def compute_new_center_points(class_list) -> bool:
    """
    Return True if at least one center changed
    (meaning we should continue),
    Return False if no center changed
    (meaning we have converged).
    """
    any_changed = False
    old_centers = []

    for c in class_list:
        old_centers.append(c.get_center_point())

    for i, c in enumerate(class_list):
        c.compute_center_point()
        if not np.array_equal(old_centers[i], c.get_center_point()):
            any_changed = True

    return any_changed


def compute_whole_criterial_function(class_list)->float:
   cumulative_sum=0
   for _,c in enumerate(class_list):
       c.compute_criterial_function()
       cumulative_sum+=c.get_criterial_function()
   return cumulative_sum
    
def plot_classes(class_list, iteration):
    
    plt.figure()
    
    colors = plt.cm.get_cmap("tab10", len(class_list))
    
    for i, c in enumerate(class_list):
        classified_points = c.get_classified_points()
        center_point = c.get_center_point()
        
        if len(classified_points) > 0:
            xs, ys = zip(*classified_points)
            plt.scatter(xs, ys, color=colors(i), label=f'Class {i}')
        
        plt.scatter(center_point[0], center_point[1],color=colors(i), marker='x', s=100, label=f'Center {i}')
    
    plt.title(f"Iteration {iteration}")
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

    # setup
    training_set = np.array([[4, -7], [-1, -3], [-3,4], [-4,6],[2,-3], [4,-6], [-3,1], [-1,2], [6,-9], [-1,-1]])
    number_of_classes = 4
    center_points = np.array([training_set[1], training_set[3], training_set[9], training_set[2]])

    assert number_of_classes == len(center_points), f"Error: number_of_classes ({number_of_classes}) does not match the number of center points ({len(center_points)})"

    class_list =[]

    for class_index in range(number_of_classes):
        class_list.append(Class(class_index))
        class_list[-1].set_center_point(center_points[class_index])

    history_of_criterial_function = []
    iteration = 0
    while True:
        iteration += 1
        center_points=[]
        for _,c in enumerate(class_list):
            center_points.append(c.get_center_point())
            c.set_of_points = []
            
        for vect in training_set:
            classification(vect,center_points, class_list)

        print(f'Iteration {iteration}:')
        for i,c in enumerate(class_list):
            print(f'Class {i}: {c.get_classified_points()}')
                
        plot_classes(class_list, iteration)
        
        current_criterial_value = compute_whole_criterial_function(class_list)
        history_of_criterial_function.append(current_criterial_value)

        if not compute_new_center_points(class_list):
            break

    plot_classes(class_list, "final division")
    plot_criterial_function(history_of_criterial_function)

if __name__ == "__main__":
    main()