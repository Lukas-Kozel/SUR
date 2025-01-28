from class_object import Class
import numpy as np
import matplotlib.pyplot as plt

def compute_euclidean_distance(v1, v2):
    return np.linalg.norm(v1 - v2)

def compute_distance_to_each_point(reference, training_set, connected_output)->np.array:
    distance_vector = []
    connected_indices = [idx for idx, _ in connected_output]
    for index, x in enumerate(training_set):
        if np.array_equal(x,reference) or index in connected_indices:
            distance_vector.append(np.inf)
        else:    
            distance_vector.append(compute_euclidean_distance(x,reference))
    return np.array(distance_vector)


def find_closest_point(distance_vector) -> tuple:
    min_index = np.argmin(distance_vector)
    min_value = distance_vector[min_index]
    return (min_value, min_index)

def load_data(filepath) -> np.array:
    training_dataset = []
    with open(filepath) as f:
        for line in f:
            training_dataset.append(list(map(float, line.strip().split())))
    return np.array(training_dataset)

def one_iteration(training_set,current_vector, connected_output):
    distance_vector = compute_distance_to_each_point(current_vector,training_set,connected_output)
    distance, index = find_closest_point(distance_vector)
    connected_output.append((index, distance))
    next_index = index
    return training_set[next_index]

def classificate(training_set,connected_output, threshold, class_array):
    class_index = 0
    for index,distance in connected_output:
        if distance < threshold:
            class_array[class_index].append_classification(training_set[index])
        else:
            class_index+=1
            if class_index >= len(class_array):
                class_array.append(Class(class_index))        
            class_array[class_index].append_classification(training_set[index])

def plot_result(class_array):
    colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k']
    for class_object in class_array:
        points = class_object.get_classified_points()
        if len(points) > 0:
            color = colors[class_object.index % len(colors)]
            plt.scatter(points[:, 0], points[:, 1], c=color, label=f'Class {class_object.index}')
    
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Classified Points')
    plt.legend()
    plt.grid()
    plt.show()

def plot_chain(training_set, connected_output):
    plt.figure()
    for i, (index, _) in enumerate(connected_output):
        point = training_set[index]
        if i == 0:

            plt.scatter(point[0], point[1], c='red', s=100, label='Starting Point')
        else:
            plt.scatter(point[0], point[1], c='black')
            prev_point = training_set[connected_output[i - 1][0]]
            plt.plot([prev_point[0], point[0]], [prev_point[1], point[1]], 'k-')

            mid_x = (prev_point[0] + point[0]) / 2
            mid_y = (prev_point[1] + point[1]) / 2
            plt.annotate(str(i), (mid_x, mid_y), textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.xlim(-9.5,9.5)
    plt.ylim(-9.5,9.5)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Connected Output Chain')
    plt.grid()
    plt.legend()
    plt.show()

def main():
    q=3
    training_set = np.array([[4, -7], [-1, -3], [-3,4], [-4,6],[2,-3], [4,-6], [-3,1], [-1,2], [6,-9], [-1,-1]])
    start_index = 4
    current_vector = training_set[start_index]
    connected_output = [] # index in training set : distance from prev point

    for _, _ in enumerate(training_set):
        if not connected_output:
            connected_output.append((start_index, 0))
        else:
            current_vector = one_iteration(training_set, current_vector, connected_output)
    
    class_array = [Class(0)]
    classificate(training_set,connected_output=connected_output,threshold=q, class_array=class_array)
    
    for class_object in class_array:
        print(f"Class {class_object.index}: Points {class_object.get_classified_points()}")

    plot_result(class_array)
    plot_chain(training_set, connected_output)

if __name__ == "__main__":
    main()