import numpy as np

class Class:
  def __init__(self, index):
      self.index = index
      self.set_of_points = []
      self.center_point = [0, 0]
      self.criterial_function_value=0

  def append_classification(self, point):
      self.set_of_points.append(point)

  def compute_center_point(self):
      points_array = np.array(self.set_of_points)
      if len(points_array) == 0:
          return None 
      center_point = np.mean(points_array, axis=0)
      self.center_point = center_point
      return center_point

  def compute_criterial_function(self):
      points_array = np.array(self.set_of_points)
      sum=0
      if len(points_array) == 0:
          return None      
      for point in points_array:          
         sum+= np.power(np.linalg.norm(self.center_point - point),2)
      self.criterial_function_value = sum
      return self.criterial_function_value

  def get_center_point(self):
      return self.center_point

  def get_criterial_function(self):
      return self.criterial_function_value
  
  def set_center_point(self, center_point):
      self.center_point = center_point

  def get_classified_points(self):
      return np.array(self.set_of_points)

  def __repr__(self):
    return f"Class(index={self.index}, set_of_points={self.get_classified_points()})"
