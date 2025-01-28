import numpy as np
class Class:
  
  def __init__(self, index):
    self.index = index
    self.set_of_points = []

  def append_classification(self,point):
    self.set_of_points.append(point)

  def get_classified_points(self):
    return np.array(self.set_of_points)

  def __repr__(self):
    return f"Class(index={self.index}, set_of_points={self.get_classified_points()})"
