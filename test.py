class Student:

	def __init__(self, name, major, gpa, is_on_probation):
		self.name = name
		self.major = major
		self.gpa = gpa
		self.is_on_probation = is_on_probation

	def is_good_student(self):
		if self.gpa >= 3.2 and self.is_on_probation == False:
			return True
		else:
			return False

	def train(self):
		print("this is from Student")

class ClassRoom(Student):

	def test(self):
		print("this is from ClassRoom")