import matplotlib.pyplot as plt

class Graph():
	def __init__(self, title="title", x_label="x", y_label="y"):
		self.figure = plt.gcf()
		self.figure.show()
		self.figure.canvas.draw()
		self.min = 0
		self.max = 100
		plt.title(title)
		plt.xlabel(x_label)
		plt.ylabel(y_label)

	def update(self, a, b, color="b", label=""):
		plt.plot(a, b, color, label=label)
		plt.xlim(0, len(a))
		self.min = min(self.min, min(b))
		self.max = max(self.max, max(b))
		plt.ylim(self.min, self.max)
		self.figure.canvas.draw()

	def addLegend(self):
		plt.legend()