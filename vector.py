from math import sqrt, sin, cos, atan2, pi


class Vector:
    def __init__(self, x=None, y=None):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y)

    def __repr__(self):
        return f"{self.x}, {self.y}"

    def __len__(self):
        return sqrt(self.x**2 + self.y**2)

    def __mul__(self, other):
        return self.x * other + self.y * other

    def __matmul__(self, other):
        return self.x * other.y - self.y * other.x

    def rotate(self, angle):
        rad = angle * pi / 180
        return Vector(
            self.x * cos(rad) - self.y * sin(rad), self.x * sin(rad) + self.y * cos(rad)
        )

    def normalize(self):
        if len(self) == 0:
            return Vector(0, 0)
        return Vector(self.x / len(self), self.y / len(self))
