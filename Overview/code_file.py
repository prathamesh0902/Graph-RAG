class Calculator:
    """A simple calculator class with three functions."""

    def add(self, x, y):
        """Return the sum of x and y."""
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            raise TypeError("Both arguments must be numbers.")
        return x + y

    def subtract(self, x, y):
        """Return the difference of x and y."""
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            raise TypeError("Both arguments must be numbers.")
        return x - y

    def multiply(self, x, y):
        """Return the product of x and y."""
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            raise TypeError("Both arguments must be numbers.")
        return x * y