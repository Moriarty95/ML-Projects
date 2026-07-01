import unittest
import numpy as np
from numpy.testing import assert_array_equal

# Import the Solution class from your main file
from solution import Solution

class TestSolution(unittest.TestCase):
    def setUp(self):
        """This method runs before every test, setting up our class instance."""
        self.solution = Solution()

    def test_sigmoid(self):
        # 1. Setup test inputs
        z = np.array([0.0, 2.0, -2.0, 100.0, -100.0], dtype=np.float64)
        
        # 2. Define expected outputs (manually calculated and rounded to 5 decimals)
        # sigmoid(0) = 0.5
        # sigmoid(2) ≈ 0.88080
        # sigmoid(-2) ≈ 0.11920
        # sigmoid(100) ≈ 1.0 (approaches 1)
        # sigmoid(-100) ≈ 0.0 (approaches 0)
        expected = np.array([0.50000, 0.88080, 0.11920, 1.00000, 0.00000], dtype=np.float64)
        
        # 3. Call the function
        result = self.solution.sigmoid(z)
        
        # 4. Assert the result matches expected output exactly
        assert_array_equal(result, expected)

    def test_relu(self):
        # 1. Setup test inputs (mixing positive, negative, and zero)
        z = np.array([0.0, 5.5, -3.2, 0.1, -0.1], dtype=np.float64)
        
        # 2. Define expected outputs (negatives become 0, positives stay the same)
        expected = np.array([0.0, 5.5, 0.0, 0.1, 0.0], dtype=np.float64)
        
        # 3. Call the function
        result = self.solution.relu(z)
        
        # 4. Assert the result matches expected output
        assert_array_equal(result, expected)

if __name__ == '__main__':
    unittest.main()
