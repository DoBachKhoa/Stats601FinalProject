import pytest
import numpy as np

# Import implemented functions
import sys
sys.path.insert(1, 'src/algorithm')
from lda import NewtonRaphson, VariationalInference
from utils import discreteNormal, discretePoisson

'''
Test function correctness of Algorithm 1: Newton Raphson iteration algorithm
    and Algorithm 2: Variational Inference approximation algorithm
    using pytest.
'''

class TestNewtonRaphson:
    '''
    Class of functions that test correctness of
        Algorithm 1: Newton Raphson iteration algorithm
    '''
    def test_type(self):
        '''Test input type'''
        with pytest.raises(ValueError, match='type'):
            NewtonRaphson(3.14, np.array([1., 2.]), np.array([3., 4.]), 1., safecheck=True)

        with pytest.raises(ValueError, match='type'):
            NewtonRaphson(np.array([1., 2.]), 3.14, np.array([3., 4.]), 1., safecheck=True)

        with pytest.raises(ValueError, match='type'):
            NewtonRaphson(np.array([1., 2.]), np.array([3., 4.]), 3.14, 1., safecheck=True)

    def test_matching(self):
        '''Dimensionality matching test'''
        with pytest.raises(ValueError, match='simmilar length'):
            NewtonRaphson(np.array([1., 2.]), np.array([3., 4.]), np.array([5., 6., 7.]),
                          1., safecheck=True)

        with pytest.raises(ValueError, match='simmilar length'):
            NewtonRaphson(np.array([1., 2.]), np.array([3., 4., 5.]), np.array([6., 7.]),
                          1., safecheck=True)

    def test_diagonal_hessian(self):
        '''
        When z = 0, the Hessian H becomes a diagonal matrix.
        Thus we know what to expect.
        '''
        alpha = np.array([1., 2., 3.])
        g     = np.array([4., 5., 6.])
        h     = np.array([7., 8., 9.])
        expected_output = alpha - g/h
        actual_output   = NewtonRaphson(alpha, g, h, z=0.)
        assert float(np.max(np.abs(actual_output-expected_output))) < 1e-9

class TestVIapproximation:
    '''
    Class of functions that test correctness of
        Algorithm 2: Variational Inference approximation algorithm
    '''
    def test_matching(self):
        '''Dimensionality matching test'''
        with pytest.raises(ValueError, match='length'):
            VariationalInference(np.array([1, 2]), np.array([3., 4., 5.]), np.array([[6., 7.], [8., 9.]]),
                          num_iter=10, safecheck=True)

