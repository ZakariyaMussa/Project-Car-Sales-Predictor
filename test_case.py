# 3 test cases changing the neurons in the hidden layer
from ProjectCarSalesPredictor import N
import pytest

#print (pytest._file_)

def test_mini_layer():
    # ANN(5) 0.000481101348559941 (40)
    # ANN(15) #0.0003910826101198538 (30)
    bestcase = 0.0015319119098182535 #ANN(25)
    # print (ANN.MSE())
    assert N(15) <= bestcase

def test_medium_layer():
    # ANN(25)
    bestcase=0.0015319119098182535
    assert N(35) <= bestcase
    #print()

def test_large_layer():
    #ANN (50)
    bestcase = 0.0015319119098182535
    assert N(50) <= bestcase
    #print()