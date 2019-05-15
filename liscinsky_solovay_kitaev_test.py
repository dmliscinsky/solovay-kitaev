#!/usr/bin/python3

# 
# CMSC 457 Introduction to Quantum Computing
# Author: Daniel Liscinsky
# 

import pytest
import numpy as np
import math

from liscinsky_solovay_kitaev import *



sqrt_2 = math.sqrt(2)

class X:
	mat = np.matrix([[0, 1], [1, 0]])
	axis = np.array([1, 0, 0])
	angle = math.pi

class Y:
	mat = np.matrix([[0, -1j], [1j, 0]])
	axis = np.array([0, 1, 0])
	angle = math.pi

class Z:
	mat = np.matrix([[1, 0], [0, -1]])
	axis = np.array([0, 0, 1])
	angle = math.pi

class H:
	mat = (1/sqrt_2) * np.matrix([[1, 1], [1, -1]])
	axis = np.array([(1/sqrt_2), 0, (1/sqrt_2)])
	angle = math.pi

class T:
	"""
	PI/8 gate
	"""
	mat = np.matrix([[1, 0], [0, math.cos(math.pi/4) + 1j * math.sin(math.pi/4)]])
	axis = np.array([0, 0, 1])
	angle = math.pi/4


I2 = numpy.identity(2)


#TODO function does not work currently
def make_SU2(my_unitary):

	# Make sure matrix determinant is 1
	det_U = numpy.linalg.det(my_unitary.mat)
	if det_U != 1:
		print("det_U = " + str(det_U))
		# Scale U so determinant is 1
		my_unitary.mat = (1 / cmath.sqrt(det_U)) * my_unitary.mat # Note this is the n-th root of det(U)

		# Correct the axis
		#my_unitary.axis *= -1 #TODO why?
		#TODO so far I have only seen cases where correcting the determinant causes the axis to flip
		
	#print("numpy.linalg.det(my_unitary.mat) = " + str(numpy.linalg.det(my_unitary.mat)))
	assert cmath.isclose(numpy.linalg.det(my_unitary.mat), 1)

make_SU2(X)
make_SU2(Y)
make_SU2(Z)
make_SU2(H)
make_SU2(T)






def test_get_axis_angle():

	axis, angle = get_axis_angle(X.mat)
	#assert np.array_equal(axis, np.array([1j, 0, 0])) and math.isclose(angle, math.pi)
	assert np.array_equal(axis, X.axis) and math.isclose(angle, X.angle)

	axis, angle = get_axis_angle(Y.mat)
	#assert np.array_equal(axis, np.array([0, 1j, 0])) and math.isclose(angle, math.pi)
	assert np.array_equal(axis, Y.axis) and math.isclose(angle, Y.angle)

	axis, angle = get_axis_angle(Z.mat)
	#assert np.array_equal(axis, np.array([0, 0, 1j])) and math.isclose(angle, math.pi)
	assert np.array_equal(axis, Z.axis) and math.isclose(angle, Z.angle)

	axis, angle = get_axis_angle(H.mat)
	print("axis = " + str(axis))
	#assert np.array_equal(axis, np.array([(1/sqrt_2) *1j, 0, (1/sqrt_2) * 1j])) and math.isclose(angle, math.pi)
	assert np.allclose(axis, H.axis) and math.isclose(angle, H.angle)

	axis, angle = get_axis_angle(T.mat)
	print("axis = " + str(axis))
	assert np.allclose(axis, T.axis) and math.isclose(angle, T.angle)


def test_from_axis_angle():
	abs_tol_zero = 1e-19

	#m = from_axis_angle(np.array([1j, 0, 0]), math.pi)
	m = from_axis_angle(X.axis, X.angle)
	assert np.allclose(m, X.mat)

	#m = from_axis_angle(np.array([0, 1j, 0]), math.pi)
	m = from_axis_angle(Y.axis, Y.angle)
	assert np.allclose(m, Y.mat)

	#m = from_axis_angle(np.array([0, 0, 1j]), math.pi)
	m = from_axis_angle(Z.axis, Z.angle)
	assert np.allclose(m, Z.mat)

	#m = from_axis_angle(np.array([(1/sqrt_2) *1j, 0, (1/sqrt_2) * 1j]), math.pi)
	m = from_axis_angle(H.axis, H.angle)
	print("m = " + str(m))
	assert np.allclose(m, H.mat)


def test_GC_Decompose():
	
	V, W = GC_Decompose(X.mat)
	X_reconstructed = np.matmul(np.matmul(np.matmul(V, W), V.getH()), W.getH())
	assert np.allclose(X.mat, X_reconstructed)


def test():
	pass