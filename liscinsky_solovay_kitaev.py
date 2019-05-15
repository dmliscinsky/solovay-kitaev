#!/usr/bin/python3

# 
# CMSC 457 Introduction to Quantum Computing
# Author: Daniel Liscinsky
# 

import numpy
import math
import cmath
import itertools
import functools
import pickle
import argparse
import os
import time



sqrt_2 = math.sqrt(2)


I2 = numpy.identity(2)
#X = numpy.matrix([[0, 1], [1, 0]])
#Y = numpy.matrix([[0, -1j], [1j, 0]])
#Z = numpy.matrix([[1, 0], [0, -1]])
H = (1/sqrt_2) * numpy.matrix([[1, 1], [1, -1]])
T = numpy.matrix([[1, 0], [0, math.cos(math.pi/4) + 1j * math.sin(math.pi/4)]])

X = numpy.matrix([[0, -1j], [-1j, 0]])
Y = numpy.matrix([[0, -1], [1, 0]])
Z = numpy.matrix([[1j, 0], [0, -1j]])


g_e0_net = None



def make_SU2(U: numpy.matrix):

	# Make sure matrix determinant is 1
	det_U = numpy.linalg.det(U)
	if not cmath.isclose(det_U, 1):
		#print("det_U = " + str(det_U))
		
		# Scale U so determinant is 1
		U = (1 / cmath.sqrt(det_U)) * U # Note this is the n-th root of det(U)

	#print("numpy.linalg.det(mU) = " + str(numpy.linalg.det(my_unitary.mat)))
	assert cmath.isclose(numpy.linalg.det(U), 1)
	return U



H = make_SU2(H)
T = make_SU2(T)



def dist(A, B):
	return numpy.linalg.norm(A - B, 2)

def conj_transpose(A):
	"""
	Because numpy.matrix has been deprecated, we won't have 
	the nice, convenient `getH()` method anymore, so instead 
	we have to make our own conjugate transpose.
	"""
	return A.transpose().conjugate()

def get_axis_angle(U: numpy.matrix):
	"""
	"""
	
	# Should be a 2x2 matrix
	assert U.A.shape == (2,2)

	# U must be in SU(2), or else the computations do not work
	U = make_SU2(U)
	

	half_trace = numpy.real(U.trace() / 2) # The imaginary part should always be zero
	half_theta = math.acos(half_trace)
	theta = half_theta * 2


	"""
	bloch_sphere_rot_mat = numpy.matrix([[0,0,0], [0,0,0], [0,0,0]])

	pauli_mats = [X, Y, Z]
	for j in range(0,3):
		for k in range(0, 3):
			bloch_sphere_rot_mat.A[j][k] = (pauli_mats[j] * U * pauli_mats[k] * U.getH()).trace() / 2

	print(bloch_sphere_rot_mat)
	"""
	"""
	axis = []

	pauli_mats = [X, Y, Z]
	for i in range(0, 3):
		axis.append(-0.5j * numpy.trace(pauli_mats[i] * U))
		#print("axis[" + str(i) + "] = " + str(axis[i]))

	axis = numpy.array(axis)

	print("axis = " + str(axis) + ", theta = " + str(theta))
	assert(axis[0].imag == 0 and axis[1].imag == 0 and axis[2].imag == 0)

	# Make sure number types are not complex
	axis = numpy.array([axis[0].real, axis[1].real, axis[2].real])

	print("numpy.linalg.norm(axis) = " + str(numpy.linalg.norm(axis)) )
	assert math.isclose(numpy.linalg.norm(axis), 1.0)
	return axis, theta
	"""


	"""
	#TODO no good
	sx1 = -1 * U.A[0][1].imag
	sx2 = U.A[1][0].real
	sx3 = ((U.A[1][1] - U.A[0][0])/2.0).imag
	print("sx1 = " + str(sx1) + ", sx2 = " + str(sx2) + ", sx3 = " + str(sx3))

	cos_theta = ((U.A[0][0] + U.A[1][1])/2.0).real

	sin_theta = math.sqrt(sx1*sx1 + sx2*sx2 + sx3*sx3)
	print("sin_theta = " + str(sin_theta))

	if sin_theta == 0:
		x = 2 * math.acos(cos_theta)
		y = 0
		z = 0
	else:
		theta = math.atan2(sin_theta, cos_theta)
		x = 2 * theta * sx1/sin_theta
		y = 2 * theta * sx2/sin_theta
		z = 2 * theta * sx3/sin_theta
	
	axis = numpy.array([x, y, z])
	
	print("axis = " + str(axis) + ", theta = " + str(theta))
	assert(axis[0].imag == 0 and axis[1].imag == 0 and axis[2].imag == 0)

	print("numpy.linalg.norm(axis) = " + str(numpy.linalg.norm(axis)) )
	assert math.isclose(numpy.linalg.norm(axis), 1.0)
	return axis, theta
	"""


	half_trace = numpy.real(U.trace() / 2) # The imaginary part should always be zero
	half_theta = math.acos(half_trace)
	theta = half_theta * 2

	# math.cos(half_theta) == U.trace() / 2
	sin_half_theta = math.sin(half_theta)
	nz = (U.A[0][0] - half_trace) / (-1j * sin_half_theta)
	nz = nz.item(0)

	nz_ = (U.A[1][1] - half_trace) / (1j * sin_half_theta)
	nz_ = nz_.item(0)
	print("nz = " + str(nz) + ", nz_ = " + str(nz_))
	assert cmath.isclose(nz, nz_)

	nx_times_sin_half_theta = (U.A[0][1] + U.A[1][0]) / (-2j)
	print(nx_times_sin_half_theta)
	nx = (nx_times_sin_half_theta) / (sin_half_theta)
	ny = (U.A[1][0] + nx_times_sin_half_theta) / sin_half_theta
	
	ny = (U.A[1][0]) / (-1j * sin_half_theta)
	ny = (ny - nx) / 1j

	ny_ = (((U.A[0][1]) / (-1j * sin_half_theta)) - nx) / -1j
	print("ny = " + str(ny) + ", ny_ = " + str(ny_))
	assert cmath.isclose(ny, ny_)

	print("U.A[1][0] = " + str(U.A[1][0]))
	print("U.A[0][1] = " + str(U.A[0][1]))
	print("nx = " + str(nx))
	print("ny = " + str(ny))
	#print("U.A[0][1] = " + str(U.A[0][1]) + ", (-1j * sin_half_theta * (nx - 1j * ny)) = " + str(-1j * sin_half_theta * (nx - 1j * ny)))
	#assert U.A[0][1] == (-1j * sin_half_theta * (nx - 1j * ny))

	print("U.A[0][1] = " + str(U.A[0][1]) + ", (sin_half_theta * (nx - 1j * ny)) = " + str(-1j * sin_half_theta * (nx - 1j * ny)))
	assert cmath.isclose( U.A[0][1], (-1j * sin_half_theta * (nx - 1j * ny)) )


	# Remove the 'i' factor, since the axis should be real
	#TODO this might not be the best/correct way to handle this
	abs_tol_zero = 1e-14
	
	if math.isclose(nx.imag, 0, abs_tol=abs_tol_zero):
		nx = nx.real
	elif math.isclose(nx.real, 0, abs_tol=abs_tol_zero):
		nx = nx.imag
	else:
		assert False
	
	if math.isclose(ny.imag, 0, abs_tol=abs_tol_zero):
		ny = ny.real
	elif math.isclose(ny.real, 0, abs_tol=abs_tol_zero):
		ny = ny.imag
	else:
		assert False

	if math.isclose(nz.imag, 0, abs_tol=abs_tol_zero):
		nz = nz.real
	elif math.isclose(nz.real, 0, abs_tol=abs_tol_zero):
		nz = nz.imag
	else:
		assert False

	
	axis = numpy.array([nx, ny, nz])

	print("axis = " + str(axis) + ", theta = " + str(theta))
	assert(axis[0].imag == 0 and axis[1].imag == 0 and axis[2].imag == 0)

	print("numpy.linalg.norm(axis) = " + str(numpy.linalg.norm(axis)) )
	assert math.isclose(numpy.linalg.norm(axis), 1.0)
	return axis, theta

def from_axis_angle(axis: numpy.ndarray, angle: float):
	"""
	"""
	angle /= 2

	nx = axis[0]
	ny = axis[1]
	nz = axis[2]

	#return math.cos(angle) * I2 - 1j * math.sin(angle) * (nx*X + ny*Y + nz*Z)
	m = math.cos(angle) * I2 - 1j * math.sin(angle) * (nx*X + ny*Y + nz*Z)

	# If value is close enough to zero, round it down to zero
	for i in range(0,4):
		if cmath.isclose(m.A1[i], 0, abs_tol=1e-14):
			m.A1[i] = 0

	return make_SU2(m)

def Bloch_Rotation_X(theta: float):
	half_theta = theta / 2
	cos_half_theta = numpy.cos(half_theta)
	sin_half_theta = numpy.sin(half_theta)
	return numpy.matrix([
		[cos_half_theta, -1j * sin_half_theta],
		[-1j * sin_half_theta, cos_half_theta]
	])

def Bloch_Rotation_Y(theta: float):
	return from_axis_angle(numpy.array([0, 1j, 0]), theta) #TODO check axis is right
	
def GC_Decompose(U):
	"""
	U        Gate
	"""

	epsilon_n_minus_1 = .01 #TODO ????????????????????????????????????? MAKE INTO A PARAMETER OR SOMETHING

	c_gc = 1 / math.sqrt(2)
	#c_gc_sqrt_epsilon = c_gc * math.sqrt(epsilon)
	c_gc_sqrt_epsilon = c_gc * math.sqrt(epsilon_n_minus_1)


	# 
	axis_U, theta_U = get_axis_angle(U)
	print("axis_U = " + str(axis_U) + ", theta_U = " + str(theta_U))
	
	# Solve for phi
	"""
	x = sin^4(phi/2) * (1 - sin^4(phi/2))
	
	-sin^8(phi/2) + sin^4(phi/2) - x = 0
	sin^8(phi/2) - sin^4(phi/2) + x = 0
	
	sin^8(phi/2) - sin^4(phi/2) + x = 0
	y^2 - y + x = 0
	sin^4(phi/2) = y = (1 (+-) sqrt(1 - 4*x)) / 2

	phi = 2 * arcsin( srqt(sqrt(y)) )
	phi = 2 * arcsin( srqt(sqrt( (1 (+-) |cos(theta/2)|)/2 )) )
	"""
	
	'''
	# Longer calculation
	x = math.sin(theta_U/2) / 2
	x = x * x 

	y_part = math.sqrt(1 - 4*x)
	y1 = (1 - y_part)/2
	y2 = (1 + y_part)/2
	
	print("y1 = " + str(y1))
	print("y2 = " + str(y2))
	print("y1 + y2 = " + str(y1 + y2))
	'''
	
	y_part = abs(math.cos(theta_U/2))
	y1 = (1 - y_part)/2
	y2 = (1 + y_part)/2

	print("y1 = " + str(y1))
	print("y2 = " + str(y2))
	print("y1 + y2 = " + str(y1 + y2))

	phi = 2 * math.asin( math.sqrt(math.sqrt(y1)) )
	print("phi = " + str(phi))
	print("phi error (should be close to 0) = " + str( 2*math.sin(phi/2)**2 * math.sqrt(1-math.sin(phi/2)**4) - math.sin(theta_U/2) ))

	# Calculate V and W
	V_tilde = Bloch_Rotation_X(phi)
	W_tilde = Bloch_Rotation_Y(phi)

	print("V_tilde = " + str(V_tilde))
	print("W_tilde = " + str(W_tilde))

	VWVtWt_tilde = numpy.matmul(numpy.matmul(numpy.matmul(V_tilde, W_tilde), V_tilde.getH()), W_tilde.getH())
	print("VWVtWt_tilde = " + str(VWVtWt_tilde))
	axis_VWVtWt_tilde, _ = get_axis_angle(VWVtWt_tilde)
	print("axis_VWVtWt_tilde = " + str(axis_VWVtWt_tilde))

	# Calcualte S, used to rotate U's axis to VWV†W†'s axis and back
	axis_S = numpy.cross(axis_U, axis_VWVtWt_tilde)
	print("axis_S = " + str(axis_S))

	theta_S = math.acos(abs(numpy.dot(axis_U, axis_VWVtWt_tilde)) / (numpy.linalg.norm(axis_U) * numpy.linalg.norm(axis_VWVtWt_tilde)) ) #TODO make sure this is right direction (do I need to negate???) ALSO MAKE SURE ARCCOS DIDN't FUCK EVERYTHING UP B/C OF DOMAIN AND STUFF
	print("theta_S = " + str(theta_S))
	print("numpy.dot(axis_U, axis_VWVtWt_tilde) = " + str(abs(numpy.dot(axis_U, axis_VWVtWt_tilde))))
	print("numpy.linalg.norm(axis_VWVtWt_tilde) = " + str(numpy.linalg.norm(axis_VWVtWt_tilde)))
	print("numpy.linalg.norm(axis_U) = " + str(numpy.linalg.norm(axis_U)))

	S = from_axis_angle(axis_S, theta_S)




	# The group commutator
	V = numpy.matmul(numpy.matmul(S, V_tilde), S.getH())
	W = numpy.matmul(numpy.matmul(S, W_tilde), S.getH())
	delta =  numpy.matmul(numpy.matmul(numpy.matmul(V, W), V.getH()), W.getH()) # V * W * V.getH() * W.getH()
	print("GC-Decompose: V = " + str(V))
	print("GC-Decompose: W = " + str(W))
	print("GC-Decompose: delta = " + str(delta))
	#print("GC-Decompose: make_SU2(delta) = " + str(make_SU2(delta)))
	print("GC-Decompose: U = " + str(U))

	# Verify values
	if not dist(I2, V) < c_gc_sqrt_epsilon or not dist(I2, W) < c_gc_sqrt_epsilon:
		print("GC_Decompose failed!")
	elif not dist(I2, delta) < epsilon_n_minus_1:
		print("GC_Decompose failed! d(I2, VWV†W†) >= epsilon_n_minus_1")
	elif not U == delta:
		print("GC_Decompose failed! U != VWV†W†")
	
	return V, W

def init_basic_approximation(G, epsilon_0: float, length_0: int, filepath: str = None):
	"""
	G            Gate set
	epsilon_0    .
	length_0     The length of the gate sequences made from G (i.e. the number of gates in each sequence).
	"""
	global g_e0_net

	assert(length_0 > 0)

	# If e0-net has been precomputed
	if filepath and os.path.isfile(filepath):
		with open(filepath, "rb") as f_in:
			e0_net = pickle.load(f_in)
			g_e0_net = e0_net
			return e0_net
	
	# Runtime benchmark
	start_time = time.time()

	if len(G) > 10 and length_0 > 10:
		#TODO making all permutations will be too slow
		pass

	max_e0_net_size = len(G) ** length_0
	e0_net = {}

	# All permutations of gate sequences of length length_0
	for seq in itertools.product(G, repeat=length_0):
		approx_gate = functools.reduce(numpy.matmul, seq)

		if approx_gate.tobytes() in e0_net:
			pass #print("init_basic_approximation: Found duplicate approx_gate " + str(approx_gate))
		else:
			e0_net[approx_gate.tobytes()] = (approx_gate, seq)

	# Runtime benchmark
	end_time = time.time()

	print("init_basic_approximation: Total time to generate e0-net: %u (s)" % (end_time - start_time))

	# Write e0-net to file
	if filepath:
		with open(filepath, "wb") as f_out:
			pickle.dump(e0_net, f_out)

	g_e0_net = e0_net
	return e0_net

def basic_approximation(U):
	"""
	U        Gate
	"""
	global g_e0_net

	#TODO
	
	return None

def Solovay_Kitaev(U: numpy.matrix, n: int):
	"""
	U        Gate
	n        depth

	Approximates U to an accuracy epsilon sub n.
	"""
	
	if n == 0:
		return basic_approximation(U)
	else:
		U_n_minus_1, U_n_minus_1_seq = Solovay_Kitaev(U, n - 1)
		V, W = GC_Decompose( numpy.matmul(U, U_n_minus_1.getH()) )

		V_n_minus_1, V_n_minus_1_seq = Solovay_Kitaev(V, n - 1)
		W_n_minus_1, W_n_minus_1_seq = Solovay_Kitaev(W, n - 1)
	
		#TODO
		#V_n_minus_1_conjtrans_seq = ?
		#W_n_minus_1_conjtrans_seq = ?

		U_n = V_n_minus_1 * W_n_minus_1 * V_n_minus_1.getH() * W_n_minus_1.getH() * U_n_minus_1
#TODO		U_seq = [V_n_minus_1_seq, W_n_minus_1_seq, V_n_minus_1_conjtrans_seq, W_n_minus_1_conjtrans_seq, U_n_minus_1_seq]
		U_seq = [gate for sublist in U_seq for gate in sublist] # Flatten list

		return U_n, U_seq



if __name__ == "__main__":

	# Parse arguments
	#TODO


	epsilon = 0.01
	

	
	"""
	This is an example gate set Dawson and Neilson used in their paper.
	"""
	G = [H, T, numpy.linalg.inv(T)] 


	os.system("rm e0_net.pickle") # For testing
	init_basic_approximation(G, 0.14, 15, "e0_net.pickle")
	
	print("g_e0_net:")
	for v in g_e0_net.values():
		approx_gate, gate_seq = v
		#print(approx_gate)
		#print(gate_seq)

#	S = Solovay_Kitaev(U, args.iterations)


	if not dist(U, S) < epsilon:
		print("Failed")
	else:
		print("Success")

	#TODO check length is O( (log(1/epsilon))^c )
	# c approx 3.97

	pass



'''

Figuring out how to implement the basic approximation lookup efficiently was challenging.
If you store a large number of basic approximations, a linear search through a list of them would be slow.


figuring out spectral norm presented difficulty.


'''

