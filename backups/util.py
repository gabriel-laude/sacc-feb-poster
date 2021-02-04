#!/usr/bin/python

__author__ = 'Jeremy O. Richardson'

import math
import numpy

#########################################

class ensembles:
	microcanonical = 0 # constant E
	canonical = 1      # constant T

#########################################

def filename(name, ID=None):
   fname = 'data/' + str(name)
   if ID is not None:
      fname = fname + '-' + str(ID)
   return fname + '.dat'

#########################################

def ordinal(n):
	"""input cardinal number, return ordinal"""
	ord = str(n)
	if n % 10 == 1 and n % 100 != 11:
		ord += 'st'
	elif n % 10 == 2 and n % 100 != 12:
		ord += 'nd'
	elif n % 10 == 3 and n % 100 != 13:
		ord += 'rd'
	else:
		ord += 'th'
	return ord

#########################

def plural(n):
	if n == 1: return ''
	else: return 's'

#########################

greek = {1:'mono', 2:'di', 3:'tri', 4:'tetra', 5:'penta', 6:'hexa', 7:'hepta', 8:'octa', 9:'nona', 10:'deca'}

#########################

def fit(x, shape):
	"""return x to fit shape"""
	while x.ndim < len(shape):
		x = numpy.expand_dims(x, 0)
	return x

#########################

def stick(x, y):
	if len(x) == 0: return y
	elif len(y) == 0: return x
	else: return numpy.concatenate((x,y))

#########################

def diagtensor(v):
	v = numpy.asarray(v)
	res = numpy.zeros(2*v.shape, v.dtype)
	i = numpy.arange(v.size)
	fi = i + i*v.size
	res.flat[fi] = v
	return res

#########################

def backwardsbroadcast(a, x):
	"""append extra axes to a to be able to broadcast with x"""
	a.shape += tuple(numpy.ones(x.ndim-a.ndim, numpy.int))
	return a

#########################

def share(apples, people):
	"""share as evenly as possible the apples among the people
	return index[:] such that person i gets apples[index[i]:index[i+1]]
	the people at the end tend to get more"""
	index = [0]
	for person in range(people):
		index.append(index[-1]+(apples-index[-1])/(people-person))
	return index

#########################

def sequential(N, length=None, order=2):
	"""Return list of digits which make up binary (or other base) representation of N."""
	binary = []
	if length is None:
		length = int(math.ceil(math.log(N, order)))
	for i in range(1,length+1):
		binary.append(N / order**(i-1) % order)
		N -= binary[-1] * order**(i-1)
		if length is None and N==0:
			break
	binary.reverse()
	return binary

#########################

detbrackets = ('','|','|','') # brackets for determinants
def printm(*args, **kwargs):
	"""Display matrix neatly among text.
	The first line takes all text and first row of matrices and following lines the remaining rows.
	Arguments should include strings and matrices interspersed.
	Keyword arguments:
	fmt    -- format description for each element of matrix e.g. '% .3f' (default None is ndarray default)
	spaces -- add a space after each entry (default True)
	brackets -- 4-tuple of open and closing brackets to use (default ('[','[',']',']'))
	"""
	options = {'fmt':None, 'spaces':True, 'brackets':('[','[',']',']')}
	for var in options:
		if var in kwargs:
			options[var] = kwargs[var]
			del kwargs[var]
	if len(kwargs) > 0:
		raise Exception("unknown keyword arguments: %s" % kwargs)
	lines = [""] # initialize blank first line
	for arg in args:
		if type(arg) == type(""):
			lines[0] += arg
		elif type(arg) == numpy.ndarray:
			if arg.ndim == 1: # column vector
				if options['fmt'] is None:
					strs = str(arg.reshape(-1,1)).split('\n')
					for i in range(len(strs)):
						strs[i] = strs[i][1:] # delete initial bracket to make it a column vector
					strs[-1] = strs[-1][:-1] # delete final bracket to make it a column vector
				else:
					column = [options['fmt']%element for element in arg]
					strs = [options['brackets'][1] + element + options['brackets'][2] for element in column]
				initial = final = ''
			elif arg.ndim == 2: # matrix
				if options['fmt'] is None:
					strs = str(arg).split('\n')
					initial =  final = ''
				else:
					rows = [[options['fmt']%element for element in row] for row in arg]
					strs = [options['brackets'][1] + ' '.join(row) + options['brackets'][2] for row in rows]
					initial, final = options['brackets'][0:4:3]
			else:
				raise TypeError("invalid shape array %s"%arg.shape)
			len0 = len(lines[0] + initial) # length of first line
			lines[0] += initial + strs[0]
			while len(strs) > len(lines):
				lines.append("")
			for n, string in enumerate(strs[1:]):
				lines[n+1] += ' '*(len0-len(lines[n+1])) # pad with zeros
				lines[n+1] += string
			lines[-1] += final
		else:
			raise TypeError("invalid argument: %s" % arg)
		if options['spaces']:
			lines[0] += ' '
	print('\n'.join(lines))

#########################

if __name__ == "__main__":
	#for i in range(25):
    	#	print(ordinal(i), end=' ')
	#print()
	#for i in range(99,125):
	#	print(ordinal(i), end=' ')
	#print()
	#for i in range(1,11):
	#	print(greek[i]+'mer', end=' ')
	#print()
	print(share(17,5))
	v = numpy.arange(6)
	print(diagtensor(v))
	v = numpy.arange(6).reshape(3,2)
	print(diagtensor(v))
	v = numpy.arange(6).reshape(6,1)
	print(diagtensor(v))
	print("100 in base 3", sequential(100, order=3))
	A = numpy.sqrt(numpy.arange(12).reshape(4,3))
	printm("vector is",A[0],"and matrix is",A,"don't you know!")
	printm("det A is",A,"and det A+1 is",A+1,"don't you know!", fmt='%.2f', spaces=True, brackets=detbrackets)
