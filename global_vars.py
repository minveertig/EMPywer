# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 16:58:51 2014

@author: rmphilipsen
"""

global nodes, sources, edges, ptdf_matrix, LMP, constraints, slack
slack = 9999
nodes = []
sources = []
edges = []
producers = []
constraints = []
LMP = {}
policy = "none"
colocator_list = []
FTR_price = []
FTR_list = []
virtual_lines = []
number_of_nodes = 0

"""
ptdf_matrix not initialised here, but declared global for access purposes
pdtf_matrix has dimensions M lines(=rows) by N-1 nodes(=columns), with element i,j equal to
 fraction of power flowing through line i if injected at node j and extracted at reference node
 For flow from  p to q through i, PTDF_i,p - PTDF_i,q
"""
