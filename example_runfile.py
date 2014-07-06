# -*- coding: utf-8 -*-
"""
Created on Fri Jun 06 16:30:34 2014

@author: rphilipsen
"""
from network_solver import solve_network
from classDefinitions import Edge, Source, Producer, Sink
import time
"""
#This network is inspired by Rourke(2003), but there are some differences.
#The congested edge is now between the two generators, for example.
"""
sources = []
sinks = []
edges = []
producers = []
ignore_nodes = []
policy=""
sources.append(Source(100,lin_term=8, quad_term=0.01))
sources.append(Source(100,lin_term=10.10, quad_term=0.01))
sources.append(Source(100,lin_term=10, quad_term=0.01))
sources.append(Source(150,lin_term=14.60, quad_term=0.01))
sinks.append(Sink(180))
#sinks.append(Sink(1))


edges.append(Edge(9999,1))
edges.append(Edge(20,1))
edges.append(Edge(9999,1))
edges.append(Edge(9999,1))
edges.append(Edge(9999,1))
#edges.append(Edge(9999,1))

#Initialise poducers and give them plants
producers.append(Producer("Rowplayer"))
producers.append(Producer("Columnplayer"))
producers[0].possess(sources[0])
producers[0].possess(sources[2])
producers[1].possess(sources[1])
producers[1].possess(sources[3])

"""To vastly reduce computational complexity, these nodes are ignored in
   FTR calculations. Add nodes which are end points with unlimited capacity"""
ignore_nodes.append(sources[2])
ignore_nodes.append(sources[1])

#connect edges to their respective nodes: connect(from,to)
edges[0].connect(sources[0],sources[1])
edges[1].connect(sources[0],sinks[0])
edges[2].connect(sources[3],sinks[0])
edges[3].connect(sinks[0],sources[2])
edges[4].connect(sources[1],sinks[0])
#edges[5].connect(sources[0],sinks[1])

#Now wrap it into a network list to pass fewer arguments
network = [x for x in sources,sinks,edges,ignore_nodes]

#Start the counter, and call the solver
start_time = time.time()
solve_network(producers,network,policy)
print time.time() - start_time, "seconds"
