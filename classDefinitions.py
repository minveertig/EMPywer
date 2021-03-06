'''
Created on 8 feb. 2014

@author: Rens Philipsen
'''
from __future__ import division
import cvxpy as cp
import numpy as np
import global_vars as gv
import define_functions as fn


np.set_printoptions(threshold=np.nan, suppress=True, linewidth=10000)

range1 = lambda start, end: range(start, end+1)
xrange1 = lambda start, end: xrange(start, end+1)


class Node(object):
    """Define superclass for sources and sinks, so that they may share id"""
    _ID = 0
    def __init__(self):
        self.uid = Node._ID
        Node._ID += 1

class Producer(object):
    """A producing company, possessing a number of source nodes"""
    def __init__ (self,name, eps=0, tau=0):
        self.name = name
        self.plants_owned = []
        self.FTR_obligations_owned = []
        self.LMP_expectations = {}
        self.FTR_expectations = np.zeros((gv.number_of_nodes,gv.number_of_nodes))
        self.money = 0
        self.original_tau = tau
        self.original_eps = eps
        self.tau = tau
        self.eps = eps

    def possess(self,plant):
        self.plants_owned.append(plant)
        plant.owner = self

    def determine_production_function(self):
        return 0

    def set_ftr_parameters(self,eps=[], type="update", congestion_rent=0):
        """tau and eta are column vectors, with _1 the parameter for FTR_list_1"""
        if type == "update":
            self.eps = eps
        elif type == "initial":
            FTRlist = fn.create_FTR_list()
            nodelist = [n.uid for n in self.plants_owned]
            if gv.SA:
                if 2 in nodelist:
                    nodelist.append(3)
                if 1 in nodelist:
                    nodelist.append(4)
            epsilon_list = []
            for x,y in FTRlist:
                if x in nodelist:
                    if self.FTR_expectations[x,y] != 0:
                        expected_quantity = congestion_rent / self.FTR_expectations[x,y]
                    else:
                        expected_quantity = 0
                    intercept = self.FTR_expectations[x,y] + 0.01 * expected_quantity
                    new_eps = [round(intercept,2), 0.005, x, y]
                    #print "I expect to need", expected_quantity,"at price",self.FTR_expectations[x,y]
                else:
                    new_eps = [round(self.FTR_expectations[x,y],2), 0.005, x, y]
                epsilon_list.append(new_eps)
            self.original_eps = epsilon_list
            self.eps = epsilon_list

    def buy_FTR(self,quant,from_node,to_node):
        """FTRs are stored as a list. P_ij = T_ij*(p_j-p_i)"""
        new_FTRs = [quant,from_node,to_node]
        self.FTR_obligations_owned.append(new_FTRs)

    def calculate_profit(self,LMP,FTR_obls=None):
        """Using the LMPs as input, returns the profit from both electricity and FTRs. """
        if not FTR_obls:
            FTR_obls = self.FTR_obligations_owned
        electricity_earnings = sum(round(i.production.value,2) * LMP[i.uid] - (
        i.constant_cost + i.lin_term * round(i.production.value,2) + round(i.production.value,2) *
        round(i.production.value,2) * i.quad_term) for i in self.plants_owned)
        if gv.policy == "none":
            obligation_earnings = 0
            for i in FTR_obls:
                rho = (LMP[i[2]] - LMP[i[1]])
                pi = gv.FTR_price[i[1],i[2]]
                q = i[0]
                payoff_this_FTR = (rho - pi) * q
                obligation_earnings += payoff_this_FTR
        elif gv.policy == "hybrid":
            obligation_earnings = 0
            production_dict = {n.uid: round(n.production.value,2) for n in self.plants_owned}
            if gv.SA:
                if 2 in [n.uid for n in self.plants_owned]:
                    production_dict[3] = production_dict[2]
                if 1 in [n.uid for n in self.plants_owned]:
                    production_dict[4] = production_dict[3]
            #production_dict[4] = sum(round(n.production.value,2) for n in self.plants_owned if n > 1)
            for i in FTR_obls:
                q = round(i[0],2) #number of ftrs held
                if q > 0:
                    rho = (LMP[i[2]] - LMP[i[1]])
                    pi = gv.FTR_price[i[1],i[2]] #Cost of FTR
                    p = production_dict.get(i[1]) #Leftove production at that node
                    if not p:
                        p = 0
                    payoff_this_FTR = rho * min(q,p) - pi * q
                    obligation_earnings += payoff_this_FTR
                    production_dict[i[1]] = p-min(q,p)
        elif gv.policy == "hedge":
            obligation_earnings = 0
            for i in FTR_obls:
                if i[1] in [n.uid for n in self.plants_owned]:
                    rho = (LMP[i[2]] - LMP[i[1]])
                    pi = gv.FTR_price[i[1],i[2]]
                    q = i[0]
                    payoff_this_FTR = (rho - pi) * q
                else:
                    pi = gv.FTR_price[i[1],i[2]]
                    q = i[0]
                    payoff_this_FTR = -1 * pi * q
                obligation_earnings += payoff_this_FTR
        #print "I hold prtfolio", FTR_obls
        #print "I earn",electricity_earnings,"and", obligation_earnings
        return electricity_earnings + obligation_earnings

    def update_expectations(self,LMP):
        self.LMP_expectations = LMP
        self.FTR_expectations = np.zeros((gv.number_of_nodes,gv.number_of_nodes))
        for x in range(gv.number_of_nodes):
            for y in range(gv.number_of_nodes):
                self.FTR_expectations[x,y] = LMP[y] - LMP[x]


 # Define what an edge looks like
class Edge(object):
    _ID = 0

    """ An undirected, capacity-limited edge. Zero resistance, but reactance may be given. """
    def __init__(self, capacity, reactance=1):
        self.capacity = capacity
        self.reactance = reactance
        self.flow = cp.Variable()
        self.uid = Edge._ID
        Edge._ID += 1

    def set_index(self):
        self.uid = gv.edges.index(self)

    # Connects two nodes via the edge.
    def connect(self, out_node, in_node):
        in_node.edge_flows.append(self.flow)
        out_node.edge_flows.append(-self.flow)
        self.in_node = in_node.uid
        self.out_node = out_node.uid

    # Returns the edge's internal constraints.
    def constraints(self):
        cons = []
        capacity_limit = [cp.abs(self.flow) <= self.capacity]
        ptdf_constraint = [sum(gv.ptdf_matrix[self.uid,s.uid] * -1 * s.accumulation for s in \
        gv.sources + gv.nodes if s.uid != gv.slack) == self.flow]
        cons.extend(ptdf_constraint)
        cons.extend(capacity_limit)
        #ptdf_constraint = [cp.abs(sum(gv.ptdf_matrix[self.uid,s.uid] * -1 * s.accumulation for s in \
        #gv.sources + gv.nodes if s.uid != gv.slack)) <= self.capacity]
        #cons.extend(ptdf_constraint)
        return cons

    def LMP_relevant(self):
        """returns constraint for LMP congestion component. """
        cons = []
        n = [sum(gv.ptdf_matrix[self.uid,s.uid] * -1 * s.accumulation for \
        s in gv.sources+gv.nodes if s.uid!=gv.slack) == self.flow]
        #n = [cp.abs(sum(gv.ptdf_matrix[self.uid,s.uid] * -1 * s.accumulation for s in \
        #gv.sources + gv.nodes if s.uid != gv.slack)) <= self.capacity]
        cons.extend(n)
        return cons

    def print_output(self):
        print "Flow through edge between", self.out_node,"-", self.in_node, ":", \
        round(self.flow.value,  2), ". Line capacity is", self.capacity

# Now define what a node looks like
class Sink(Node):
    """ A node with accumulation; a load-serving entity. """
    def __init__(self, accumulation=0):
        self.accumulation = accumulation
        super(Sink, self).__init__()
        self.edge_flows = []

    # Returns the node's internal constraints.
    def constraints(self):
        return [sum(f for f in self.edge_flows) == self.accumulation]

    def LMP_relevant(self):
        """returns the constraint for the LMP energy component. """
        return [sum(f for f in self.edge_flows) == self.accumulation]

    def print_output(self):
        print "Load in node ", self.uid, ":", round(self.accumulation, 2)

# Now to create sources and sinks
class Source(Node):
    """ A Source node; a power plant, or a number of aggregated plants"""
    capacity = 0
    production = 0
    profit = 0
    marginalCost = 0
    inflate = 0

    def __init__(self, capacity, marginalCost=0, lin_term=0, quad_term=0, constant_cost=0):
        self.owner = 0
        self.edge_flows = []
        #self.uid = 99
        self.capacity = capacity
        self.marginalCost = marginalCost
        self.production = cp.Variable()
        self.accumulation = -1 * self.production
        self.lin_term = lin_term
        self.quad_term = quad_term
        self.constant_cost = constant_cost
        super(Source, self).__init__()

    # Returns the node's internal constraints.
    def constraints(self):
        constraint_list = []
        constraint_list.append(sum(f for f in self.edge_flows) == self.accumulation)
        constraint_list.append(self.production <= self.capacity)
        constraint_list.append(self.production >= 0)
        return constraint_list

    def LMP_relevant(self):
        return [sum(f for f in self.edge_flows) == self.accumulation]

    def bid_cost(self, multiplier=1):
        if self.quad_term == 0:
            return (self.constant_cost + self.lin_term * self.production) * multiplier
        return (self.constant_cost + cp.quad_form(self.production,self.quad_term) + self.lin_term * self.production) * multiplier

    def false_cost(self):
        if self.quad_term == 0:
            return (self.constant_cost + self.lin_term * self.production) * self.inflate
        return (self.constant_cost + cp.quad_form(self.production,self.quad_term) + self.lin_term * self.production) * self.inflate

    def print_output(self):
        print "Production in node ", self.uid, ":", round(self.production.value, 2), \
        "Production capacity is", self.capacity

    def save_comp(self):
        """Call this function after running competitive scenario"""
        self.comp_production = self.production.value
