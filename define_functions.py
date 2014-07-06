# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 17:40:08 2014

@author: rmphilipsen
"""
import global_vars as gv
import numpy as np
import cvxpy as cp
from operator import itemgetter
import lrs_nash as lrs
import copy
import time
import itertools
import cPickle as pickle

xrange1 = lambda start, end: xrange(start, end+1)

def prune_ftr_action_spaces(row_ftr,col_ftr):
    rowplayer = gv.producers[0]
    colplayer = gv.producers[1]
    print row_ftr
    input()
    rlist = []
    for action in row_ftr:
        for x,y,z in   itertools.permutations(action,3):
            if x[3] == y[2] and y[3] == z[2] and z[3] == x[2] and (x[0] + y[0]) * z[0] > 0:
                rlist.append(action)
    print len(rlist)
    new = [k for k,v in itertools.groupby(sorted(rlist))]
    print len(new)
    for i in new:
        row_ftr.remove(i)
    print row_ftr
    print "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"



def run_action_combinations(my_electricity_action_space,their_electricity_action_space,my_ftr_strategy,their_ftr_strategy,final_run = False):
    #Step 4: Market 1 settlement
    Rowplayer = gv.producers[0]
    Columnplayer = gv.producers[1]
    my_payoff_matrix_electricity_market = np.zeros((len(my_electricity_action_space),len(their_electricity_action_space)))
    their_payoff_matrix_electricity_market = np.zeros((len(my_electricity_action_space),len(their_electricity_action_space)))

    Rowplayer.set_ftr_parameters(eps=my_ftr_strategy)
    Columnplayer.set_ftr_parameters(eps=their_ftr_strategy)
    FTR_price, my_ftr, their_ftr= clear_FTR_market()
    gv.FTR_price = FTR_price
    #Step 5 and 11: Choose market 2 instance
    #Step 6 and 9: choose next market 2 action combination
    for col_move,row_move in itertools.product(their_electricity_action_space,my_electricity_action_space):
        index_row = my_electricity_action_space.index(row_move)
        index_col = their_electricity_action_space.index(col_move)
        #Step 7: Market 2 settlement
        #print "row move is", row_move
        #print "col move is", col_move
        problem = clear_Elec_market(row_move,col_move)
        problem.solve()
        expectedLMP = calculate_LMPs(gv.constraints)
        #print "Expected LMPs are", expectedLMP
        #for i in gv.sources:
        #    i.print_output()
        #Step 8: Populating Market 2 payoff matrix
        row_payoff = round(Rowplayer.calculate_profit(expectedLMP,FTR_obls=my_ftr),0)
        col_payoff = round(Columnplayer.calculate_profit(expectedLMP,FTR_obls=their_ftr),0)
        my_payoff_matrix_electricity_market[index_row,index_col] = row_payoff
        their_payoff_matrix_electricity_market[index_row,index_col] = col_payoff
    #Step 10: Run algorithm for equilibrium payoff
    a = lrs.Normal_Form_Game(my_payoff_matrix_electricity_market,their_payoff_matrix_electricity_market)
    a.solve()
    #But what if there are multiple equilibria?
    #We'll take the highest combined-utility equilibrium
    equi_list = [-9999,-9999]
    my_elec = my_electricity_action_space[0]
    their_elec = their_electricity_action_space[0]
    print my_payoff_matrix_electricity_market
    print their_payoff_matrix_electricity_market
    for e in a.equilibria:
        try:
            my_elec = my_electricity_action_space[e.row_strategy_distribution.index(1)]
            their_elec = their_electricity_action_space[e.col_strategy_distribution.index(1)]
            if e.row_utility+e.col_utility > equi_list[0]+equi_list[1]:
                equi_list = [e.row_utility,e.col_utility]
                print "Better PNE at", e.row_strategy_distribution.index(1), e.col_strategy_distribution.index(1)
        except ValueError:
            pass
    if final_run == True:
        print my_payoff_matrix_electricity_market
        print their_payoff_matrix_electricity_market
        print my_electricity_action_space
        print their_electricity_action_space
        return equi_list, [my_ftr,their_ftr], [my_elec,their_elec]
    return equi_list

def calculate_LMPs(constraints):
    """Calculate LMPs for all nodes based on the dual values for the
    relevant constraints. """
    LMPs = dict.fromkeys(range(len(gv.nodes)+len(gv.sources)),0)
    for i in constraints:
        for o in gv.sources + gv.nodes:
            for specific in o.LMP_relevant():
                if str(specific) == str(i):
                    LMPs[o.uid] += -1 * i.dual_value
                    continue
                #Does this loop return the correct value for slack node????? CHECK
                if o != gv.nodes[-1]:
                    for e in gv.edges:
                        n = e.LMP_relevant()
                        for line in n:
                            if str(line) == str(i):
                                LMPs[o.uid] += -1 * i.dual_value * gv.ptdf_matrix[e.uid,o.uid]
                                continue
    for k,v in LMPs.items():
        LMPs[k] = round(v,4)
    return LMPs

def clear_Elec_market(row_move,col_move):
    Rowplayer = gv.producers[0]
    Columnplayer = gv.producers[1]

    total_cost =  sum (s.bid_cost(multiplier=col_move[s_index]) for s_index,s in enumerate(Columnplayer.plants_owned)) + \
    sum (s.bid_cost(multiplier=row_move[s_index]) for s_index,s in enumerate(Rowplayer.plants_owned))
    p = cp.Problem(cp.Minimize(total_cost), gv.constraints)
    return p

def clear_FTR_market():
    """Based on current FTR bidding strategies, clear the market."""
    number_of_nodes = len(gv.nodes) + len(gv.sources)
    FTR_price = np.zeros((number_of_nodes,number_of_nodes))
    allocation_HENK = cp.Variable(len(gv.FTR_list))
    allocation_BERT = cp.Variable(len(gv.FTR_list))
    allocation = []
    for i in range(len(gv.FTR_list)):
        allocation.append (allocation_HENK[i] + allocation_BERT[i])
    Henk = gv.producers[0]
    Bert = gv.producers[1]
    constraints = []
    for line in gv.edges:
        this_constraint = 0
        for ftr_index, ftr in enumerate(gv.FTR_list):
            if ftr[0] == gv.slack:
                term = -1 * gv.ptdf_matrix[line.uid,ftr[1]] * allocation[ftr_index]
                this_constraint += term
            elif ftr[1] == gv.slack:
                term = gv.ptdf_matrix[line.uid,ftr[0]] * allocation[ftr_index]
                this_constraint += term
            else:
                term = (gv.ptdf_matrix[line.uid,ftr[0]] - gv.ptdf_matrix[line.uid,ftr[1]]) * allocation[ftr_index]
                this_constraint += term
        pos_addition = [this_constraint <= line.capacity]
        neg_addition = [-1 * this_constraint <= line.capacity]
        constraints.extend(pos_addition)
        constraints.extend(neg_addition)

    for index, i in enumerate(allocation_HENK):
        new = [i >= 0]
        if Henk.eps[index][0] < 0:
            new = [i == 0]
        constraints.extend(new)

    for index, i in enumerate(allocation_BERT):
        new = [i >= 0]
        if Bert.eps[index][0] < 0:
            new = [i == 0]
        constraints.extend(new)

    #Time to set up Objective function
    gain_HENK = sum(Henk.eps[i][0] * allocation_HENK[i] for i in range(len(allocation_HENK))) - \
    sum(cp.square(cp.sqrt(Henk.eps[i][1]) * allocation_HENK[i]) for i in range(len(allocation_HENK)))
    gain_BERT = sum(Bert.eps[i][0] * allocation_BERT[i] for i in range(len(allocation_BERT))) - \
    sum(cp.square(cp.sqrt(Bert.eps[i][1]) * allocation_BERT[i]) for i in range(len(allocation_BERT)))
    total_gain = gain_BERT + gain_HENK
    #Solve the problem
    p = cp.Problem(cp.Maximize(total_gain), constraints)
    p.solve()

    #Determine willingness to pay to set price.
    WtP = np.zeros((number_of_nodes,number_of_nodes))
    for i in xrange(len(allocation_HENK)):
        WtP[gv.FTR_list[i][0],gv.FTR_list[i][1]] = round(max(0,max(Henk.eps[i][0] - \
        2 * Henk.eps[i][1] * allocation_HENK[i].value, Bert.eps[i][0] - \
        2 * Bert.eps[i][1] * allocation_BERT[i].value)),2)

    FTRs = create_FTR_list()
    Henk_buys = []
    Bert_buys = []
    for index,item in enumerate(allocation_HENK):
        newitem = [item.value,FTRs[index][0],FTRs[index][1]]
        Henk_buys.append(newitem)
    for index,item in enumerate(allocation_BERT):
        newitem = [item.value,FTRs[index][0],FTRs[index][1]]
        Bert_buys.append(newitem)
    #return FTR_price, Henk_buys, Bert_buys
    return WtP, Henk_buys, Bert_buys

def create_FTR_list():
        """Return a list with possible FTRs."""
        FTR_list = [[n.uid,s.uid] for n in gv.nodes+gv.sources for s in gv.nodes+gv.sources if n.uid != s.uid and
        n.uid not in gv.colocator_list and s.uid not in gv.colocator_list]
        FTR_list = sorted(FTR_list, key=itemgetter(0,1))
        return FTR_list

def create_ptdf_matrix():
    """Do all matrix calculations required to end up with PTDF matrix"""
    number_of_nodes = len(gv.nodes) + len(gv.sources)
    number_of_edges = len(gv.edges)
    admittance_matrix = np.zeros((number_of_nodes,number_of_nodes))

    for index,item in enumerate(gv.edges):
        o = item
        #Element ii is the sum of inverse line reactances, off-diagonal elements are
        # negative inverse line reactance for that line
        admittance_matrix[o.in_node, o.out_node] = admittance_matrix.item((o.in_node, o.out_node)) + (-1 / o.reactance)
        admittance_matrix[o.out_node, o.in_node] = admittance_matrix.item((o.out_node, o.in_node)) +  (-1 / o.reactance)
        admittance_matrix[o.in_node, o.in_node] = admittance_matrix.item((o.in_node, o.in_node)) +  (1 / o.reactance)
        admittance_matrix[o.out_node, o.out_node] = admittance_matrix.item((o.out_node, o.out_node)) +  (1 / o.reactance)

    #Now define admittance matrix by looking at the nodes to which edges connect
    # We also use the node susceptance matrix for the PTDFs
    bus_branch_incidence_matrix = np.zeros((number_of_edges,number_of_nodes))
    branch_susceptance_matrix = np.zeros((number_of_edges,number_of_edges))

    for index,item in enumerate(gv.edges):
        bus_branch_incidence_matrix[index,item.in_node] = -1
        bus_branch_incidence_matrix[index,item.out_node] = 1

        branch_susceptance_matrix[index,index] = 1/ item.reactance
    node_susceptance_matrix = np.dot(branch_susceptance_matrix,bus_branch_incidence_matrix)

    #Set the last Load node as slack, whose columns we remove from the matrices
    gv.slack = gv.nodes[-1].uid

    reduced_am = np.delete(admittance_matrix, (gv.slack), axis=0)
    reduced_am = np.delete(reduced_am, (gv.slack), axis=1)
    X=np.linalg.inv(reduced_am)
    reduced_bsm = np.delete(node_susceptance_matrix, (gv.slack), axis=1)
    ptdf_matrix = np.dot(reduced_bsm,X)
    return ptdf_matrix

def create_FTR_payoff_matrix():
    """Return a matrix with nodal price differences"""
    number_of_nodes = len(gv.nodes) + len(gv.sources)
    FTR_payoff = np.zeros((number_of_nodes,number_of_nodes))
    for n in gv.nodes+gv.sources:
        for s in gv.nodes+gv.sources:
            FTR_payoff[s.uid,n.uid] = gv.LMP[s.uid] - gv.LMP[n.uid]
    return FTR_payoff

def ConvertToNFG(m1,m2):
    """
    #Iteration order should be as follows: first player 1 plays strategy 1 for all
    #player 2 strategies, then moves on to 2 for all player 2 strategies, etc.
    #This means we first examine all rows for column 1, then all rows for column 2.
    """
    docstring = "NFG 1 R \"First attempt\" \n"
    description = "{ \"Row Player\" \"Column Player\" } { %d %s } \n \n" % (len(m1),len(m1[0]) )
    nfgform = " "
    nfgform += docstring
    nfgform += description
    it = np.nditer(m1, flags=['multi_index'], order="F")
    while not it.finished:
        column_value = m2[it.multi_index]
        row_value = m1[it.multi_index]
        nfgform += str(row_value)
        nfgform += " "
        nfgform += str(column_value)
        nfgform += " "
        it.iternext()
    return nfgform



def determine_FTR_bid(load_pickle=0):

    """Return parameters for the FTR auction
    Comments indicate the step from Rocha & Das, 2012.
    Step 0 is implied in 1 & 2"""
     #Step 1: Initialise Market 1 action combinations
    Rowplayer = gv.producers[0]
    Columnplayer = gv.producers[1]
    #FTR_action_space = [-0.1,0.9,1,1.1,1.2]
    #FTR_action_space = [0.9,1,1.1]
    FTR_action_space = [0.8,1.2]
    #scenario_elec_list = [0.8,1,1.2]
    scenario_elec_list = [0.8,1.2]
    #scenario_elec_list = [0.8,1,1.1,1.3]
    total = [[[n,i[0],i[1]] for n in FTR_action_space] for i in gv.FTR_list]
    scenario_FTR_list = list(itertools.product(*total))

    rlist = [item for item in scenario_FTR_list for x in item for y in item if x[0] * y[0] < 0 and x[1] == y[2] and y[1] == x[2]]
    new = [k for k,v in itertools.groupby(sorted(rlist))]
    for i in new:
        scenario_FTR_list.remove(i)

    #At this point, the lists contain multipliers, not the actual actions yet
    my_ftr_scenarios = copy.deepcopy(scenario_FTR_list)
    their_ftr_scenarios = copy.deepcopy(scenario_FTR_list)

    #Set up two lists, one for each producer, to create the actual action spaces
    my_FTR_action_space = [[[round(c_item[0] * Rowplayer.original_eps[c_index][0],2),Rowplayer.original_eps[c_index][1],
                             c_item[1],c_item[2]] for c_index, c_item in enumerate(strategy)] for strategy in my_ftr_scenarios ]
    their_FTR_action_space = [[[round(c_item[0] * Columnplayer.original_eps[c_index][0],2),Columnplayer.original_eps[c_index][1],
                                c_item[1],c_item[2]] for c_index, c_item in enumerate(strategy)] for strategy in their_ftr_scenarios]
    for i in my_FTR_action_space:
        for item in i:
            if item[0] < 0:
                item[0] = -999
            if gv.policy == "hybrid":
                if item[2] not in [y.uid for y in Rowplayer.plants_owned] and item[2] != 1 and item[0] > 0:
                    item[0] = -999
                if abs(item[0]) < 0.01:
                    item[0] = -999
    my_FTR_action_space = [k for k,v in itertools.groupby(sorted(my_FTR_action_space))]

    for i in their_FTR_action_space:
        for item in i:
            if item[0] < 0:
                item[0] = -999
            if gv.policy == "hybrid":
                if item[2] not in [y.uid for y in Columnplayer.plants_owned] and item[2] != 4 and item[0] > 0:
                    item[0] = -999
                if abs(item[0]) < 0.01:
                    item[0] = -999
    their_FTR_action_space = [k for k,v in itertools.groupby(sorted(their_FTR_action_space))]
    print my_FTR_action_space
    print their_FTR_action_space

    #Step 2: Initialise Market 2 action combinations
    a = copy.deepcopy(scenario_elec_list)
    b = copy.deepcopy(scenario_elec_list)
    if len(Rowplayer.plants_owned) > 1:
        my_electricity_action_space = list(itertools.product(a, repeat=len(Rowplayer.plants_owned)))
    else:
        my_electricity_action_space = list([x] for x in a)
    if len(Columnplayer.plants_owned) > 1:
        their_electricity_action_space = list(itertools.product(b, repeat=len(Columnplayer.plants_owned)))
    else:
        their_electricity_action_space = list([x] for x in b)
    del a, b, total, scenario_FTR_list, rlist
    my_ftr_payoff_matrix = np.zeros((len(my_FTR_action_space),len(their_FTR_action_space)))
    their_ftr_payoff_matrix = np.zeros((len(my_FTR_action_space),len(their_FTR_action_space)))


    """
    ######Testing action spaces
    my_FTR_action_space = [[[-999, 0.01, 0, 1], [-999, 0.01, 0, 4], [-999, 0.01, 1, 0], [-999, 0.01, 1, 4], [0.74, 0.01, 4, 0], [0.37, 0.01, 4, 1]]]
    their_FTR_action_space = [[[-999, 0.01, 0, 1], [-999, 0.01, 0, 4], [-999, 0.01, 1, 0], [3.87, 0.01, 1, 4], [0.74, 0.01, 4, 0], [-999, 0.01, 4, 1]]]
    my_electricity_action_space = [(1.2, 0.8)]
    their_electricity_action_space = [(0.8, 1.2)]
    """
    if load_pickle == 0:
        print "Calculating payoff matrix. This may take a while."
        #Step 3 and 13: choose next action 1 market combination
        #All created by the generator
        for my_ftr_strategy,their_ftr_strategy in itertools.product(my_FTR_action_space,their_FTR_action_space):
            index_f = my_FTR_action_space.index(my_ftr_strategy)
            index_g = their_FTR_action_space.index(their_ftr_strategy)
            print "Currently processing item",index_f+1,"-",index_g+1,"of",len(my_FTR_action_space),"-",len(their_FTR_action_space)
            start_time = time.time()

            #Clearing both markets and finding the electricity market equilibrium is in the function
            equi_list = run_action_combinations(my_electricity_action_space,their_electricity_action_space,my_ftr_strategy,their_ftr_strategy)

            #Step 12: populate Market 1 payoff matrix
            my_ftr_payoff_matrix[index_f,index_g] = equi_list[0]
            their_ftr_payoff_matrix[index_f,index_g] = equi_list[1]
            if index_f % 10 == 0 and index_g == 0:
                print "Time for an intermediary save."
                pickle.dump( my_ftr_payoff_matrix, open( "subset%s_rowplayer.p" % index_f, "wb" ) )
                pickle.dump( their_ftr_payoff_matrix, open( "subset%s_colplayer.p" % index_f, "wb" ) )
            print "That FTR action combination took me", time.time() - start_time, "seconds"

        pickle.dump( my_ftr_payoff_matrix, open( "firstrun_rowplayer.p", "wb" ) )
        pickle.dump( their_ftr_payoff_matrix, open( "firstrun_colplayer.p", "wb" ) )
        print "Saved pickles."
    else:
         my_ftr_payoff_matrix = pickle.load( open( "firstrun_rowplayer.p", "rb" ) )
         their_ftr_payoff_matrix = pickle.load( open( "firstrun_colplayer.p", "rb" ) )
    print "Calculating equilibria."
    """
    nfgstring = fn.ConvertToNFG(my_ftr_payoff_matrix,their_ftr_payoff_matrix)
    g = neng.Game(nfgstring)
    a = g.findEquilibria(method='pne')
    x = g.printNE(a, payoff=True)
    print my_ftr_payoff_matrix
    print their_ftr_payoff_matrix
    #qwe = g.getPNE()
    input("AAAAAAAAAAAAAAAAAA")

    """
    print my_ftr_payoff_matrix,their_ftr_payoff_matrix
    a = lrs.Normal_Form_Game(my_ftr_payoff_matrix,their_ftr_payoff_matrix)
    a.solve()
    list_of_ftr_equilibria = [-9999,-9999]
    my_ftr_strategy = my_FTR_action_space[0]
    their_ftr_strategy = their_FTR_action_space[0]
    for e in a.equilibria:           
        #Same principle as above. If there are multiple equilibria, take the highest total payoff
        #...but only takes PNEs
        try:
            my_ftr_strategy = my_FTR_action_space[e.row_strategy_distribution.index(1)]
            their_ftr_strategy = their_FTR_action_space[e.col_strategy_distribution.index(1)]
            if e.row_utility+e.col_utility > list_of_ftr_equilibria[0]+list_of_ftr_equilibria[1]:
                list_of_ftr_equilibria = [e.row_utility,e.col_utility]
                print "Found FTR PNE at", e.row_strategy_distribution.index(1), e.col_strategy_distribution.index(1)
        #Won't be able to find '1' if there's a mixed-strategy equilibrium
        except ValueError:
            continue

    #Manually set the optimal results from NENG

    """
    #We now have a combination of pure strategies that result in the highest total payoff.
    #What is left is to determine the accompanying electricity market strategy.
    #Can't look back, so must solve the problem once more.
    """
    equi_list, ftrs, elecs = run_action_combinations(my_electricity_action_space,their_electricity_action_space,my_ftr_strategy,their_ftr_strategy,final_run = True)
    my_ftr = ftrs[0]
    their_ftr = ftrs[1]
    my_elec = elecs[0]
    their_elec = elecs[1]
    print my_ftr_payoff_matrix
    print their_ftr_payoff_matrix
    return my_ftr, my_elec, their_ftr, their_elec, my_ftr_strategy, their_ftr_strategy


if __name__ == '__main__':
    a = np.array([[-2125,-4625],[300,875]])
    b = np.array([[-175, 925], [750, 425]])
    test = ConvertToNFG(a,b)
    g = neng.Game(test)
    a = g.getPNE()
    print g.printNE(a)
