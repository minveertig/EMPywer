'''
Created on 8 feb. 2014

@author: Rens Philipsen
'''
from __future__ import division
import cvxpy as cp
import global_vars as gv
import define_functions as fn

def solve_network(producers,network,policy="none"):
    """The function that takes the input and calls everything it needs.
       Inputs are all lists with relevant objects, policy is a string."""
    gv.policy = policy
    gv.producers = producers
    gv.sources = network[0]
    gv.nodes = network[1]
    gv.edges = network[2]
    gv.number_of_nodes = len(gv.nodes) + len(gv.sources)
    gv.colocator_list = [i.uid for i in network[3]]
 
    #Quick reference for the players
    Rowplayer = producers[0]
    Columnplayer = producers[1]

    #Create globals variables for PTDFs and FTRs
    gv.ptdf_matrix = fn.create_ptdf_matrix()
    gv.FTR_list = fn.create_FTR_list()
    """Network initialisation above"""

    #Set up problem, solve it
    for o in gv.nodes + gv.edges + gv.sources:
        gv.constraints += o.constraints()
    total_cost =  sum (s.bid_cost() for s in gv.sources)
    p = cp.Problem(cp.Minimize(total_cost), gv.constraints)
    result = p.solve()
    print "Competitive bidding results in", result

    #Calculate LMPs and print them as output
    gv.LMP = fn.calculate_LMPs(gv.constraints)
    comp_LMPs = gv.LMP
    print "LMPs at respective nodes are:"
    for k,v in gv.LMP.items():
        gv.LMP[k] = round(v,4)
        if k not in gv.colocator_list:
            print "node",k,":",gv.LMP[k],"Eur/MWh"
    for s in gv.sources + gv.edges:
        s.print_output()
    for s in gv.sources:
        s.save_comp()

    input("Resting here.")
    #Define which constraints we're interested in evaluating, match them with results, and get the values
    gv.LMP = fn.calculate_LMPs(gv.constraints)
    comp_payment_to_generators = sum(gv.LMP[s.uid] * s.production.value for s in gv.sources)
    comp_payment_by_load = sum(gv.LMP[s.uid] * s.accumulation for s in gv.nodes)
    comp_congestion_rent = comp_payment_by_load - comp_payment_to_generators

    #Set the competitive expectations producers have of FTRs
    for p in gv.producers:
        p.update_expectations(gv.LMP)
        p.set_ftr_parameters(type="initial", congestion_rent=comp_congestion_rent)


    VoLL = 100
    comp_payment_to_generators = sum(gv.LMP[s.uid] * s.production.value for s in gv.sources)
    comp_consumer_benefit = sum((VoLL - gv.LMP[s.uid]) * s.accumulation for s in gv.nodes)
    comp_congestion_rent = comp_payment_by_load - comp_payment_to_generators
    comp_Rowplayer_profit = Rowplayer.calculate_profit(gv.LMP)
    comp_Columnplayer_profit = Columnplayer.calculate_profit(gv.LMP)


    """Forward operating point now known. Below is the non-competitive situation."""
    #First, we determine bids for FTRs and electricity markets
    iplay, my_elec, heplay, his_elec, row_ftr_strategy, col_ftr_strategy = fn.determine_FTR_bid(load_pickle=0)
    print iplay
    print my_elec
    print heplay
    print his_elec

    #Producers buy FTRs, and some output is printed.
    for i in iplay:
        Rowplayer.buy_FTR(i[0],i[1],i[2])
    for i in heplay:
        Columnplayer.buy_FTR(i[0],i[1],i[2])
    print "Rowplayer plays", row_ftr_strategy
    print "Columnplayer plays", col_ftr_strategy

    #Set the multiplier for the false-cost function.
    for index, s in enumerate(Rowplayer.plants_owned):
        s.inflate = my_elec[index]
    for index, s in enumerate(Columnplayer.plants_owned):
        s.inflate = his_elec[index]

    #Printing FTR actions as output we can read.
    for count,i in enumerate(iplay):
        print "For node path",i[1],"-",i[2],"Rowplayer buys",round(i[0],2),"and Columnplayer buys",round(heplay[count][0],2),"at",round(gv.FTR_price[i[1],i[2]],2),"each."

    #FTRs are now in possession, time to clear the gamed electricity market.
    total_cost =  sum (s.false_cost() for s in gv.sources)
    p = cp.Problem(cp.Minimize(total_cost), gv.constraints)
    result = p.solve()
    print "False cost functions optimise to", result

    #Calculate LMPs and print them as output
    gv.LMP = fn.calculate_LMPs(gv.constraints)
    print "LMPs at respective nodes are:"
    for k,v in gv.LMP.items():
        gv.LMP[k] = round(v,4)
        print "node",k,":",gv.LMP[k],"Eur/MWh"

    #Now print some output for the network items
    for i in gv.nodes+gv.sources+gv.edges:
        i.print_output()

    payment_to_generators = sum(gv.LMP[s.uid] * s.production.value for s in gv.sources)
    consumer_benefit = sum((VoLL - gv.LMP[s.uid]) * s.accumulation for s in gv.nodes)
    payment_by_load = sum(gv.LMP[s.uid] * s.accumulation for s in gv.nodes)
    congestion_rent = payment_by_load - payment_to_generators
    total_Rowplayer_profit = Rowplayer.calculate_profit(gv.LMP)
    total_Columnplayer_profit = Columnplayer.calculate_profit(gv.LMP)

    #Merchandise surplus
    K_m = (congestion_rent - comp_congestion_rent) / comp_congestion_rent

    #Consumer surplus
    K_d = (consumer_benefit - comp_consumer_benefit) / comp_consumer_benefit

    #Producer surplus
    K_h = (total_Rowplayer_profit - comp_Rowplayer_profit) / comp_Rowplayer_profit
    K_b = (total_Columnplayer_profit - comp_Columnplayer_profit) / comp_Columnplayer_profit
    K_p = (total_Rowplayer_profit - comp_Rowplayer_profit + total_Columnplayer_profit - comp_Columnplayer_profit) \
    / (comp_Rowplayer_profit + comp_Columnplayer_profit)
    #Social Surplus

    #K_s = (total_social_surplus - comp_social_surplus) / comp_social_surplus
    print "merchants make", congestion_rent,"instead of", comp_congestion_rent
    print "Merchant surplus is",K_m
    print "Consumers make", consumer_benefit,"instead of",  comp_consumer_benefit
    print "Consumer surplus is",K_d
    print "Rowplayer makes", total_Rowplayer_profit,"instead of",  comp_Rowplayer_profit
    print "Rowplayer surplus is",K_h
    print "Columnplayer makes",total_Columnplayer_profit,"instead of", comp_Columnplayer_profit
    print "Columnplayer surplus is",K_b
    print "Producer surplus is",K_p
    #Price distortions
    for k,v in gv.LMP.items():
        dist = (v - comp_LMPs[k])/comp_LMPs[k]
        print "At node",k,", the price is distorted by",round(dist,4)

    print payment_to_generators, "paid to generators for electricity."
    print payment_by_load, "paid by load for electricity."
    print congestion_rent, "total congestion rents."

