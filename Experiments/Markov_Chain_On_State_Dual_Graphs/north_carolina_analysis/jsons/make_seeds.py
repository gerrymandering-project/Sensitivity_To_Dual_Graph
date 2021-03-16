# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 13:25:07 2020

@author: daryl
"""

from gerrychain import Graph
from gerrychain.tree import recursive_tree_part
import geopandas as gpd

state_ids = ['WI','NC','GA','PA','CO']
state_names = ['./WI_wards_12_16/WI_ltsb_corrected_final.shp','./NC_VTD/NC_VTD.shp',
              './GA_precincts/GA_precincts16.shp','./PA_VTDs/PA_VTD_PLANS.shp',
              './CO_precincts/co_precincts.shp']
num_dists = [8, 13, 14, 18, 7]
popcols= ['PERSONS','PL10AA_TOT','TOTPOP','TOTPOP','TOTPOP']

for i in range(1,5):
    state_id = state_ids[i]
    state_name = state_names[i]
    num_dist = num_dists[i]
    popcol=popcols[i]

    
    graph = Graph.from_file(state_name,reproject=False,ignore_errors = True)
    
    print(f'finished building {state_id}')
    df = gpd.read_file(state_name)
    
    centroids = df.centroid
    c_x = centroids.x
    c_y = centroids.y
    
    totpop = 0
    for node in graph.nodes():
        totpop += int(graph.node[node][popcol])
        graph.node[node]["population"] = int(graph.node[node][popcol]) 
        graph.node[node]['C_X'] = c_x[node]
        graph.node[node]['C_Y'] = c_y[node]
     
    if state_id  =='WI':
        graph.remove_nodes_from([1395,63])
        
    graph.to_json(f'./jsons/{state_id}_initial.json')
       
    cddict = recursive_tree_part(graph,range(num_dist),totpop/num_dist,"population", .01,1)
    
    for node in graph.nodes():
        graph.node[node]['part'] = cddict[node]
        
    graph.to_json(f'./jsons/{state_id}_seed.json')
    print(f'saved {state_id}')


