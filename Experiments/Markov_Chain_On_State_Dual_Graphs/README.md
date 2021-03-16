This directory contains a markov chain on the state dual graph of north carolina. This markov chain is defined in the file "north_carolina_analysis/chain_on_subset_of_faces.py". The markov chain currently makes a proposal at each step to Sierpinskify a percentage of the faces of the dual graph. Gerrychain is then run on this modified dual graph to determine the seat distribution of this map. 

Both north_carolina_analysis/facefinder.py and north_carolina_analysis/metamandering_north_carolina.py contain dependencies used in north_carolina_analysis/chain_on_subset_of_faces.py. 

north_carolina_analysis/watch_for_changes.py and north_carolina_analysis/north_carolina_visualization.py are both used to process the output of the chain. watch_for_changes can be configured to trigger an event once the chain of state dual graphs finds a new optimal seat total. north_carolina_visualization.py is used to visualize statistics about the chain over time. 
