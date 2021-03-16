import matplotlib as plt
import pickle
import statistics
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('agg')
def main(): 
    with open('obj/North Carolina100cs,50.pkl', 'rb') as f:
        data = pickle.load(f)
    rep_means = []
    dem_means = []
    for step in data["rep_seat_data"]:
        rep_means.append(statistics.mean(step))
    for step in data["dem_seat_data"]:
        dem_means.append(statistics.mean(step))
    
    plt.plot(range(1,len(rep_means)+1), rep_means)
    plt.xlabel("Chain Step")
    plt.ylabel("rep_seat_data")
    plt.savefig("plots/rep_seats")
    fig = plt.figure(1, figsize=(9, 6))
    ax = fig.add_subplot(111)
    bp = ax.boxplot(data["rep_seat_data"])
    fig.savefig('fig1.png', bbox_inches='tight')


main()