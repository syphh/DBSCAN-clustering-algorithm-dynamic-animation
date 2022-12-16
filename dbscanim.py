import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
from matplotlib.widgets import Slider, Button
from collections import defaultdict
from queue import Queue


class Dbscan:
    def __init__(self, points, min_nei, epsilon):
        self.points = list(map(tuple, points))
        self.min_nei = min_nei
        self.epsilon = epsilon
        self.neighbors_map = None
        self.is_core = None
        self.c = 0
        self.clusters = [0] * len(self.points)
        self.unvisited_core = None
        self.idx_map = {self.points[i]: i for i in range(len(self.points))}
        self.queue = Queue()

    @staticmethod
    def euclidean(x0, y0, x1, y1):
        return np.sqrt(np.power(x0 - x1, 2) + np.power(y0 - y1, 2))

    def find_neighbors(self):
        neighbors_map = defaultdict(lambda: [])  # maps a point to its neighbors
        rounded_map = defaultdict(lambda: defaultdict(lambda: []))
        # maps an integer i to a map that maps an integer j to points between (i, j) and (i+1, j+1)
        for x, y in self.points:  # partitioning points into buckets
            rounded_map[int(x)][int(y)].append((x, y))
        for x, y in self.points:  # finding neighbors of a point by traversing buckets that contain potential neighbors
            for i in range(int(np.floor(x-self.epsilon)), int(np.ceil(x+self.epsilon))+1):
                for j in range(int(np.floor(y-self.epsilon)), int(np.ceil(y+self.epsilon))+1):
                    for nx, ny in rounded_map[i][j]:
                        if (x, y) != (nx, ny) and self.euclidean(x, y, nx, ny) <= self.epsilon:
                            neighbors_map[(x, y)].append((nx, ny))
        return neighbors_map

    def iterate(self):  # performs one process iteration to make the next frame
        if self.neighbors_map is None:  # needs to be done only once, at the beginning
            self.neighbors_map = self.find_neighbors()
            self.is_core = {point: len(self.neighbors_map[point]) >= self.min_nei for point in self.points}
            self.unvisited_core = set([point for point in self.points if self.is_core[point]])
        point = None
        if self.queue.empty():  # empty queue => we start making a new cluster
            if self.unvisited_core:
                self.c += 1
                point = next(iter(self.unvisited_core))  # extracting a random unvisited core point
                self.unvisited_core.remove(point)
                self.clusters[self.idx_map[point]] = self.c
                self.queue.put(point)
        else:  # non-empty queue => we continue making the current cluster
            point = self.queue.get()
            for nx, ny in self.neighbors_map[point]:
                self.clusters[self.idx_map[(nx, ny)]] = self.c
                if self.is_core[(nx, ny)] and (nx, ny) in self.unvisited_core:
                    self.queue.put((nx, ny))
                    self.unvisited_core.remove((nx, ny))
        return point


class AnimatedDbscan:
    def __init__(self, datasets):
        self.datasets = datasets
        self.play = False
        self.points = self.datasets[0]
        self.min_nei = 3
        self.epsilon = 1
        self.db = Dbscan(self.points, self.min_nei, self.epsilon)
        self.fig = plt.figure()
        self.ax = self.fig.add_axes([0.2, 0.1, 0.8, 0.8])
        self.ax.set_xlim([0, 15])
        self.ax.set_ylim([0, 15])
        self.ax.set_aspect('equal', adjustable='box')
        self.ax_epsilon = self.fig.add_axes([0.15, 0.1, 0.15, 0.03])
        self.ax_min_nei = self.fig.add_axes([0.15, 0.05, 0.15, 0.03])
        self.ax_start = self.fig.add_axes([0.81, 0.05, 0.1, 0.075])
        self.btn = Button(self.ax_start, 'start')
        self.btn.on_clicked(self.start)
        self.scat = None
        self.ax_datasets = []
        for i in range(3):
            for j in range(2):
                dataset = self.datasets[i*2+j]
                new_ax = self.fig.add_axes([0.07+0.14*j, 0.7-0.26*i, 0.12, 0.24])
                new_ax.get_xaxis().set_visible(False)
                new_ax.get_yaxis().set_visible(False)
                new_ax.scatter(dataset[:, 0], dataset[:, 1], s=2, c='black')
                self.ax_datasets.append(new_ax)
        self.min_nei_slider = Slider(
            ax=self.ax_min_nei,
            label='Min # of neighbors',
            valmin=1,
            valmax=10,
            valinit=self.min_nei,
            valfmt='%0.0f'
        )
        self.min_nei_slider.on_changed(self.update_slider)
        self.epsilon_slider = Slider(
            ax=self.ax_epsilon,
            label='Epsilon',
            valmin=0.1,
            valmax=3,
            valinit=self.epsilon,
            valfmt='%.1f'
        )
        self.epsilon_slider.on_changed(self.update_slider)
        self.fig.canvas.mpl_connect('button_press_event', self.change_dataset)
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=1,
                                           init_func=self.setup_plot, blit=True)

    def update_slider(self, val):
        self.min_nei = round(self.min_nei_slider.val)
        self.epsilon = round(self.epsilon_slider.val, 1)
        self.play = False
        self.db = Dbscan(self.points, self.min_nei, self.epsilon)

    def start(self, event):
        self.play = True

    def setup_plot(self):
        self.scat = self.ax.scatter(self.points[:, 0], self.points[:, 1], c='black', s=15, cmap='hsv')
        return self.scat,

    def update(self, i):  # updates plot for the next frame
        point = self.db.iterate() if self.play else None
        colors = ['black', 'blue', 'red', 'green', 'orange', 'aqua', 'purple', 'pink', 'royalblue', 'yellowgreen', 'salmon', 'olive', 'magenta', *mcolors.XKCD_COLORS]
        self.ax.clear()
        self.ax.set_xlim([0, 15])
        self.ax.set_ylim([0, 15])
        self.ax.set_aspect('equal', adjustable='box')
        self.scat = self.ax.scatter(self.points[:, 0], self.points[:, 1], c=[colors[i] for i in self.db.clusters], s=15)
        if point:
            circle = plt.Circle(point, self.epsilon, color=colors[self.db.clusters[self.db.idx_map[point]]], alpha=0.3)
            self.ax.add_patch(circle)
            emphasized = self.ax.scatter(*point, c=colors[self.db.clusters[self.db.idx_map[point]]], edgecolors='black')
            return self.scat, circle, emphasized
        return self.scat,

    def change_dataset(self, event):
        for i in range(len(self.ax_datasets)):
            if event.inaxes == self.ax_datasets[i]:
                self.points = self.datasets[i]
                self.play = False
                self.db = Dbscan(self.points, self.min_nei, self.epsilon)
                break


if __name__ == '__main__':
    datasets = []
    for i in range(1, 7):  # loading datasets
        df = pd.read_csv(f'datasets/dataset_clustering_{i}.csv')
        points = df.to_numpy()
        datasets.append(points)
    animation = AnimatedDbscan(datasets)
    try:
        wm = plt.get_current_fig_manager()
        wm.window.state('zoomed')
    except Exception as e:
        pass
    plt.show()
