import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time

class Plotter:
    def __init__(self):
        self.losses = []
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], label="Loss")
        self.ax.set_xlabel("Iterations")
        self.ax.set_ylabel("Loss")
        self.ax.legend()

    def update_plot(self, loss):
        self.losses.append(loss)
        self.line.set_xdata(range(len(self.losses)))
        self.line.set_ydata(self.losses)
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()