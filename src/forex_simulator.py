#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.finance as fin
import matplotlib.dates as mdates

class ForexSimulator(object):
    """ Simulator for forex market. """
    def __init__(self, csv_path):
        self.csv = csv_path
        self.dates = []
        self.opens = []
        self.highs = []
        self.lows = []
        self.closes = []
        self.fx = []
        self.revalage = 10
        
    def readData(self):
        with open(self.csv, encoding='shift-jis') as f:
            filename = self.csv.split('/')[-1]
            reader = csv.reader(f)
            next(reader)
            for i, row in enumerate(reader):
                date, opens, highs, lows, closes = row
                self.dates.append(date)
                self.opens.append(float(opens))
                self.highs.append(float(highs))
                self.lows.append(float(lows))
                self.closes.append(float(closes))
                self.fx.append([float(opens), float(highs), float(lows), float(closes)])

                
    def showCandles(self, span=[]):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        fin.candlestick2_ochl(ax, self.opens, self.closes, self.highs, self.lows, width=1)
        ax.set_xticks([i for i in range(len(self.dates))])
        ax.set_xlim([0, len(self.dates)])
        ax.set_xticklabels(["{}".format(i) for i in self.dates])
        fig.autofmt_xdate()
        plt.show()
        
    def run(self):
        self.readData()
        self.showCandles()
        

    

if __name__ == '__main__':
    sim = ForexSimulator('USDJPY.csv')
    sim.run()
