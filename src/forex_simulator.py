#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.finance as fin
import matplotlib.dates as mdates

class ForexSimulator(object):
    """ Simulator for forex market. """
    def __init__(self, csv_path, spread=3):
        self.csv = csv_path
        self.dates = []
        self.opens = []
        self.highs = []
        self.lows = []
        self.closes = []
        self.fx = []
        self.spread = spread
        self.window = 0
        self.frame = 0
        self.fdates = []
        
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


    def makeFrameD(self, frame=30):
        self.fopens = []
        self.fhighs = []
        self.flows = []
        self.fcloses = []
        self.ffx = []
        for i in range(0, len(self.opens), frame):
            self.fopens.append(self.opens[i])
            self.fcloses.append(self.closes[i+frame-1])
            self.fhighs.append(max(self.highs[i: i+frame-1]))
            self.flows.append(min(self.lows[i: i+frame-1]))
            
    def showCandles(self, span, opens, closes, highs, lows):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.ticklabel_format(useOffset=False)
        fin.candlestick2_ochl(ax, opens, closes, highs, lows, width=1)
        #ax.set_xticks([i for i in range(len(self.dates))])
        if span:
            ax.set_xlim([span[0], span[1]])
        #ax.set_xticklabels(["{}".format(i) for i in self.dates])
        fig.autofmt_xdate()

        plt.show()
        
        
    def run(self):
        self.readData()
        self.makeFrameD(frame=15)
        self.showCandles([], self.fopens, self.fcloses, self.fhighs, self.flows)
        
        
if __name__ == '__main__':
    sim = ForexSimulator('data/csv/200701/USDJPY_20070102.csv')
    sim.run()
