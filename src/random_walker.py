#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import random

import numpy as np
import scipy as sc
import scipy.optimize
import matplotlib.pyplot as plt
import matplotlib.finance as fin
import matplotlib.dates as mdates

from forex_simulator import ForexSimulator


class Walker(object):
    def __init__(self):
        self.csv = 'USDJPY.csv'
        self.fxs = ForexSimulator(self.csv)
        self.fxs.readData()
        self.dates = self.fxs.dates
        self.opens = np.array(self.fxs.opens)
        self.highs = np.array(self.fxs.highs)
        self.closes = np.array(self.fxs.closes)
        self.lows = np.array(self.fxs.lows)        

    def plotHist(self, ax):
        oc = self.opens - self.closes
        self.y, self.x, _ = ax.hist(oc, normed=True, bins=100)
        ax.set_xlim([-4, 4])

    def plotNorm(self, ax):
        oc = self.opens - self.closes
        x = np.arange(-4, 4, 0.1)
        ave = np.average(oc)
        var = np.var(oc)
        nd = self.normalD(x, ave, var)                
        nd2 = self.normalD(x, ave, var**2)
        ax.plot(x, nd, lw=3, c='r')
        #ax.plot(x, nd2)

    def normalD(self, x, ave, var):
        return 1/np.sqrt(2*np.pi*var)*np.exp(-(x-ave)**2/(2*var))        

    def normal2D(self, x, ave, var):
        return 1/np.sqrt(2*np.pi*var)*np.exp(-(x-ave)**4/(2*var))        

    def func(self, x, a):
        return 1/np.sqrt(2*np.pi*a)*np.exp(-x**2/(2*a))

    def residual_f(self, param, x, y):
        residual = y - self.func(x, param[0])
        return residual

    def fitting(self, ax):
        pinit = [1]
        result = scipy.optimize.leastsq(self.residual_f, pinit,  args=(self.x[:-1], self.y))
        opt = result[0]
        self.f_var = opt

        x = np.arange(-4, 4, 0.1)
        nd = self.normalD(x, 0, self.f_var)                
        ax.plot(x, nd, lw=3, c='y')

    def walk(self, ax, seed, sigma):
        steps = len(self.closes)
        ivalue = self.opens[0]
        random.seed(10)
        counts = 0
        total = 100
        for i in range(total):
            rnormal = [random.normalvariate(0, sigma) for x in range(steps)]
            walks = self.recursive(ivalue, rnormal)
            ax.plot(walks, alpha=0.3 , c='b')
            counts += self.window(ax, walks)
            print(i, counts)
        print(counts/total)
        print(counts)
        ax.set_ylim([70, 130])
        #ax.plot(self.closes, c='r')
        
    def recursive(self, init, xlist):
        a = np.zeros(len(xlist))
        a[0] = init + xlist[0]
        for i in range(1, len(xlist)):
            a[i] = xlist[i] + a[i-1]
        return a

    def window(self, ax, data):
        width = 9
        dh = 1.0
        counts = 0
        #ax.plot(data)
        for i in range(width, len(data)-1):
            if abs(data[i-width] - data[i]) > dh:
                x = range(i-width, i)
                counts += 1
                ax.plot(x, data[i-width: i], c='r')
        return counts

    def window_candles(self):
        width = 9
        dh = 3
        counts = 0
        fig = plt.figure(figsize=(8, 6), dpi=260)
        ax = fig.add_subplot(111)
        
        for i in range(width, len(self.closes)-1):
            if abs(self.closes[i-width] - self.closes[i]) > dh:
                self.showCandlesDup(ax, [i-width, i+1])
                counts += 1
        plt.show()
        return counts

    def showCandlesDup(self, ax, span=[]):
        s, e = span
        base = self.closes[s]
        opens = np.array(self.opens[s: e]) - base
        closes = np.array(self.closes[s: e]) - base
        highs = np.array(self.highs[s: e]) - base
        lows = np.array(self.lows[s: e]) - base
        try:
            #fin.candlestick2_ochl(ax, opens, closes, highs, lows, width=1, alpha=0.5)
            ax.plot(closes, c='b', alpha=0.3)
        except:
            print(opens)
            print(closes)
            print(lows)
            print(highs)            
            pass
        #ax.set_xticks([i for i in range(e-s)])
        ax.set_xlim([-0.5, e-s-0.5])
        #plt.show()
    
    def showCandles(self, span=[]):
        fig = plt.figure(figsize=(8, 6), dpi=260)
        s, e = span
        ax = fig.add_subplot(111)
        fin.candlestick2_ochl(ax, self.opens[s: e], self.closes[s: e], self.highs[s: e], self.lows[s: e], width=1)
        ax.set_xticks([i for i in range(e-s)])
        ax.set_xlim([-0.5, e-s-0.5])
        ax.set_xticklabels(["{}".format(i) for i in self.dates[s: e]])
        png = ''.join(self.dates[s: e][0].split('/'))+'.png'
        fig.autofmt_xdate()
        plt.savefig(png)
        print(png)
        #plt.show()
                
    def main(self):
        fig = plt.figure(figsize=(8,6), dpi=100)
        #fig = plt.figure(figsize=(4,3), dpi=260)
        ax = fig.add_subplot(111)
        self.plotHist(ax)
        self.plotNorm(ax)
        self.fitting(ax)
        plt.cla()
        #ax = fig.add_subplot(111)
        self.walk(ax, 0, self.f_var)
        #num = self.window(ax, self.closes)
        #print(self.window_candles()/2448.0)
        plt.savefig('window2.png')
        #plt.show()
        
        
if __name__ == '__main__':
    walker = Walker()
    walker.main()

    
