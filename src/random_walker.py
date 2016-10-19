#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import random

import numpy as np
import scipy as sc
import scipy.optimize
import scipy.spatial.distance as sdist
import matplotlib.pyplot as plt
import matplotlib.finance as fin
import matplotlib.dates as mdates
from sklearn.cluster import KMeans

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
        self.sigma = 0.498

    def plotHist(self, ax):
        oc = self.opens - self.closes
        self.y, self.x, _ = ax.hist(oc, normed=True, bins=100, alpha=0.0)
        under = np.array(oc)

        # big black real body
        p = np.where(self.x[:-1] > self.sigma)
        ax.bar(self.x[p], self.y[p], width=8/100, color='k')

        # big red real body
        p = np.where(self.x[:-1] < -self.sigma)
        ax.bar(self.x[p], self.y[p], width=8/100, color='r')

        # black spinning top 
        p = np.where(np.logical_and(self.x[:-1] > -self.sigma, self.x[:-1] < 0))
        ax.bar(self.x[p], self.y[p], width=8/100, color='r', alpha=0.5)

        # red spinning top
        p = np.where(np.logical_and(self.x[:-1] < self.sigma, self.x[:-1] > 0))
        ax.bar(self.x[p], self.y[p], width=8/100, color='k', alpha=0.5)
        
        
        ax.set_xlim([-4.0, 4.0])
        
    def plotNorm(self, ax):
        oc = self.opens - self.closes
        x = np.arange(-4, 4, 0.1)
        ave = np.average(oc)
        var = np.var(oc)
        nd = self.normalD(x, ave, var)                
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
        np.random.seed(0)
        self.ups = []
        self.downs = []
        self.dummies = []
        width = 9
        dh = 3
        counts = 0
        fig = plt.figure(figsize=(8, 6), dpi=260)
        ax = fig.add_subplot(111)
        cmap = np.zeros((5, 10))
        for i in range(width, len(self.closes)-1):
            if abs(self.closes[i-width] - self.closes[i]) > dh:
                if (self.closes[i-width] - self.closes[i]) < 0:
                    self.classifyCandlesPlot(ax, [i-width, i], 'down')
                    #self.showCandles([i-width, i])
                    #cmap += self.classifyCandlesCmap(ax, [i-width, i+1], 'down')
                else:
                    #cmap += self.classifyCandles(ax, [i-width, i+1], .'up')
                    self.classifyCandlesPlot(ax, [i-width, i], 'up')
                    y = np.array(self.closes[i-width: i] ) - self.closes[i-width]
                counts += 1
            else:
                if( np.random.rand() > 0.9 ):
                    self.dummies.append(self.makeCandlesParttern([i-width, i]))
        return counts            
        '''
        #For matshow
        mt = ax.matshow(cmap, origin='lower')
        plt.colorbar(mt)
        '''


    def makeCandlesParttern(self, span):
        s, e = span
        cclass = np.zeros((e-s))
        diff = np.array(self.closes[s: e]) - np.array(self.opens[s: e])
        for i, idiff in enumerate(diff):
            if idiff < - self.sigma:
                cclass[i] = -2
            elif idiff > -self.sigma and idiff < 0:
                cclass[i] = -1
            elif idiff == 0:
                cclass[i] = 0
            elif idiff < self.sigma and idiff > 0:
                cclass[i] = 1
            elif idiff > self.sigma:
                cclass[i] = 2
        return cclass

                        
    def classifyCandlesPlot(self, ax, span=[], updown='up'):
        cclass = self.makeCandlesParttern(span)
        if updown == 'down':
            self.downs.append(cclass)
        if updown == 'up':
            self.ups.append(cclass)
        
    
    def classifyCandlesCmap(self, ax, span=[], updown='up'):
        if updown != 'up':
            return np.zeros((5, 10))
        s, e = span
        diff = np.array(self.opens[s: e]) - np.array(self.closes[s: e])
        cmap = np.zeros((5, 10))
        for i, idiff in enumerate(diff):
            if idiff < - self.sigma:
                cmap[0][i] += 1
            elif idiff > -self.sigma and idiff < 0:
                cmap[1][i] += 1
            elif idiff == 0:
                cmap[2][i] += 1
            elif idiff < self.sigma and idiff > 0:
                cmap[3][i] += 1
            elif idiff > self.sigma:
                cmap[4][i] += 1
        return cmap    
        #ax.set_xticks([i for i in range(e-s)])
    
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

    def showNetwork(self):
        plt.clf()
        import networkx as nx

        candles = np.array(self.ups + self.downs ) * 10 #+ self.dummies)       
        teaher = [1 for i in range(len(self.ups))] + \
                 [0 for i in range(len(self.downs))]
        #+[10 for i in range(len(self.dummies))]
        labels = ['u' for i in range(len(self.ups))] + \
                 ['d' for i in range(len(self.downs))]
        #+ ['0' for i in range(len(self.dummies))]
        
        M = sdist.cdist(candles, candles)
        G = nx.from_numpy_matrix(M)
        pos = nx.spring_layout(G)
        #G = nx.relabel_nodes(G, dict(zip(range(len(G.nodes())), labels)))    
        #G = nx.to_agraph(G)

        #G.node_attr.update(color="red", style="filled")
        #G.edge_attr.update(color="blue", width="2.0")
        nx.draw(G, pos=pos)
        plt.show()
        

        
        
    def kmeans(self):
        candles = self.ups + self.downs + self.dummies
        teaher = [1 for i in range(len(self.ups))] + \
                 [0 for i in range(len(self.downs))] + \
                 [10 for i in range(len(self.dummies))]

        
        kmeans = KMeans(n_clusters=3, random_state=10).fit(candles)

        labels = kmeans.labels_
        ldict = {'1': np.zeros((3)),
                 '0': np.zeros((3)),
                 '10': np.zeros((3))}

        for label, t, candle in zip(labels, teaher, candles):
            print(label, t, candle)
            if str(t) == '1':
                if str(label) == '0':
                    print('1', str(label))
                    ldict[str(t)][0] += 1
                elif str(label) == '1':
                    ldict[str(t)][1] += 1
                elif str(label) == '2':
                    ldict[str(t)][2] += 1
            elif str(t) == '0':
                if str(label) == '0':
                    ldict[str(t)][0] += 1
                elif str(label) == '1':
                    ldict[str(t)][1] += 1
                elif str(label) == '2':
                    ldict[str(t)][2] += 1
            elif str(t) == '10':
                if str(label) == '0':
                    ldict[str(t)][0] += 1
                elif str(label) == '1':
                    ldict[str(t)][1] += 1
                elif str(label) == '2':
                    ldict[str(t)][2] += 1
        print(ldict)

    def svm(self):
        from sklearn.svm import LinearSVC, SVC
        num = 70
        learning_set = self.ups[:num] + self.downs[:num] + self.dummies[:num]
        label_set = [0 for i in range(len(self.ups[:num]))] + \
                    [1 for i in range(len(self.downs[:num]))] + \
                    [2 for i in range(len(self.dummies[:num]))]
        test_set = self.ups[num:] + self.downs[num:] + self.dummies[num:]
        answer_set = [0 for i in range(len(self.ups[num:]))] + \
                    [1 for i in range(len(self.downs[num:]))] + \
                    [2 for i in range(len(self.dummies[num:]))]

        estimator = SVC(C=1.0)
        estimator.fit(learning_set, label_set)

        prediction = estimator.predict(test_set)
        rate = np.zeros((3))
        sums = np.array([len(self.ups[num:]), len(self.downs[num:]), len(self.dummies[num:])])
        for p, ans in zip(prediction, answer_set):
            if p == ans:
                rate[ans] += 1
        
        print(rate/sums)
        print(rate, sums)
        print(len(self.ups), len(self.downs), len(self.dummies))
    def main(self):
        #fig = plt.figure(figsize=(8,6), dpi=100)
        #fig = plt.figure(figsize=(4,3), dpi=260)
        #ax = fig.add_subplot(111)
        #self.plotHist(ax)
        #self.plotNorm(ax)
        #self.fitting(ax)
        #plt.cla()
        #ax = fig.add_subplot(111)
        #self.walk(ax, 0, self.f_var)
        #num = self.window(ax, self.closes)
        print(self.window_candles()/2448.0)
        #self.showNetwork()
        #self.kmeans()
        self.svm()
        #plt.savefig('bigsmall_hist.png')
        #plt.show()
        
        
if __name__ == '__main__':
    walker = Walker()
    walker.main()

    
