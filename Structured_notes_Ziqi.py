#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 21:13:07 2020

@author: ziqiyuan
"""

import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import time

class Bond(object):
    
    def __init__(self):
        pass
    
    
    def BondPriceByYTM(self, par, T, ytm, coup, freq=2):
        """
        par: face Value
        T: maturity
        ytm: yield to maturity
        coup: coupon relative number
        """
        freq = float(freq)
        periods = T * freq
        coupon = coup / 100.0 * par / freq
        dt = [(i+1) / freq for i in range(int(periods))]
        price = sum([coupon / (1.+ytm/freq)**(freq*t) for t in dt]) + \
                par / (1.+ytm/freq)**(freq*T)
        return price
    
    def BondPriceBySpotInterest(self, par, T, Rs, coup, freq=2):
        """
        par: face Value
        T: maturity
        Rs: series of discout rate (yearly)
        coup: coupon relative number
        freq: coupon yearly times
        """
        periods = T * freq
        if int(periods) != len(Rs):
            raise Exception("R series should have the same length as periods!")
        
        freq = float(freq)
        #periods = T * freq
        coupon = coup / 100.0 * par / freq
        dt = [(i+1.) / freq for i in range(int(periods))]
        Price = 0
        for i in range(int(periods)):
            Price += coupon / ((1 + Rs[i]) ** dt[i])
        Price = par / (1. + Rs[-1]) ** dt[-1]   # needs to be thought 
        return Price


    
class Option(object):

    def __init__(self):
        pass
        
    def euro_vanilla_dividend(self, S, K, T, r, q, sigma, option='call'):
        """
        S: spot price
        K: strike price
        T: time to maturity
        r: interest rate
        q: rate of continuous dividend paying asset
        sigma: volatility of underlying asset
        """
        if not (option == 'call' or option == 'put'):
            raise Exception("Please enter call or put!")
            
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / \
             (sigma * np.sqrt(T))
        d2 = (np.log(S / K) + (r - q - 0.5 * sigma ** 2) * T) / \
             (sigma * np.sqrt(T))
             
        if option == 'call':
            result = (S * np.exp(-q * T) * norm.cdf(d1, 0.0, 1.0) - \
                      K * np.exp(-r * T) * norm.cdf(d2, 0.0, 1.0))
        if option == 'put':
            result = (K * np.exp(-r * T) * norm.cdf(-d2, 0.0, 1.0) - \
                      S * np.exp(-q * T) * norm.cdf(-d1, 0.0, 1.0))
            
        return result
        
    
class StructuredNotes(object):
    
    def __init__(self, par, maturity):
        self.maturity = maturity
        self.par = par
        self.Option = Option()
        self.Bond = Bond()
        self.OptionPrice = None
        self.BondPrice = None
        pass
    
    def EmbedBond(self, targetFundCost):
        ZCB_price = self.Bond.BondPriceByYTM(self.par, self.maturity, \
                                             targetFundCost, 0)
        self.BondPrice = ZCB_price
        pass
            
    def EmbedOption(self, S, K, r, q, sigma, option):
        #T = self.maturity
        Option_price = self.Option.euro_vanilla_dividend(S, K, self.maturity, \
                                                         r, q, sigma, option)
        self.OptionPrice = Option_price
        pass
    
    def OptionPayoff(self, ST, K):
        return max(ST-K, 0)
    
    def Payoff(self, ST, K):
        
        if self.OptionPrice is None or self.BondPrice is None:
            raise Exception("Calculate Bond price and option price first!")
            
        if self.BondPrice >= self.par:
            raise Exception("ZCB price is higher than par!")
            
        optionPart = self.par - self.BondPrice
        optionNum = optionPart / self.OptionPrice
        return optionNum * self.OptionPayoff(ST, K)
        

tradNum = 252 # we assume there are 252 trading days in a year
##T = 2.0
## Frequence of All simulation is day

def StockPriceSim(S0, sigma, r, T):
    '''
    Assume Stock Price follows GBM, with r and constant 
    volatility sigma (all measured in years)
    S0: inital value of stock price path
    T: measure in years
    ret: np.array
    '''
    N = int(T*tradNum)
    path = np.zeros((N))
    path[0] = S0
    noise = np.random.normal(0, 1, N-1)*sigma/np.sqrt(tradNum)
    for i in range(1, N):
        path[i] = path[i-1]*np.exp(r/tradNum-0.5*sigma**2/tradNum + noise[i-1]) 
    return path


def PathValue(Spath, K, r): ## need to be modified for each structured Note 
    '''
    Calculate option value on each stock path
    input: Spath: array like stock price
    K: strike price
    r: discounted rate measure yearly
    '''
    ST = Spath[-1]
    path_value = StructNotes.Payoff(ST, K) * np.exp(-r / tradNum * ((len(Spath))*(1.)))
    return path_value
     

def MonteCarlo(epochs, K, S0, r, sigma, T, plot=False):
    '''
    K: strike price
    S0: initial value
    r: interest rate measured yearly
    sigma: yearly volatility
    T: yrs
    '''
    path_value = []
   # pathList = []
    for epoch in range(epochs):
        path = StockPriceSim(S0, sigma, r, T)
        #pathList.append(path)
        path_value.append(PathValue(path, K, r))
        if plot:
            plt.plot(path)
    
    return np.mean(path_value) 


if __name__ == '__main__':
    time0 = time.time()
    def func(epochs, S0, K, sigma, option, r, T, targetCostFund):
        StructNotes = StructuredNotes(100, T)
        StructNotes.EmbedBond(targetCostFund)
        StructNotes.EmbedOption(S0, K, r, 0.0, sigma, option)
        value = MonteCarlo(epochs, K, S0, r, sigma, T)
        return value
    
    targetCostFundList = [0.01 + i/200.0 for i in list(range(1, 10))]
    epochs = 10000
    S0 = 1000.0
    sigma = 0.2
    option = 'call'
    r = 0.02
    T = 1.0
    Klist = [i + 1000 for i in list(range(1, 20))]
    Values = []
    for k in Klist:
        for target in targetCostFundList:
            value = func(epochs, S0, k, sigma, option, r, T, target)
            Values.append(value)
            print(str(k)+", "+str(target) + "is done!")
    print(time.time()-time0)
    '''
    strike = []
    for i in range(19):
        strike += [Klist[i]] * 9 
        
    target = targetCostFundList * 19
    df = pd.DataFrame({'StrikePrice': strike, 'targetCost': target, 'Payoff': Values})
    
    
    dd = df[df['StrikePrice']== 1008]
    dd = dd.set_index('targetCost')
    dd['Price'].plot()
    
    dd1 = df[df['targetCost'] == 0.035]
    dd1 = dd1.set_index('StrikePrice')
    dd1['Price'].plot()
        
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(df['StrikePrice'], df['targetCost'], df['Payoff'], cmap=plt.cm.viridis, linewidth=0.2)
    #plt.show()
    ax.view_init(30, 45)
    plt.show()
    
    '''



















