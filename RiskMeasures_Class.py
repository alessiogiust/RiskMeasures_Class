import pandas as pd
import numpy as np
import scipy.stats as sp
import statsmodels.api as sm
from datetime import datetime 


class RiskMeasures():
    """
    Compute VaR, ES and other risk measures for financial returns time series using different methods
    """
    
    def __init__(self, returns, alphas, h=1, position=1.0, weights=None, portfolio=False):
        """
        - returns = pandas Dataframe, daily returns for 1 or more assets for which we want to estimate the VaR (arithmetic)
        - alphas = float (list of floats), significance levels for the VaR
        - h = int, to obtain the VaR at h days
        - position = float, amount of capital invested exposed to risk
        - weights = list, weights for the portfolio
        - portfolio = bool, if False use every return series in the df, otherwise combine to get portfolio returns series using the weights defined earlier
        """
        self.returns = returns
        self.alphas = alphas
        self.h = h
        self.position = position
        self.weights = weights
        self.portfolio = portfolio
        if self.portfolio == False:
            self.r = self.returns
            self.tickers = list(self.r.columns)
        else:
            # compute returns for the portfolio given weights
            weights = np.array(self.weights)
            rmat = np.matrix(self.returns)
            self.r = pd.DataFrame(np.fromiter(np.sum(np.multiply(rmat, weights), axis=1), dtype=float), index=self.returns.index, columns=["Portfolio"])
            self.tickers = list(self.r.columns)

    def get_portfolio_returns(self):
        """
        Obtain computed portfolio returns if 'portfolio' = True
        """
        if self.portfolio==True:
            return self.r
        else:
            return None

    def get_VaR_basic(self, dist="gaussian", df=5, rolling=False, start=60, win=252, new_w_incr=None):
        """
        Compute VaR based on GAUSSIAN, TSTUDENT, HISTORICAL sim.
        To obtain the VaR at h days you can just pass h-days returns as input or the mean and volatility will be transformed in h-days equivalents.
        - dist = string (options: gaussian, tstudent, hist), select the distribution for the VaR
        - df = int, degrees of freedom for the t-Student distribution
        - rolling = bool, if False return a single VaR value, if True return a Series
        - start = int, minimum number of datapoints to use in the rolling function, identifies the datapoint from which you will start to have VaR values
        - win = int, number of datapoints in the rolling window
        - new_w_incr = list of floats, new list of weights (!! used only in the PortfolioVaR_components class)
        OUTPUT: pandas DataFrame of float values (positive) for the VaR or pandas DataFrame with VaR values starting from the datapoint selected with 'start'.
        """
        
        if new_w_incr is None: # Normal usage of the method
            if rolling == False: 
                VARs = pd.DataFrame(index=self.tickers, columns=map(str, self.alphas))
                for tick in self.tickers:
                    mean = self.r[tick].mean()*self.h
                    std = self.r[tick].std()*np.sqrt(self.h)
                    for alpha in self.alphas:
                        quant = np.quantile(self.r[tick], alpha)
                        if dist == "gaussian": VARs.loc[tick, str(alpha)] = -self.position*(mean+sp.norm.ppf(alpha)*std)
                        elif dist == "tstudent": VARs.loc[tick, str(alpha)] =  -self.position*(mean+sp.t.ppf(alpha, df)*std)
                        elif dist == "hist": VARs.loc[tick, str(alpha)] = (-self.position*quant)*np.sqrt(self.h)
                        else: raise ValueError("Please use an accepted distribution type")
                return VARs
            else:
                data = self.r.copy()
                for tick in self.tickers:
                    # replaced for every tick
                    data["mean"] = data[tick].rolling(win, min_periods=start).mean()*self.h
                    data["std"] = data[tick].rolling(win, min_periods=start).std()*np.sqrt(self.h)
                    # compute rolling parameters
                    for alpha in self.alphas:
                        # replaced for every alpha and every tick
                        data["quant"] = data[tick].rolling(win, min_periods=start).quantile(alpha)
                        # use .shift(1) to use data up to t (today) to compute VaR associated to date t+1 (tomorrow)
                        if dist == "gaussian": data[f"{tick}_VaR_{alpha}"] = -self.position*(data["mean"].shift(1)+sp.norm.ppf(alpha)*data["std"].shift(1))
                        elif dist == "tstudent": data[f"{tick}_VaR_{alpha}"] = -self.position*(data["mean"].shift(1)+sp.t.ppf(alpha, df)*data["std"].shift(1))
                        elif dist == "hist": data[f"{tick}_VaR_{alpha}"] = (-self.position*data["quant"].shift(1))*np.sqrt(self.h) 
                        else: raise ValueError("Please use an accepted distribution type")
                col_todrop = [tick for tick in self.tickers]+["mean", "std", "quant"]
                return data.drop(columns=col_todrop)  # there are NaN values at the beginning based on start parameter
        # Specifi usage only for PortfolioVaR_componenets --> get_incremental_VaR
        else: 
            w = np.array(new_w_incr)
            rmat = np.matrix(self.returns)
            retpf = pd.DataFrame(np.fromiter(np.sum(np.multiply(rmat, w), axis=1), dtype=float), index=self.returns.index, columns=["Portfolio"])
            if rolling == False: 
                VARs = pd.DataFrame(index=retpf.columns, columns=map(str, self.alphas))
                for tick in retpf.columns:
                    mean = retpf[tick].mean()*self.h
                    std = retpf[tick].std()*np.sqrt(self.h)
                    for alpha in self.alphas:
                        quant = np.quantile(retpf[tick], alpha)
                        if dist == "gaussian": VARs.loc[tick, str(alpha)] = -self.position*(mean+sp.norm.ppf(alpha)*std)
                        elif dist == "tstudent": VARs.loc[tick, str(alpha)] =  -self.position*(mean+sp.t.ppf(alpha, df)*std)
                        elif dist == "hist": VARs.loc[tick, str(alpha)] = (-self.position*quant)*np.sqrt(self.h)
                        else: raise ValueError("Please use an accepted distribution type")
                return VARs
            else: raise ValueError("'rolling' option is not available")

    def get_ES_basic(self, dist="gaussian", df=5, rolling=False, start=60, win=252):
        """
        Compute Expected Shortfall based on GAUSSIAN, TSTUDENT, HISTORICAL simulation
        ARGUMENTS: same as get_VaR_basic()
        OUTPUT: pandas DataFrame of float values (positive) for the ES or pandas DataFrame with ES values starting from the datapoint selected with 'start'
        """
        if rolling == False: 
            ESs = pd.DataFrame(index=self.tickers, columns=map(str, self.alphas))
            for tick in self.tickers:
                mean = (self.r[tick].mean())*self.h
                std = (self.r[tick].std())*np.sqrt(self.h)
                for alpha in self.alphas:
                    quant = np.quantile(self.r[tick], alpha)
                    if dist == "gaussian": ESs.loc[tick, str(alpha)] = (alpha**-1 *sp.norm.pdf(sp.norm.ppf(alpha))*std - mean)*self.position
                    elif dist == "tstudent": ESs.loc[tick, str(alpha)] = (-1/alpha * (1-df)**(-1) * (df - 2 + sp.t.ppf(alpha, df)**2) * sp.t.pdf(sp.t.ppf(alpha, df), df)*std - mean)*self.position
                    elif dist == "hist": ESs.loc[tick, str(alpha)] = ((-(self.r[tick][self.r[tick]<quant].mean()))*np.sqrt(self.h))*self.position  # minus in front to get positive ES value
                    else: raise ValueError("Please use an accepted distribution type")
            return ESs
        else:
            data = self.r.copy()
            for tick in self.tickers:
                data["mean"] = (data[tick].rolling(win, min_periods=start).mean())*self.h
                data["std"] = (data[tick].rolling(win, min_periods=start).std())*np.sqrt(self.h)
                for alpha in self.alphas:
                    data["quant"] = data[tick].rolling(win, min_periods=start).quantile(alpha)
                    # use .shift(1) to use data up to t-1 (today) to compute VaR associated to date t (tomorrow)
                    if dist == "gaussian": data[f"{tick}_ES_{alpha}"] = (alpha**-1 * sp.norm.pdf(sp.norm.ppf(alpha))*data["std"].shift(1) - data["mean"].shift(1))*self.position
                    elif dist == "tstudent": data[f"{tick}_ES_{alpha}"] = (-1/alpha* (1-df)**(-1) * (df - 2 + sp.t.ppf(alpha, df)**2) * sp.t.pdf(sp.t.ppf(alpha, df), df) * data["std"].shift(1) - data["mean"].shift(1))*self.position
                    elif dist == "hist": 
                        es = np.zeros(len(self.r[tick]))
                        for i in range(start+1, len(self.r[tick])):
                            # until i<win the rolling window will be smaller than what implied by win parameter and it is expanding until it reaches the win size
                            if i<win:
                                ser = self.r[tick].iloc[:i-1]
                                quant = data["quant"].iloc[i-1]
                                es[i] = ser[ser < quant].mean()
                            else:
                                ser = self.r[tick].iloc[(i-win):(i-1)]
                                quant = data["quant"].iloc[i-1]
                                es[i] = ser[ser < quant].mean()
                        es[es==0] = np.nan   # put NaN instead of 0 to have an output that can be plotted immediately (there is no jump from 0 to the value but simply no values)
                        data[f"{tick}_ES_{alpha}"] = ((-es)*np.sqrt(self.h))*self.position  # minus to get positive values
                    else: raise ValueError("Please use an accepted distribution type")
            col_todrop = [tick for tick in self.tickers]+["mean", "std", "quant"]
            return data.drop(columns=col_todrop)  # there are NaN values at the beginning based on start

    def get_VaRportfolio_gaussian(self, new_w=None, dist="gaussian", df=5, rolling=False, start=60, win=252):
        """
        Compute portfolio VaR based on gaussian distribution, given the portfolio components and weights.
        ARGUMENTS: same as other methods except for:
        - new_w = list, weights for the portfolio, required only if you don't want to use the weights passed in the class instantiation
        !! 'dist' only "gaussian" or "tstudent", not "hist"
        OUTPUT: pandas DataFrame of float value (positive) for the VaR or pandas DataFrame with VaR values starting from the datapoint selected with 'start'
        """
        if self.portfolio == False:
            if new_w == None:
                ww = self.weights
            else:
                ww = new_w
            w = np.array(ww)
            if rolling==False:
                VARs = pd.DataFrame(index=["Portfolio"], columns=map(str, self.alphas))
                mean = self.h*(self.r.mean())
                cov = self.r.cov()  
                vol_pf = (np.dot(w.T, np.dot(cov, w))**(1/2))*np.sqrt(self.h)   # volatility of the portfolio computed using covariance matrix and weights
                for alpha in self.alphas:
                    if dist == "gaussian": 
                        VARs.loc["Portfolio", str(alpha)] = -self.position* (np.dot(w.T, mean) + sp.norm.ppf(alpha)*(vol_pf))
                    elif dist == "tstudent": 
                        VARs.loc["Portfolio", str(alpha)] = -self.position* (np.dot(w.T, mean) + sp.t.ppf(alpha, df)*(vol_pf))
                    else: raise ValueError("Please use an accepted distribution type")
                return VARs
            else:
                data = pd.DataFrame(index=self.r.index)
                for alpha in self.alphas:
                    # Estimate rolling parameters for each alpha
                    var = np.zeros(len(self.r))
                    for i in range(start+1, len(self.r)):
                        if i<win:
                            ser = self.r.iloc[:i-1, :]
                            mean = self.h*(ser.mean())
                            cov = np.sqrt(self.h)*(ser.cov())
                            vol_pf = (np.dot(w.T, np.dot(cov, w))**(1/2))*np.sqrt(self.h)
                            if dist == "gaussian": 
                                var[i] = -self.position*(np.dot(w.T, mean) + sp.norm.ppf(alpha)*(vol_pf))
                            elif dist == "tstudent": 
                                var[i] = -self.position*(np.dot(w.T, mean) + sp.t.ppf(alpha, df)*(vol_pf))
                            else: raise ValueError("Please use an accepted distribution type")
                        else:
                            ser = self.r.iloc[(i-win):(i-1), :]
                            mean = self.h*(ser.mean())
                            cov = np.sqrt(self.h)*(ser.cov())
                            vol_pf = (np.dot(w.T, np.dot(cov, w))**(1/2))*np.sqrt(self.h)
                            if dist == "gaussian": 
                                var[i] = -self.position* (np.dot(w.T, mean) + sp.norm.ppf(alpha)*(vol_pf))
                            elif dist == "tstudent": 
                                var[i] = -self.position* (np.dot(w.T, mean) + sp.t.ppf(alpha, df)*(vol_pf))
                            else: raise ValueError("Please use an accepted distribution type")
                    # Create a new column in the "data" df for each alpha, but data in lists var and es is replaced each time
                    var[var==0] = np.nan
                    data[f"VaR_{alpha}"] = var
                return data
        else:
            raise ValueError("This function doesn't work if the 'portfolio' argument is set to True")

    def get_ESportfolio_gaussian(self, new_w=None, dist="gaussian", df=5, rolling=False, start=60, win=252):
        """
        Compute portfolio ES based on gaussian distribution, given the portfolio components and weights.
        ARGUMENTS: same as other methods except for:
        - new_w = list, weights for the portfolio, required only if you don't want to use the weights passed in the class initialization
        !! 'dist' only "gaussian" or "tstudent", not "hist"
        OUTPUT: pandas DataFrame of float value (positive) for the ES or pandas DataFrame with ES values starting from the datapoint selected with the start argument.
        """
        if self.portfolio == False:
            if new_w == None:
                ww = self.weights
            else:
                ww = new_w
            w = np.array(ww)
            if rolling==False:
                ESs = pd.DataFrame(index=["Portfolio"], columns=map(str, self.alphas))
                mean = self.h*(self.r.mean())
                cov = self.r.cov()
                vol_pf = (np.dot(w.T, np.dot(cov, w))**(1/2))*np.sqrt(self.h)   # volatility of the portfolio computed using covariance matrix and weights
                for alpha in self.alphas:
                    if dist == "gaussian":
                        ESs.loc["Portfolio", str(alpha)] = self.position* (alpha**-1 * sp.norm.pdf(sp.norm.ppf(alpha))*vol_pf - np.dot(w.T, mean))
                    elif dist == "tstudent":
                        ESs.loc["Portfolio", str(alpha)] = self.position* (-1/alpha * (1-df)**(-1) * (df - 2 + sp.t.ppf(alpha, df)**2) * sp.t.pdf(sp.t.ppf(alpha, df), df)*vol_pf - np.dot(w.T, mean))
                    else: raise ValueError("Please use an accepted distribution type")
                return ESs
            else:
                data = pd.DataFrame(index=self.r.index)
                for alpha in self.alphas:
                    es = np.zeros(len(self.r))
                    for i in range(start+1, len(self.r)):
                        if i<win:
                            ser = self.r.iloc[:i-1, :]
                            mean = self.h*(ser.mean())
                            cov = np.sqrt(self.h)*(ser.cov())
                            vol_pf = (np.dot(w.T, np.dot(cov, w))**(1/2))*np.sqrt(self.h)
                            if dist == "gaussian":
                                es[i] = self.position* (alpha**-1 * sp.norm.pdf(sp.norm.ppf(alpha))*vol_pf - np.dot(w.T, mean))
                            elif dist == "tstudent":
                                es[i] = self.position* (-1/alpha * (1-df)**(-1) * (df - 2 + sp.t.ppf(alpha, df)**2) * sp.t.pdf(sp.t.ppf(alpha, df), df)*vol_pf - np.dot(w.T, mean))
                            else: raise ValueError("Please use an accepted distribution type")
                        else:
                            ser = self.r.iloc[(i-win):(i-1), :]
                            mean = self.h*(ser.mean())
                            cov = np.sqrt(self.h)*(ser.cov())
                            vol_pf = (np.dot(w.T, np.dot(cov, w))**(1/2))*np.sqrt(self.h)
                            if dist == "gaussian":
                                es[i] = self.position * (alpha**-1 * sp.norm.pdf(sp.norm.ppf(alpha))*vol_pf - np.dot(w.T, mean))
                            elif dist == "tstudent":
                                es[i] = self.position * (-1/alpha * (1-df)**(-1) * (df - 2 + sp.t.ppf(alpha, df)**2) * sp.t.pdf(sp.t.ppf(alpha, df), df)*vol_pf - np.dot(w.T, mean))
                            else: raise ValueError("Please use an accepted distribution type")
                    es[es==0] = np.nan
                    data[f"ES_{alpha}"] = es
                return data
        else:
            raise ValueError("This function doesn't work if the 'portfolio' argument is set to True")

    def get_VaRportfolio_fact(self, fact, new_w=None, dist="gaussian", df=5, rolling=False, start=60, win=252):
        """
        Compute VaR using mapping technique.
        ARGUMENTS: same as other methods expect for:
        - fact = pandas DataFrame, return series of the factors used to map the portfolio.
        OUTPUT: pandas dataframe with float value (positive) for factorial VaR or series of factorial VaR
        """
        if self.portfolio == False:
            if new_w == None:
                ww = self.weights
            else:
                ww = new_w
            w = np.array(ww)
            if rolling == False:
                VARs = pd.DataFrame(index=["Portfolio"], columns=map(str, self.alphas))
                mean = self.r.mean()*self.h
                mean_pf = np.dot(w.T, mean)  
                cov_fact = fact.cov()  
                beta = np.zeros((len(self.tickers), len(fact.columns)))  # prepare the matrix for the beta coefficients
                for i in range(len(self.tickers)):
                    # linear regression to get the beta for each asset related to each factor
                    y = self.r.iloc[:, i]
                    X = fact
                    X = sm.add_constant(X)
                    reg = sm.OLS(y, X)
                    results = reg.fit()
                    for j in range(len(fact.columns)):
                        beta[i, j] = results.params[j+1]  # betas for asset i given each factor j
                # compute volatility of the portfolio given factor covariance matrix and betas matrix
                vol_pf = ((np.dot(w.T, np.dot(beta, np.dot(cov_fact, np.dot(beta.T, w)))))**(1/2))*np.sqrt(self.h)
                for alpha in self.alphas:
                    if dist == "gaussian": 
                        VARs.loc["Portfolio", str(alpha)] = -self.position*(mean_pf + sp.norm.ppf(alpha)*vol_pf)
                    elif dist == "tstudent": 
                        VARs.loc["Portfolio", str(alpha)] = -self.position*(mean_pf + sp.t.ppf(alpha, df)*vol_pf)
                    else: raise ValueError("Please use an accepted distribution type")
                return VARs
            else:
                data = pd.DataFrame(index=self.r.index)
                for alpha in self.alphas:
                    VAR = np.zeros(len(self.r))
                    for k in range(start+1, len(self.r)):  # same process as before but with beta and covariance obtain using rolling window of data
                        if k<win:
                            r_ = self.r.iloc[:k-1, :]
                            fact_ = fact.iloc[:k-1, :]
                            mean = r_.mean()*self.h
                            mean_pf = np.dot(w.T, mean)
                            cov_fact = fact_.cov()
                            beta = np.zeros((len(self.tickers), len(fact.columns)))
                            for i in range(len(self.tickers)):
                                y = r_.iloc[:, i]
                                X = fact_
                                X = sm.add_constant(X)
                                reg = sm.OLS(y, X)
                                results = reg.fit()
                                for j in range(len(fact.columns)):
                                    beta[i, j] = results.params[j+1]
                            vol_pf = ((np.dot(w.T, np.dot(beta, np.dot(cov_fact, np.dot(beta.T, w)))))**(1/2))*np.sqrt(self.h)
                            if dist == "gaussian": 
                                VAR[k] = -self.position*(mean_pf + sp.norm.ppf(alpha)*vol_pf)
                            elif dist == "tstudent": 
                                VAR[k] = -self.position*(mean_pf + sp.t.ppf(alpha, df)*vol_pf)
                            else: raise ValueError("Please use an accepted distribution type")
                        else:
                            r_ = self.r.iloc[(k-win):(k-1), :]
                            fact_ = fact.iloc[(k-win):(k-1), :]
                            mean = r_.mean()*self.h
                            mean_pf = np.dot(w.T, mean)
                            cov_fact = fact_.cov()
                            beta = np.zeros((len(self.r.columns), len(fact.columns)))
                            for i in range(len(self.r.columns)):
                                y = r_.iloc[:, i]
                                X = fact_
                                X = sm.add_constant(X)
                                reg = sm.OLS(y, X)
                                results = reg.fit()
                                for j in range(len(fact.columns)):
                                    beta[i, j] = results.params[j+1]
                            vol_pf = ((np.dot(w.T, np.dot(beta, np.dot(cov_fact, np.dot(beta.T, w)))))**(1/2))*np.sqrt(self.h)
                            if dist == "gaussian": 
                                VAR[k] = -self.position*(mean_pf + sp.norm.ppf(alpha)*vol_pf)
                            elif dist == "tstudent": 
                                VAR[k] = -self.position*(mean_pf + sp.t.ppf(alpha, df)*vol_pf)
                            else: raise ValueError("Please use an accepted distribution type")
                    VAR[VAR==0] = np.nan
                    data[f"VaR_{alpha}"] = VAR
                return data
        else:
            raise ValueError("This function doesn't work if the 'portfolio' argument is set to True")

    def get_ESportfolio_fact(self, fact, new_w=None, dist="gaussian", df=5, rolling=False, start=60, win=252):
        """
        Compute ES using mapping technique.
        ARGUMENTS: same as other methods expect for:
        - fact = pandas DataFrame, return series of the factors used to map the portfolio.
        OUTPUT: pandas dataframe with float value (positive) for factorial ES or series of factorial ES
        """
        if self.portfolio == False:
            if new_w == None:
                ww = self.weights
            else:
                ww = new_w
            w = np.array(ww)
            if rolling == False:
                ESs = pd.DataFrame(index=["Portfolio"], columns=map(str, self.alphas))
                mean = self.r.mean()*self.h
                mean_pf = np.dot(w.T, mean)  
                cov_fact = fact.cov()  
                beta = np.zeros((len(self.tickers), len(fact.columns)))  # prepare the matrix for the beta coefficients
                for i in range(len(self.tickers)):
                    # linear regression to get the beta for each asset related to each factor
                    y = self.r.iloc[:, i]
                    X = fact
                    X = sm.add_constant(X)
                    reg = sm.OLS(y, X)
                    results = reg.fit()
                    for j in range(len(fact.columns)):
                        beta[i, j] = results.params[j+1]  # betas for asset i given each factor j
                # compute volatility of the portfolio given factor covariance matrix and betas matrix
                vol_pf = ((np.dot(w.T, np.dot(beta, np.dot(cov_fact, np.dot(beta.T, w)))))**(1/2))*np.sqrt(self.h)
                for alpha in self.alphas:
                    if dist == "gaussian": 
                        ESs.loc["Portfolio", str(alpha)] = self.position* (alpha**-1 * sp.norm.pdf(sp.norm.ppf(alpha))*vol_pf - mean_pf)
                    elif dist == "tstudent": 
                        ESs.loc["Portfolio", str(alpha)] = self.position* (-1/alpha * (1-df)**(-1) * (df - 2 + sp.t.ppf(alpha, df)**2) * sp.t.pdf(sp.t.ppf(alpha, df), df)*vol_pf - mean_pf)
                    else: raise ValueError("Please use an accepted distribution type")
                return ESs
            else:
                data = pd.DataFrame(index=self.r.index)
                for alpha in self.alphas:
                    ES = np.zeros(len(self.r))
                    for k in range(start+1, len(self.r)):  # same process as before but with beta and covariance obtain using rolling window of data
                        if k<win:
                            r_ = self.r.iloc[:k-1, :]
                            fact_ = fact.iloc[:k-1, :]
                            mean = r_.mean()*self.h
                            mean_pf = np.dot(w.T, mean)
                            cov_fact = fact_.cov()
                            beta = np.zeros((len(self.tickers), len(fact.columns)))
                            for i in range(len(self.tickers)):
                                y = r_.iloc[:, i]
                                X = fact_
                                X = sm.add_constant(X)
                                reg = sm.OLS(y, X)
                                results = reg.fit()
                                for j in range(len(fact.columns)):
                                    beta[i, j] = results.params[j+1]
                            vol_pf = ((np.dot(w.T, np.dot(beta, np.dot(cov_fact, np.dot(beta.T, w)))))**(1/2))*np.sqrt(self.h)
                            if dist == "gaussian": 
                                ES[k] = self.position* (alpha**-1 * sp.norm.pdf(sp.norm.ppf(alpha))*vol_pf - mean_pf)
                            elif dist == "tstudent": 
                                ES[k] = self.position* (-1/alpha * (1-df)**(-1) * (df - 2 + sp.t.ppf(alpha, df)**2) * sp.t.pdf(sp.t.ppf(alpha, df), df)*vol_pf - mean_pf)
                            else: raise ValueError("Please use an accepted distribution type")
                        else:
                            r_ = self.r.iloc[(k-win):(k-1), :]
                            fact_ = fact.iloc[(k-win):(k-1), :]
                            mean = r_.mean()*self.h
                            mean_pf = np.dot(w.T, mean)
                            cov_fact = fact_.cov()
                            beta = np.zeros((len(self.tickers), len(fact.columns)))
                            for i in range(len(self.tickers)):
                                y = r_.iloc[:, i]
                                X = fact_
                                X = sm.add_constant(X)
                                reg = sm.OLS(y, X)
                                results = reg.fit()
                                for j in range(len(fact.columns)):
                                    beta[i, j] = results.params[j+1]
                            vol_pf = ((np.dot(w.T, np.dot(beta, np.dot(cov_fact, np.dot(beta.T, w)))))**(1/2))*np.sqrt(self.h)
                            if dist == "gaussian": 
                                ES[k] = self.position* (alpha**-1 * sp.norm.pdf(sp.norm.ppf(alpha))*vol_pf - mean_pf)
                            elif dist == "tstudent": 
                                ES[k] = self.position* (-1/alpha * (1-df)**(-1) * (df - 2 + sp.t.ppf(alpha, df)**2) * sp.t.pdf(sp.t.ppf(alpha, df), df)*vol_pf - mean_pf)
                            else: raise ValueError("Please use an accepted distribution type")
                    ES[ES==0] = np.nan
                    data[f"ES_{alpha}"] = ES
                return data
        else:
            raise ValueError("This function doesn't work if the 'portfolio' argument is set to True")

    def VaR_validation(self, ValueatRisk, alpha, new_r, alpha_test=0.05, print_output=False):
        """
        Verify if the theoretical and realized number of exceedances against the VaR are statistically equal.
        Statistical test based on normal approximation to the binomial
        ARGUMENTS:
        - ValueatRisk = pandas Series, VaR values obtained from functions defined in this class (first n elememtns are NaN)
        - new_r = pandas Series, returns from which the VaR is computed
        - alpha = float, alpha level for the passed VaR series
        - alpha_test = float, significance level for validation test
        - print_output = bool, if True print results
        OUTPUT: float, zscores and pvalues for the statistical tests for each alpha
        """
        ValueatRisk = ValueatRisk.dropna()   # Drop non matching values (first values in VaR series are NaN)
        df = pd.merge(new_r, -ValueatRisk, right_index=True, left_index=True)    # VaR with minus becasue we need it has a loss here
        T = len(df)
        theo_exc = T*alpha
        real_exc = len(df.iloc[:, 1][df.iloc[:, 0]<df.iloc[:, 1]]) # lenght of the df that contains only datapoints where returns are below VaR level
        zscore = (real_exc-theo_exc)/(np.sqrt(alpha*(1-alpha)*T))
        pval = sp.norm.cdf(-abs(zscore))
        pval = pval*2
        if print_output == True:
            print("---------------------------------------")
            print(f"----- VaR alpha equal to {alpha} -----")
            print(f"Theoretical exceedances: {theo_exc}")
            print(f"Realized exceedances: {real_exc}")
            print("--> Zscore = ", zscore.round(4))
            print("--> pvalue = ", pval.round(4))
            if pval>alpha_test: print("The VaR is valid")
            else: print("The VaR is NOT valid")
            print("---------------------------------------")
        return zscore, pval

    def _get_rolling_drawdown(self, window=252):
        """
        Compute rolling drawdown (peak-to-trough decline) for a given time window (in days).
        ARGUMENTS:
        - window = int, number of days to consider in the rolling window
        OUTPUT: pandas DataFrame with single (if portfolio=True in instantiation) or multiple rolling drawdown columns
        """
        dd = pd.DataFrame(index=self.r.index)
        for tick in self.tickers:
            # sum the returns in the given window (get the grawdown for day t using past n days data)
            roll_sum = self.r[tick].rolling(window, min_periods=1).sum()
            # We only want the negative sums (peak-to-trough decline), set to zero if the sum is positive
            roll_sum[roll_sum > 0.0] = 0.0
            # You can add this next commented line if you want a smoothed value (get the minimun value for the sums over a certain time window)
            # roll_sum = roll_sum.rolling(10).min()
            dd[tick] = roll_sum
        return dd
  
    def Risk_Report(self, var_es_level=0.05):
        """
        Summary report of the most important risk measures for a given returns series (or multiple return series)
        ARGUMENTS:
        - portfolio = bool, if False use self.r, else use self.r_pf
        - var_es_level = float, single alpha level to be used in the report 
        !! One of the alphas defined in the class must be equal to the var_es_level argument
        OUTPUT: pandas DataFrame, key risk metrics
        """
        riskmetrics = pd.DataFrame(columns=self.tickers)
        dd = self._get_rolling_drawdown(window=126)  # 6-months
        hvar = self.get_VaR_basic(dist="hist")
        tvar = self.get_VaR_basic(dist="tstudent")
        hes = self.get_ES_basic(dist="hist")
        tes = self.get_ES_basic(dist="tstudent")
        for tick in self.tickers:
            stdev = self.r[tick].std()
            semi_dev = (self.r[tick][self.r[tick] < self.r[tick].mean()]).std()  # standard dev of returns below the average
            min_dd = dd[tick].min()  # select the lowest value over the entire sample period 
            riskmetrics.loc["Daily Volatility", tick] = stdev
            riskmetrics.loc["Daily SemiDeviation", tick] = semi_dev
            riskmetrics.loc["Max Rolling Drawdown", tick] = min_dd
            riskmetrics.loc[f"Historical_{1-var_es_level}_VaR", tick] = hvar.loc[tick, str(var_es_level)]
            riskmetrics.loc[f"tStudent_{1-var_es_level}_VaR", tick] = tvar.loc[tick, str(var_es_level)]
            riskmetrics.loc[f"Historical_{1-var_es_level}_ES", tick] = hes.loc[tick, str(var_es_level)]
            riskmetrics.loc[f"tStudent_{1-var_es_level}_ES", tick] = tes.loc[tick, str(var_es_level)]
        return riskmetrics


class PortfolioVaR_components(RiskMeasures):
    """
    SubClass to compute Portfolio VaR components (marginal VaR, component VaR, incremental VaR)
    """

    def __init__(self, returns, alphas, position=1, weights=None, portfolio=True, var_computation_method="gaussian"):
        """
        Use 'returns' as input data (time series for all portfolio components)
        Raise error if value of 'alphas' is not unique (but still alphas must be a list with a single value)
        Fix 'portfolio' to True (in this way inherited methods will compute portfolio VaR), raise error if changed
        - var_computation_method = string, distirbution for the VaR computed in the summary report ("gaussian", "tstudent", "hist")
        """
        super().__init__(returns, alphas, position=position, weights=weights, portfolio=portfolio)
        self.var_computation_method = var_computation_method
    
        if len(self.alphas) > 1: raise ValueError("The argument 'alphas' must be a list with a unique value")
        if self.portfolio == False: raise ValueError("The argument 'portfolio' must be set to True")
        if weights is None: raise ValueError("Please define portfolio weights")

    def get_returns(self):
        """
        Obtain passed returns (self.returns) and compute portfolio returns (self.r)
        """
        return self.returns, self.r

    def get_marginal_VaR(self):
        """
        Compute marginal VaR for all portfolio components
        OUTPUT: pandas dataframe
        """
        marginal_vars = pd.DataFrame(columns = self.returns.columns)
        # compute diversified var (=portfolio var obtained with one of the methods defined in the RiskMeasures class) --> single value
        if self.var_computation_method == "gaussian": VAR = self.get_VaR_basic(dist="gaussian")
        elif self.var_computation_method == "tstudent": VAR = self.get_VaR_basic(dist="tstudent")
        else: VAR = self.get_VaR_basic(dist="hist")
        # Compute Beta
        cov_mat = self.returns.cov()
        w = np.array(self.weights)
        matprod = np.dot(cov_mat, w)
        var_pf = np.dot(w.T, np.dot(cov_mat, w))
        beta = matprod/var_pf
        for i, tick in zip(range(len(self.returns.columns)), self.returns.columns):
            marginal_vars.loc["Beta", tick] = beta[i] 
            marginal_vars.loc["Marginal_VaR", tick] = np.float_(((marginal_vars.loc["Beta", tick])*VAR/self.position).values)
        marginal_vars.loc["Diversified_VaR", "Portfolio"] = VAR.iloc[0,0]  # the VAR dataframe will always have only one cell (portfolio var with single alpha, no rolling option)
        return marginal_vars

    def get_component_VaR(self):
        """
        Compute component VaR and % contribution for all portfolio components
        OUTPUT: pandas dataframe
        """
        component_vars = pd.DataFrame(columns = self.returns.columns)
        # use get_marginal_VaR results to compute component var for each asset in the pf
        marg_vars = self.get_marginal_VaR()
        for i, tick in zip(range(len(self.returns.columns)), self.returns.columns):
            component_vars.loc["Component_VaR", tick] = marg_vars.loc["Marginal_VaR", tick]*self.weights[i]*self.position
            component_vars.loc["%_Contribution", tick] = (component_vars.loc["Component_VaR", tick]) / (marg_vars.loc["Diversified_VaR", "Portfolio"])
        sums = component_vars.sum(axis=1)
        component_vars.loc["Component_VaR", "Portfolio"] = sums["Component_VaR"]
        component_vars.loc["%_Contribution", "Portfolio"] = sums["%_Contribution"]
        return component_vars
        
    def get_incremental_VaR(self, delta_position_pct=0.1):
        """
        Compute approximated and exact incrememtal VaR for all portfolio components, 
        given the same percentage increase in the overall portfolio position applied to each portfolio component
        ARGUMENTS:
        - delta_position_pct = float, percentage increase in the overall portfolio position (attribute singularly to each portfolio component)
        OUTPUT: pandas dataframe 
        """
        incr_var = pd.DataFrame(columns = self.returns.columns)
        marg_vars = self.get_marginal_VaR()
        VAR = self.get_VaR_basic(dist=self.var_computation_method)

        for i, tick in zip(range(len(self.returns.columns)), self.returns.columns):
            incr_var.loc["Starting_Portfolio_VaR", tick] = VAR.iloc[0,0]
            # approximated incremental VaR
            incr_var.loc["Approx_Incremental_VaR", tick] = marg_vars.loc["Marginal_VaR", tick]*delta_position_pct*self.position
            incr_var.loc["NewApprox_Portfolio_VaR", tick] = incr_var.loc["Approx_Incremental_VaR", tick] + VAR.iloc[0,0]
            # exact (full re-evaluation) incremental VaR
            ww = self.weights.copy()
            for j in range(len(self.weights)):
                if i != j: ww[j] = self.weights[j]/(sum(self.weights)+delta_position_pct)
                else: ww[j] = (self.weights[j]+delta_position_pct)/(sum(self.weights)+delta_position_pct)
                new_VAR = self.get_VaR_basic(dist=self.var_computation_method, new_w_incr=ww)
                new_VAR = new_VAR + (new_VAR/self.position)*(self.position*delta_position_pct)
                incr_VAR = new_VAR.iloc[0,0] - VAR.iloc[0,0]
                incr_var.loc["Exact_Incremental_VaR", tick] = incr_VAR
                incr_var.loc["NewExact_Portfolio_VaR", tick]  = new_VAR.iloc[0,0]
                incr_var.loc["%_VaR_Increment", tick]  = new_VAR.iloc[0,0]/VAR.iloc[0,0] - 1
        return incr_var

    def VaR_Components_Report(self, delta_position_pct=0.1):
        """
        Final report on Portfolio VaR components
        OUTPUT: pandas dataframe
        """
        report = pd.DataFrame(columns = self.returns.columns)
        marg_vars = self.get_marginal_VaR()
        comp_vars = self.get_component_VaR()
        incr_vars = self.get_incremental_VaR(delta_position_pct=delta_position_pct)
        for tick in self.returns.columns:
            report.loc["Marginal_VaR", tick] = marg_vars.loc["Marginal_VaR", tick]
            report.loc["Diversified_VaR", "Portfolio"] = marg_vars.loc["Diversified_VaR", "Portfolio"]
            report.loc["Component_VaR", tick] = comp_vars.loc["Component_VaR", tick]
            report.loc["%_Contribution", tick] = comp_vars.loc["%_Contribution", tick]
            report.loc["Component_VaR", "Portfolio"] = comp_vars.loc["Component_VaR", "Portfolio"]
            report.loc["%_Contribution", "Portfolio"] = comp_vars.loc["%_Contribution", "Portfolio"]
            report.loc["Exact_Incremental_VaR", tick] = incr_vars.loc["Exact_Incremental_VaR", tick]
            report.loc["New_Portfolio_VaR", tick] = incr_vars.loc["NewExact_Portfolio_VaR", tick]
            report.loc["%_VaR_Increment", tick] = incr_vars.loc["%_VaR_Increment", tick]
        return report



