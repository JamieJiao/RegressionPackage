import statsanalysis from preregression

import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt


def linear_regression(y, X):
    X = sm.add_constant(X)
    model = sm.OLS(y, X)
    results = model.fit()
    return results

# to exclude y in a variable list
def exclude_col_name(X, variable_name):
    col_names = []
    for col_name in X.columns:
        if col_name != variable_name:
            col_names.append(col_name)
    return col_names

# vif to check multicollinearity between variables
def vif_results(X, *arg):
    # X, pandas dataframe
    vifs = {}
    r_squares = {}
    for variable_name in arg:
        variables_names = exclude_col_name(X, variable_name)
        variables_value = X[variables_names]
        variable_value = X[variable_name]
        r_square = linear_regression(variable_value, variables_value).rsquared
        vif = 1 / (1 - r_square)
        vifs[variable_name] = vif
        r_squares[variable_name] = r_square
    return vifs, r_squares

def residual_distribution(residuals):
    mu = np.mean(residuals)
    sigma = np.sqrt(statsanalysis.calculate_sum_of_squares(residuals) / len(residuals))
    count, bins, ignored = plt.hist(residuals, 30, density=True)
    plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * \
        np.exp( - (bins - mu)**2 / (2 * sigma**2) ),linewidth=2, color='r')
    plt.show()

# stepwise method to select the best regression model
def find_var_with_smallest_p(p_values):
    mini = 0
    initial = True
    var = None
    for key, value in p_values.items():
        if key == 'const':
            continue
        else:
            if initial:
                mini = value
                var = key
            initial = False
            if value < mini:
                mini = value
                var = key
    return var

def enter_new_var(y, X, entered_vars, dropped_vars):
    var_candidates = {}
    for var in X.columns:
        if var not in entered_vars and var not in dropped_vars:
                ols = linear_regression(y, X[entered_vars + [var]])
                p_values = ols.pvalues
                var_candidates[var] = p_values[var]

    var_enter = find_var_with_smallest_p(var_candidates)
    try:
        if var_candidates[var_enter] < 0.1:
            entered_vars.append(var_enter)
            stop = False
            return entered_vars, stop
        else:
            stop = True
            return entered_vars, stop
    except:
        if var_enter == None:
            stop = True
            return entered_vars, stop

def vars_drop(y, X, entered_vars, p_criteria):
    ols = linear_regression(y, X[entered_vars])
    p_values = ols.pvalues
    dropped_vars = []
    for index in p_values.index:
        if p_values[index] > p_criteria and index != 'const':
            entered_vars.remove(index)
            dropped_vars.append(index)
    return entered_vars, dropped_vars

def stepwise_regression(y, X, p_criteria=0.15):
    ols = linear_regression(y, X)
    p_values = {}
    # convert to dictionary
    for index in ols.pvalues.index:
        p_values[index] = ols.pvalues[index]

    first_var = find_var_with_smallest_p(p_values)
    entered_vars = []
    dropped_vars = []
    entered_vars.append(first_var)
    while True:
        entered_vars, stop = enter_new_var(y, X, entered_vars, dropped_vars)
        if stop == True:
            ols = linear_regression(y, X[entered_vars])
            return ols
            new_entered_vars, dropped_vars = vars_drop(y, X, entered_vars, p_criteria)
