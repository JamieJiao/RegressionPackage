import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import stats


class statsanalysis:
    def __init__(self, df):
        self.df = df    

    def loop_plot(self, plot_col_num, plot_row_num, axs, y=None, *arg, box_plot=False, \
        scatter_plot=False):
        position = 0
        for row in range(plot_row_num):
            for col in range(plot_col_num):
                if position == len(arg):
                    break
                df_col_name = arg[position].name
                var = arg[position]
                if box_plot:
                    try:
                        axs[row, col].boxplot(var)
                        axs[row, col].set_title(df_col_name)
                    except:
                        # ** when var number less than subplots column number
                        # ** subplots then is one dimentional
                        axs[col].boxplot(var)
                        axs[col].set_title(df_col_name)
                if scatter_plot:
                    try:
                        axs[row, col].plot(var, y, 'bv', mfc='red')
                        axs[row, col].set_title(df_col_name)
                    except:
                        axs[col].plot(var, y, 'bv', mfc='red')
                        axs[col].set_title(df_col_name)
                position += 1
    
    def plot(self, *args, plot_col_num = 4, y = None, box_plot=False, scatter_plot=False):
        plot_num = len(args)
        plot_row_num = math.ceil(plot_num / plot_col_num)
        fig, axs = plt.subplots(plot_row_num, plot_col_num, figsize=(15,6))
        if box_plot:
            self.loop_plot(plot_col_num, plot_row_num, axs, *args, box_plot=True)
        if scatter_plot:
            self.loop_plot(plot_col_num, plot_row_num, axs, y, *args, scatter_plot=True)
        fig.subplots_adjust(left=0.08, right=0.98, bottom=0.05, top=0.9,
                    hspace=0.7, wspace=0.3)
        plt.show()

    def calculate_whiskers(self, df_series, obs_num):
        Q1_quartile = round(obs_num * 0.25)
        Q3_quartile = round(obs_num * 0.75)
        Q1_quartile_value = df_series.iloc[Q1_quartile]
        Q3_quartile_value = df_series.iloc[Q3_quartile]
        # **default value
        whis = 1.5
        Q3_Q1_range = Q3_quartile_value - Q1_quartile_value
        lower_whisker = Q1_quartile_value - whis * (Q3_Q1_range)
        upper_whisker = Q3_quartile_value + whis * (Q3_Q1_range)
        return lower_whisker, upper_whisker

    def find_outliers(self, df_col_name):
        df = self.df[[df_col_name]]
        df = df.sort_values(by = [df_col_name])
        df_series = df[df_col_name]
        obs_num = len(df_series)
        lower_whisker, upper_whisker = self.calculate_whiskers(df_series, obs_num)

        df_outliers = df.loc[(df[df_col_name] >= upper_whisker) | (df[df_col_name] <= lower_whisker)]

        return df_outliers

    def calculate_sum_of_squares(self, data):
        sum_of_squares = []
        mean_value = np.mean(data)
        for value in data:
            sum_of_squares.append((value - mean_value)**2)
        return sum(sum_of_squares)

    def t_test(self, data1, data2):
        degree_of_freedom = len(data1) + len(data2) - 2
        mean_difference = np.mean(data1) - np.mean(data2)
        variance1 = self.calculate_sum_of_squares(data1)/(len(data1) - 1)
        variance2 = self.calculate_sum_of_squares(data2)/(len(data2) - 1)
        t_value = mean_difference / math.sqrt(variance1/len(data1) + variance2/len(data2))
        p_value = (1.0 - stats.t.cdf(abs(t_value), degree_of_freedom)) * 2.0
        standard_deviations = (math.sqrt(variance1), math.sqrt(variance2))
        means = (np.mean(data1), np.mean(data2))
        return p_value, t_value, standard_deviations, means
    
    def caculate_covariance(self, data1, data2):
        data1_array = data1.values
        data2_array = data2.values
        mean1 = np.mean(data1_array)
        mean2 = np.mean(data2_array)
        # ** data1 and data2 are two variables, with same number of observations
        observation_num = len(data1_array)
        diff1 = (data1_array - mean1)
        diff2 = (data2_array - mean2)
        covariance = np.dot(diff1, diff2)
        return covariance

    def caculate_correlation(self, data1, data2):
        variance1 = self.calculate_sum_of_squares(data1)
        variance2 = self.calculate_sum_of_squares(data2)
        covariance = self.caculate_covariance(data1, data2)
        correlation = covariance/math.sqrt(variance1 * variance2)
        return correlation

    def correlation_significance(self, data1, data2):
        correlation = self.caculate_correlation(data1, data2)
        degree_of_freedom = len(data1) - 2
        t_value = correlation * math.sqrt((degree_of_freedom)/(1.0 - correlation**2))
        p_value = (1 - stats.t.cdf(abs(t_value), degree_of_freedom)) * 2.0
        return p_value
    
    def correlation_results(self, data1, data2):
        correlation = self.caculate_correlation(data1, data2)
        p_value = self.correlation_significance(data1, data2)
        return correlation, p_value
    
    def correlation_matrix_display(self, *arg):
        df_corr_matrix = pd.DataFrame(index=arg, columns=arg)
        
        row = 0
        for index in df_corr_matrix.index:
            for col in df_corr_matrix.columns:
                data1 = self.df[index]
                data2 = self.df[col]
                try:
                    correlation, p_value = self.correlation_results(data1, data2)
                    correlation = round(correlation, 4)
                    p_value = round(p_value, 4)
                except:
                    # ** when same vars, correlation = 1
                    # ** the denominator(1 - correlation) = 0 in calculating t value
                    correlation = 1
                    p_value = 0.0
                df_corr_matrix.iloc[row][col] = (correlation, p_value)
            row = row + 1

        return df_corr_matrix

    def find_distinct_categories(self, data1, data2):
        categories_data1 = []
        categories_data2 = []
        for i in range(len(data1)):
            if data1[i] not in categories_data1:
                categories_data1.append(data1[i])
            if data2[i] not in categories_data2:
                categories_data2.append(data2[i])

        return categories_data1, categories_data2
    
    def count_values(self, data1, data2, value1, value2):
        count = 0
        size = len(data1)
        for i in range(size):
            if data1[i] == value1 and data2[i] == value2:
                count = count + 1
        return count
    
    def contingency_table_create(self, data1, data2, data1_category_n=2, data2_category_n=2):
        categories_data1, categories_data2 = self.find_distinct_categories(data1, data2)
        shape = (data1_category_n, data2_category_n)
        contingency_table = np.zeros(shape)
        for c in range(data1_category_n):
            for r in range(data2_category_n):
                contingency_table[c][r] = self.count_values(data1, data2, categories_data1[c], categories_data2[r])
        return contingency_table

    def chi_square_test(self, data1, data2, data1_category_n=2, data2_category_n=2, alpha=0.05):
        n = len(data1)
        df = (data1_category_n - 1) * (data2_category_n - 1)
        contingency_table = self.contingency_table_create(data1, data2)
        column_num = contingency_table.shape[0]
        row_num = contingency_table.shape[1]
        column_totals = [sum(contingency_table[:, x]) for x in range(column_num)]
        row_totals = [sum(contingency_table[x, :]) for x in range(row_num)]

        X_square = 0
        for c in range(column_num):
            for r in range(row_num):
                expected_value_mle = (column_totals[r]*row_totals[c])/n
                deviation = (contingency_table[c][r] - expected_value_mle)**2/expected_value_mle
                X_square = X_square + deviation
        p_value = 1 - stats.chi2.cdf(x=X_square, df=df)
        return X_square, p_value
        
    def total_mean_and_obs(self, *arg):
        total = 0
        total_obs = 0
        for data in arg:
            total += + sum(data)
            total_obs += + len(data)
        mean = total / total_obs
        return total_obs, mean

    def anova_test(self, *arg):
        # **sstr: treatment_sum_of_squares, sse: error sum of squares
        number_of_factor = len(arg)
        sse = 0
        sstr = 0
        total_obs, total_mean = self.total_mean_and_obs(*arg)
        for i in range(number_of_factor):
            sse = sse + self.calculate_sum_of_squares(arg[i])
            sstr = sstr + len(arg[i]) * (np.mean(arg[i]) - total_mean)**2
        
        degree_of_freedom_numerator = number_of_factor - 1
        degree_of_freedom_denominator = total_obs - number_of_factor

        f_value = (sstr/degree_of_freedom_numerator) / (sse/degree_of_freedom_denominator)
        p_value = 1 - stats.f.cdf(f_value, degree_of_freedom_numerator, degree_of_freedom_denominator)

        return f_value, p_value
