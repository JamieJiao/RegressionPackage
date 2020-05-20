import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import t
from math import sqrt
from numpy import mean
from scipy.stats import sem
from scipy import stats

df_mississauga = pd.read_excel('Jamie_conv_store_Mississauga_2000-2019.xlsx')

class ManageMississaugaData:
    def __init__(self, df_m):
        self.df_m = df_m

    def after_replace_sales_column_value(self):
        self.df_m['sales/per week'] = self.df_m.iloc[:, -1]
        return self.df_m

    def after_drop_columns(self):
        df = self.after_replace_sales_column_value()
        droped_column_index = [ - (i + 1) for i in range(7)]
        self.df_m = self.df_m.drop(self.df_m.columns[[droped_column_index]], axis=1)
        return self.df_m

    def after_add_col_city(self):
        df_m = self.after_drop_columns()
        df_m['City'] = 'Mississauga'
        return df_m

    # ?? how to apply nomral function intead of lambda function
    # def convert_data_to_one_or_zero(self, one_value):
    #     if x == one_value:
    #         x = 1
    #     else:
    #         x = 0
    #     return x

df_toronto = pd.read_excel('conv store_Toronto_2000-2019.xlsx')

class ManageTwoData(ManageMississaugaData):
    def __init__(self, df_m, df_t):
        ManageMississaugaData.__init__(self, df_m)
        self.df_t = df_t
        self.df_m = self.after_add_col_city()

    def create_id(self):
        observation_num = len(self.df_t)
        return [i + 1 for i in range(observation_num)]

    def find_diff_cols_in_two_dataframe(self, df, df_being_checked):
        same_col_names = []
        diff_col_names = []
        for col_name_being_checked in df_being_checked.columns:
            find = False
            for col_name_df in df.columns:
                if col_name_being_checked == col_name_df:
                    same_col_names.append(col_name_being_checked)
                    find = True
            if find == False:
                diff_col_names.append(col_name_being_checked)

        return diff_col_names
    
    def change_col_names_as_first_dataset(self):
        df_t = self.df_t
        df_t.rename(columns={'MLS_Num':'MLS', 'Size_Sq_Ft':'area(sqft)', 
                                    'Address':'location(address)', 'Occupation':'occupation', 
                                    'Transaction_Time':'sold date', 'Rent':'Rental/month', 
                                    'Lottery':'lottery', 'Franchise':'franchise', 
                                    'Sold_Price':'Sold Price', 'Tax':'taxes', 
                                    'Open_Days':'days open'}, inplace=True)

        return df_t
    
    def add_or_del_cols(self):
        df_t = self.change_col_names_as_first_dataset()
        df_t['ID'] = self.create_id()
        df_t['City'] = 'Toronto'
        df_t['cont date'] = 'NaN'
        df_t['sales/per week'] = 'NaN'
        df_t.drop(['Zoning', 'Park_Space', 'Garage'], axis=1, inplace=True)
        return df_t

    def concat_data_series(self):
        df_m = self.df_m
        df_t = self.add_or_del_cols()
        obs_num = len(df_m) + len(df_t)
        col_num = len(df_m.columns)
        values_dic = {}

        for col_name in df_m.columns:
            df_m_series = df_m[col_name]
            df_t_series = df_t[col_name]
            concated_series = pd.concat([df_m_series, df_t_series], ignore_index=True)
            values_dic[str(col_name)] = concated_series
        
        return values_dic

    def map_col_names(self, old_col_names, standard_col_names):
        col_mapper = {}
        col_len = len(old_col_names)
        for i in range(col_len):
            col_mapper[old_col_names[i]] = standard_col_names[i]
        return col_mapper

    def combine_data(self):
        columns = self.df_m.columns
        values_dic = self.concat_data_series()
        df_combined = pd.DataFrame(values_dic, columns=columns)
        df_combined.replace('NaN', np.NaN, inplace=True)
        standard_col_names = ['ID', 'MLS', 'Sold_Price', 'Address', 'Area_Size', 
        'Sold_Date', 'Cont_Date', 'Occupation', 'Franchise', 'Days_Open', 'DOM',
        'Garage_Type', 'Lottery', 'Sales_Per_Week', 'Rental_Month', 'Taxes' ,'City']
        col_mapper = self.map_col_names(columns, standard_col_names)
        df_combined.rename(columns=col_mapper, inplace=True)
        df_combined.set_index(['MLS'], inplace=True)
        return df_combined
    
    def drop_nan(self):
        df = self.combine_data()
        df = df[df['Rental_Month'].notna()]
        df = df[df['Area_Size'].notna()]
        df = df[df['Days_Open'].notna()]
        return df

    def convert_col(self):
        df = self.drop_nan()

        df['Franchise_Converted'] = df['Franchise'].apply(lambda x: 1 if x == 'Y' \
                                                or x == 1 else 0)
        df['Garage_Type_Converted'] = df['Garage_Type'].apply(lambda x: 0 if x == 'None' \
                                                or x == np.NaN else 1)
        df['Lottery_Converted'] = df['Lottery'].apply(lambda x: 1 if x == 'Y' or x == 1 else 0)

        df['Occupation_Converted'] = df['Occupation'].apply(lambda x: 1 if x == 'Owner' \
                                                or x == 'Own+Ten' else 0)

        return df

# df_mississauga = ManageMississaugaData(df_mississauga).add_col_city()
# df_mississauga_cols = df_mississauga.columns

df_toronto = ManageTwoData(df_mississauga, df_toronto)
diff_col_names = df_toronto.find_diff_cols_in_two_dataframe(
    df=df_toronto.df_m, df_being_checked=df_toronto.df_t)

# print('different columns:', diff_col_names)
# print('Mississauga Data Columns names:', len(manage_two_datasets.df_m.columns))
# print('Toronto Data Columns names:', len(manage_two_datasets.add_or_del_cols().columns))

# **double check if there are different columns 
diff_col_names_new = df_toronto.find_diff_cols_in_two_dataframe(
    df=df_toronto.df_m, df_being_checked=df_toronto.df_t)

# print('double check if different columns:', diff_col_names_new)
# print(type(manage_two_datasets.df_m['ID']))

# **combiend_data
df = df_toronto.convert_col()
# print(df)
# df.to_excel('Combined_Data.xlsx')
# df.to_excel('Combined_Data_After_Conversion.xlsx')

class Analysis:
    def __init__(self, df):
        self.df = df    

    def loop_plot(self, plot_col_num, plot_row_num, axs):
        df = self.df
        df_cols = df.columns
        position = 0
        for row in range(plot_row_num):
            for col in range(plot_col_num):
                if position == len(df_cols):
                    break
                df_col_name = df_cols[position]

                # print(df[df_col_name].dtype)
                # print(df_col_name)
                if str(df[df_col_name].dtype) == 'int64' or str(df[df_col_name].dtype) == 'float64':
                    axs[row, col].boxplot(df[df_col_name])
                    axs[row, col].set_title(df_col_name) 
                position += 1
    
    def box_plot(self):
        plot_num = len(self.df.columns)
        plot_col_num = 4
        plot_row_num = math.ceil(plot_num / plot_col_num)
        fig, axs = plt.subplots(plot_row_num, plot_col_num, figsize=(15,6))
        self.loop_plot(plot_col_num, plot_row_num, axs)
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
    
    def scatter_plot(self, df_col_name):
        plt.plot(self.df[df_col_name], self.df.iloc[:, 0], 'bv', mfc='red')
        plt.xlabel(df_col_name,fontsize=16)
        plt.ylabel('Sold_Price',fontsize=16)
        plt.show()

    def calculate_sum_of_squares(self, data):
        list_sum_of_squares = []
        mean_value = mean(data)
        for value in data:
            list_sum_of_squares.append((value - mean_value)**2)
        return sum(list_sum_of_squares)
    
    def calculate_sample_variance(self, data1, data2, degree_of_freedom):
        mean1, mean2 = mean(data1), mean(data2)
        sum_of_square_data1 = self.calculate_sum_of_squares(data1)
        sum_of_square_data2 = self.calculate_sum_of_squares(data2)
        return (sum_of_square_data1 + sum_of_square_data2) / degree_of_freedom

    def t_test(self, data1, data2):
        degree_of_freedom = len(data1) + len(data2) - 2
        two_sample_difference = mean(data1) - mean(data2)
        # sample_variance = self.calculate_sample_variance(data1, data2, degree_of_freedom)
        # ** not assume that two samples have the same variance
        variance1 = self.calculate_sum_of_squares(data1)/(len(data1) - 1)
        variance2 = self.calculate_sum_of_squares(data2)/(len(data2) - 1)
        t_value = two_sample_difference / sqrt(variance1/len(data1) + variance2/len(data2))
        # t_value = two_sample_difference / sqrt(sample_variance/len(data1) + sample_variance/len(data2))
        p_value = (1.0 - t.cdf(abs(t_value), degree_of_freedom)) * 2.0
        standard_deviations = (sqrt(variance1), sqrt(variance2))
        means = (mean(data1), mean(data2))
        return p_value, t_value, standard_deviations, means

    def independent_ttest(self, data1, data2, alpha):
        # calculate means
        mean1, mean2 = mean(data1), mean(data2)
        # calculate standard errors
        se1, se2 = sem(data1), sem(data2)
        # standard error on the difference between the samples
        sed = sqrt(se1**2.0 + se2**2.0)
        # calculate the t statistic
        t_stat = (mean1 - mean2) / sed
        # degrees of freedom
        df = len(data1) + len(data2) - 2
        # calculate the critical value
        cv = t.ppf(1.0 - alpha, df)
        # calculate the p-value
        p = (1.0 - t.cdf(abs(t_stat), df)) * 2.0
        # return everything
        return t_stat, p
    
    def caculate_covariance(self, data1, data2):
        data1_array = data1.values
        data2_array = data2.values
        mean1 = mean(data1_array)
        mean2 = mean(data2_array)
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
        correlation = covariance/sqrt(variance1 * variance2)
        return correlation
        
# results = Analysis(df)
# results.box_plot()

# **drop 'Taxes' column where most of the values are NaN,
# **drop NaN based on 'Rental_Month' and 'Area_Size'.
df.drop(['Taxes'], axis=1, inplace=True)

results = Analysis(df)
# results.box_plot()

# ** find outliers
# outliers_sold_price = results.find_outliers('Sold_Price')
# print('Sold_Price outliers:', '\n', outliers_sold_price)
# outliers_rental = results.find_outliers('Rental_Month')
# print('Rental_Month outliers:', '\n', outliers_rental)
# outliers_DOM = results.find_outliers('DOM')
# print('DOM outliers:', '\n', outliers_DOM)

# ** drop the outliers by its indexs, MLS
try:
    df.drop(index='W984160', inplace=True)    
except:
    pass
# ** find which relationship between Sold_Price and other variables
# results.scatter_plot('Area_Size')
# results.scatter_plot('DOM')
# results.scatter_plot('Rental_Month')

def t_test_results_display(df, variable):
    data1 = df.loc[df[variable] == 1]['Sold_Price']
    data2 = df.loc[df[variable] == 0]['Sold_Price']
    p_value, t_value, means, stds = results.t_test(data1, data2)
    print('{}:\n'.format(variable.replace('_Converted', '')), \
        'p value:',p_value, '\n', 't value', t_value, '\n', 'means:', means, \
        '\n', 'standard deviations:', stds)

# t_test_results_display(df, 'Franchise_Converted')
# t_test_results_display(df, 'Garage_Type_Converted')
# t_test_results_display(df, 'Lottery_Converted')
# t_test_results_display(df, 'Occupation_Converted')

# ** check data for both cities separately
df_m = df.loc[df['City'] == 'Mississauga']
df_t = df.loc[df['City'] == 'Toronto']
# t_test_results_display(df_t, 'Franchise_Converted')

data1 = df_t.loc[df_t['Franchise_Converted'] == 1]['Sold_Price']
data2 = df_t.loc[df_t['Franchise_Converted'] == 0]['Sold_Price']
# t, p = results.independent_ttest(data1, data2, 0.05)
# print(t, p)
# t, p = stats.ttest_ind(data1,data2)
# print(t, p)

# ** test p value by using original data, without dropping any missing value
df_t_origin = pd.read_excel('conv store_Toronto_2000-2019.xlsx')
# t_test_results_display(df_t_origin, 'Franchise')
data1 = df_t_origin.loc[df_t_origin['Franchise'] == 1]['Sold_Price']
data2 = df_t_origin.loc[df_t_origin['Franchise'] == 0]['Sold_Price']
t, p = stats.ttest_ind(data1,data2)
# print(t, p)

def correlation_display(df, variable1, variable2):
    print('{} {} correlation:'.format(variable1, variable2), \
        results.caculate_correlation(df[variable1], df[variable2]))

correlation_display(df, 'Rental_Month', 'Sold_Price')
correlation_display(df, 'DOM', 'Sold_Price')
correlation_display(df, 'Area_Size', 'Sold_Price')
# print(df.corr())
