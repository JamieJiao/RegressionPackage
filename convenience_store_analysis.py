import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import stats
from datetime import datetime 

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

df_t = pd.read_excel('conv store_Toronto_2000-2019.xlsx')

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
    
    def modify_data_in_cols(self):
        df_t = self.change_col_names_as_first_dataset()
        df_t['ID'] = self.create_id()
        df_t['City'] = 'Toronto'
        df_t['cont date'] = 'NaN'
        df_t['sales/per week'] = 'NaN'
        df_t.drop(['Zoning', 'Park_Space', 'Garage'], axis=1, inplace=True)
        df_t['sold date'] = df_t['sold date'].apply(lambda x: \
                                                datetime.strptime(str(x), \
                                                '%Y-%m-%d %H:%M:%S').strftime('%m/%d/%Y'))
        return df_t

    def concat_data_series(self):
        df_m = self.df_m
        df_t = self.modify_data_in_cols()
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
        df['Year'] = df['Sold_Date'].apply(lambda x: int(x.split('/')[-1]))

        df.sort_values('Year', ascending=True, inplace=True)
        return df
    
    def creare_dummies_for_opendays(self):
        df = self.convert_col()
        # ** only one observation has 4 open days,
        # ** so we include it into the 5 open days group
        df['Days_Open_5'] = df['Days_Open'].apply(lambda x: 1 if x == 4 or x == 5 else 0)
        df['Days_Open_6'] = df['Days_Open'].apply(lambda x: 1 if x == 6 else 0)
        df['Days_Open_7'] = df['Days_Open'].apply(lambda x: 1 if x == 7 else 0)
        return df

# df_mississauga = ManageMississaugaData(df_mississauga).add_col_city()
# df_mississauga_cols = df_mississauga.columns

df_t = ManageTwoData(df_mississauga, df_t)
diff_col_names = df_t.find_diff_cols_in_two_dataframe(
    df=df_t.df_m, df_being_checked=df_t.df_t)

# print('different columns:', diff_col_names)
# print('Mississauga Data Columns names:', len(manage_two_datasets.df_m.columns))
# print('Toronto Data Columns names:', len(manage_two_datasets.add_or_del_cols().columns))

# **double check if there are different columns 
diff_col_names_new = df_t.find_diff_cols_in_two_dataframe(
    df=df_t.df_m, df_being_checked=df_t.df_t)

# print('double check if different columns:', diff_col_names_new)
# print(type(manage_two_datasets.df_m['ID']))

# **combiend_data
df = df_t.creare_dummies_for_opendays()
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
    
    def scatter_plot(self, data1, data2):
        plt.plot(data1, data2, 'bv', mfc='red')
        plt.xlabel(data1.name,fontsize=16)
        plt.ylabel(data2.name,fontsize=16)
        plt.show()

    def calculate_sum_of_squares(self, data):
        list_sum_of_squares = []
        mean_value = np.mean(data)
        for value in data:
            list_sum_of_squares.append((value - mean_value)**2)
        return sum(list_sum_of_squares)
    
    def calculate_sample_variance(self, data1, data2, degree_of_freedom):
        mean1, mean2 = np.mean(data1), np.mean(data2)
        sum_of_square_data1 = self.calculate_sum_of_squares(data1)
        sum_of_square_data2 = self.calculate_sum_of_squares(data2)
        return (sum_of_square_data1 + sum_of_square_data2) / degree_of_freedom

    def t_test(self, data1, data2):
        degree_of_freedom = len(data1) + len(data2) - 2
        two_sample_difference = np.mean(data1) - np.mean(data2)
        # sample_variance = self.calculate_sample_variance(data1, data2, degree_of_freedom)
        # ** not assume that two samples have the same variance
        variance1 = self.calculate_sum_of_squares(data1)/(len(data1) - 1)
        variance2 = self.calculate_sum_of_squares(data2)/(len(data2) - 1)
        t_value = two_sample_difference / math.sqrt(variance1/len(data1) + variance2/len(data2))
        # t_value = two_sample_difference / sqrt(sample_variance/len(data1) + sample_variance/len(data2))
        p_value = (1.0 - stats.t.cdf(abs(t_value), degree_of_freedom)) * 2.0
        standard_deviations = (math.sqrt(variance1), math.sqrt(variance2))
        means = (np.mean(data1), np.mean(data2))
        return p_value, t_value, standard_deviations, means

    def independent_ttest(self, data1, data2, alpha):
        # calculate means
        mean1, mean2 = np.mean(data1), np.mean(data2)
        # calculate standard errors
        se1, se2 = stats.sem(data1), stats.sem(data2)
        # standard error on the difference between the samples
        sed = math.sqrt(se1**2.0 + se2**2.0)
        # calculate the t statistic
        t_stat = (mean1 - mean2) / sed
        # degrees of freedom
        df = len(data1) + len(data2) - 2
        # calculate the critical value
        cv = stats.t.ppf(1.0 - alpha, df)
        # calculate the p-value
        p = (1.0 - stats.t.cdf(abs(t_stat), df)) * 2.0
        # return everything
        return t_stat, p
    
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
# results.scatter_plot(df['Area_Size'], df['Sold_Price'])
# results.scatter_plot(df['DOM'], df['Sold_Price'])
# results.scatter_plot(df['Rental_Month'], df['Sold_Price'])
# ** check if there is time effect
# results.scatter_plot(df['Year'], df['Sold_Price'])

# results.scatter_plot(df['Area_Size'], df['Sold_Price']/df['Area_Size'])
# results.scatter_plot(df['DOM'], df['Sold_Price']/df['Area_Size'])
# results.scatter_plot(df['Rental_Month'], df['Sold_Price']/df['Area_Size'])
# results.scatter_plot(df['Year'], df['Sold_Price']/df['Area_Size'])

def t_test_results_display(df, variable, price_per_square_feet=False):
    if price_per_square_feet:
        data1 = df.loc[df[variable] == 1]['Sold_Price'] / df.loc[df[variable] == 1]['Area_Size']
        data2 = df.loc[df[variable] == 0]['Sold_Price'] / df.loc[df[variable] == 0]['Area_Size']
    else:
        data1 = df.loc[df[variable] == 1]['Sold_Price']
        data2 = df.loc[df[variable] == 0]['Sold_Price']
    p_value, t_value, means, stds = results.t_test(data1, data2)
    print('{}:\n'.format(variable.replace('_Converted', '')), \
        'p value:',p_value, '\n', 't value', t_value, '\n', 'means:', means, \
        '\n', 'standard deviations:', stds)

print('Total Sold Price:')
t_test_results_display(df, 'Franchise_Converted')
t_test_results_display(df, 'Garage_Type_Converted')
t_test_results_display(df, 'Lottery_Converted')
t_test_results_display(df, 'Occupation_Converted')
print('\n')
print('Sold Price Per Square Feet:')
t_test_results_display(df, 'Franchise_Converted', price_per_square_feet=True)
t_test_results_display(df, 'Garage_Type_Converted', price_per_square_feet=True)
t_test_results_display(df, 'Lottery_Converted', price_per_square_feet=True)
t_test_results_display(df, 'Occupation_Converted', price_per_square_feet=True)
print('\n')

# ** t test for both cities data separately
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
# df_t_origin = pd.read_excel('conv store_Toronto_2000-2019.xlsx')
# t_test_results_display(df_t_origin, 'Franchise')
# data1 = df_t_origin.loc[df_t_origin['Franchise'] == 1]['Sold_Price']
# data2 = df_t_origin.loc[df_t_origin['Franchise'] == 0]['Sold_Price']
# t, p = stats.ttest_ind(data1,data2)
# print(t, p)

def correlation_display(df, variable1, variable2, price_per_square_feet=False):
    if price_per_square_feet:
        data1 = df[variable1]
        data2 = df[variable2]/df['Area_Size']
    else:
        data1 = df[variable1]
        data2 = df[variable2]
    correlation, p_value =  results.correlation_results(data1, data2)
    print('{} {}: correlation: {} p_value: {}'.format(variable1, variable2, correlation, p_value))

print('correlation with sold price:')
correlation_display(df, 'Rental_Month', 'Sold_Price')
correlation_display(df, 'DOM', 'Sold_Price')
correlation_display(df, 'Area_Size', 'Sold_Price')
# ** check if there is time effect
correlation_display(df, 'Year', 'Sold_Price')
print('\n')
print('correlation with sold price per square feet:')
correlation_display(df, 'Rental_Month', 'Sold_Price', price_per_square_feet=True)
correlation_display(df, 'DOM', 'Sold_Price', price_per_square_feet=True)
correlation_display(df, 'Year', 'Sold_Price', price_per_square_feet=True)
print('\n')

open_days_5 = df.loc[df['Days_Open_5'] == 1]['Sold_Price']
open_days_6 = df.loc[df['Days_Open_6'] == 1]['Sold_Price']
open_days_7 = df.loc[df['Days_Open_7'] == 1]['Sold_Price']
f, p = results.anova_test(open_days_5, open_days_6, open_days_7)
print('anova test for open days 5, 6, 7 with sold price: ', f, p)
open_days_5_size_price = df.loc[df['Days_Open_5'] == 1]['Sold_Price']/df.loc[df['Days_Open_5'] == 1]['Area_Size']
open_days_6_size_price = df.loc[df['Days_Open_6'] == 1]['Sold_Price']/df.loc[df['Days_Open_6'] == 1]['Area_Size']
open_days_7_size_price = df.loc[df['Days_Open_7'] == 1]['Sold_Price']/df.loc[df['Days_Open_7'] == 1]['Area_Size']
f, p = results.anova_test(open_days_5_size_price, open_days_6_size_price, open_days_7_size_price)
print('anova test for open days 5, 6, 7 with sold price per square feet: ', f, p)
