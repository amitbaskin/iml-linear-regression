import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dateutil.parser import parse
import re
import os


def fit_linear_regression(design_mat, response_vec):
    singular_values = np.linalg.svd(design_mat, compute_uv=False)
    coefficients_vec = np.linalg.pinv(design_mat.T) @ response_vec
    return coefficients_vec, singular_values


def predict(design_mat, coefficients_vec):
    return np.matmul(design_mat.T, coefficients_vec)


def mse(prediction_vec, response_vec):
    diff = prediction_vec - response_vec
    vec_len = len(response_vec)
    return np.sum(diff ** 2) / vec_len


def is_int_and_not_negative(num):
    if isinstance(num, int) or (isinstance(num, float) and num % 1 == float(0)):
        if num >= 0:
            return True
    return False


def is_float_and_not_negative(num):
    if isinstance(num, float) or isinstance(num, int):
        if num >= 0:
            return True
    return False


def load_data(path):
    data_df = load_valid_data(path)
    data_df = data_df.iloc[:, 1:21]

    data_price_by_zip_code = data_df[['price', 'zipcode']]
    mean_price = data_price_by_zip_code.groupby('zipcode').mean()
    mean_price = mean_price.rename(
        columns={'zipdoe': 'zipcode', 'price': 'mean_price_by_zipcode'})

    data_df = data_df.merge(mean_price, on='zipcode')
    data_df = data_df.drop(['zipcode'], axis=1)

    data_df[['year', 'month']] = data_df.date.str.split(expand=True)
    data_df = pd.get_dummies(data_df, columns=['year', 'month'])
    data_df = data_df.drop(['date'], axis=1)

    cols = data_df.columns.tolist()

    def convert_to_int(string):
        return int(string) if string.isdigit() else string

    def keys(string):
        return [convert_to_int(component)
                for component in re.split('(\d+)', string)]

    cols.sort(key=keys)
    data_df = data_df[cols]

    return data_df


def load_valid_data(path):
    os.chdir(os.path.dirname(path))
    data_df = pd.read_csv(os.path.basename(path))
    for row in data_df.iterrows():
        if not is_id(row):
            data_df = data_df.drop([row[0]], axis=0)
            continue

        if not is_date(row):
            data_df = data_df.drop([row[0]], axis=0)
            continue
        date = row[1]['date']
        actual_date = parse(date, fuzzy=True)
        year = actual_date.year
        month = actual_date.month
        data_df.at[row[0], 'date'] = str(year) + ' ' + str(month)

        if not is_price(row):
            data_df = data_df.drop([row[0]], axis=0)
            continue

        if not is_bedrooms_num(row):
            data_df = data_df.drop([row[0]], axis=0)
            continue

        if not is_bathrooms_num(row):
            data_df = data_df.drop([row[0]], axis=0)
            continue

        if not is_sqft_living(row):
            data_df = data_df.drop([row[0]], axis=0)
            continue

        if not is_sqft_lot(row):
            data_df = data_df.drop([row[0]], axis=0)
            continue

        if not is_floors(row):
            data_df = data_df.drop([row[0]], axis=0)
            continue

        if not is_waterfront(row):
            data_df = data_df.drop([row[0]], axis=0)
            continue

        if not is_view(row):
            data_df = data_df.drop([row[0]], axis=0)
            continue

        if not is_condition(row):
            data_df = data_df.drop([row[0]], axis=0)
            continue

        if not is_grade(row):
            data_df = data_df.drop([row[0]], axis=0)
            continue

        if not is_sqft_above(row):
            data_df = data_df.drop([row[0]], axis=0)
            continue

        if not is_sqft_basement(row):
            data_df = data_df.drop([row[0]], axis=0)
            continue

        if not is_yr_built(row):
            data_df = data_df.drop([row[0]], axis=0)
            continue
        data_df.at[row[0], 'yr_built'] /= 2015

        if not is_yr_renovated(row):
            data_df = data_df.drop([row[0]], axis=0)
            continue
        data_df.at[row[0], 'yr_renovated'] /= 2015

        if not is_zipcode(row):
            data_df = data_df.drop([row[0]], axis=0)
            continue

        if not is_lat(row):
            data_df = data_df.drop([row[0]], axis=0)
            continue
        data_df.at[row[0], 'lat'] /= 47.8

        if not is_long(row):
            data_df = data_df.drop([row[0]], axis=0)
            continue
        data_df.at[row[0], 'long'] /= -123

        if not is_sqft_living15(row):
            data_df = data_df.drop([row[0]], axis=0)
            continue

        if not is_sqft_lot15(row):
            data_df = data_df.drop([row[0]], axis=0)
            continue

    return data_df


def is_id(row):
    id_num = row[1]['id']
    if is_int_and_not_negative(id_num):
        return 10 ** 6 < id_num < 10 ** 10


def is_date(row):
    date = row[1]['date']
    if isinstance(date, str):
        nums = list(map(int, re.findall(r'\d+', date)))
    else:
        return False
    is_date = None
    date_str = ''
    if len(nums) > 0:
        date_str = str(nums[0])
        try:
            actual_date = parse(date, fuzzy=True)
            year = actual_date.year
            if year == 2014 or year == 2015:
                is_date = True

        except ValueError:
            is_date = False
    if is_date:
        date_format = date_str + 'T000000'
        if date_format.__eq__(date):
            return True
    return False


def is_price(row):
    price = row[1]['price']
    if is_float_and_not_negative(price):
        return 75 * 10 ** 3 <= price <= 8 * 10 ** 6


def is_bedrooms_num(row):
    num = row[1]['bedrooms']
    if is_int_and_not_negative(num):
        return num <= 33


def is_bathrooms_num(row):
    num = row[1]['bathrooms']
    if is_float_and_not_negative(num):
        return num <= 8


def is_sqft_living(row):
    num = row[1]['sqft_living']
    if is_float_and_not_negative(num):
        return 200 <= num <= 14 * 10 ** 3


def is_sqft_lot(row):
    num = row[1]['sqft_lot']
    if is_float_and_not_negative(num):
        return 520 <= num <= 1.7 * 10 ** 6


def is_floors(row):
    num = row[1]['floors']
    if is_float_and_not_negative(num):
        return 1 <= num <= 5


def is_waterfront(row):
    num = row[1]['waterfront']
    if is_int_and_not_negative(num):
        return num <= 1


def is_view(row):
    num = row[1]['view']
    if is_int_and_not_negative(num):
        return num <= 4


def is_condition(row):
    num = row[1]['condition']
    if is_float_and_not_negative(num):
        return num <= 5


def is_grade(row):
    num = row[1]['grade']
    if is_float_and_not_negative(num):
        return 1 <= num <= 13


def is_sqft_above(row):
    num = row[1]['sqft_above']
    if is_float_and_not_negative(num):
        return 200 <= num <= 10 ** 4


def is_sqft_basement(row):
    num = row[1]['sqft_basement']
    if is_float_and_not_negative(num):
        return num <= 4820


def is_yr_built(row):
    num = row[1]['yr_built']
    if is_int_and_not_negative(num):
        return 1900 <= num <= 2015


def is_yr_renovated(row):
    num = row[1]['yr_renovated']
    if is_int_and_not_negative(num):
        return num == 0 or 1900 <= num <= 2015


def is_zipcode(row):
    num = row[1]['zipcode']
    if is_int_and_not_negative(num):
        return 98 * 10 * 3 <= num <= 98.2 * 10 ** 3


def is_lat(row):
    num = row[1]['lat']
    if is_float_and_not_negative(num):
        to_return = 47 <= num <= 48
        return to_return


def is_long(row):
    num = row[1]['long']
    return -123 <= num <= -121


def is_sqft_living15(row):
    num = row[1]['sqft_living15']
    if is_float_and_not_negative(num):
        return 300 <= num <= 7 * 10 ** 3


def is_sqft_lot15(row):
    num = row[1]['sqft_lot15']
    if is_float_and_not_negative(num):
        return 600 <= num <= 880 * 10 ** 3


def plot_singular_values(singular_values):
    singular_values.sort()
    singular_values = singular_values[::-1]
    plt.plot(np.arange(singular_values.shape[0]), singular_values, 'bo-')
    plt.title('Scree Plot')
    plt.xlabel('Index')
    plt.ylabel('Singular Values')
    plt.legend(['Singular Values'])
    plt.show()


def add_ones_col(data_df):
    column_length = len(data_df)
    ones_col = np.ones(column_length)
    data_df.insert(loc=0, column='ones_col', value=ones_col)


def q15(path):
    data_df = load_data(path)
    add_ones_col(data_df)
    data_df = data_df.drop('price', axis=1)
    data_mat = data_df.values.T
    singular_values = np.linalg.svd(data_mat, compute_uv=False)
    plot_singular_values(np.array(singular_values))


def q17(path):
    data_df = load_data(path)
    add_ones_col(data_df)
    data_df = data_df.sample(frac=1)
    response_vec = np.array(data_df['price'])
    data_df = data_df.drop('price', axis=1)
    desin_mat = data_df.values

    rows_amount = desin_mat.shape[0]
    test_mat_size = int(0.25 * rows_amount)
    test_mat = desin_mat[:test_mat_size]
    test_response_vec = response_vec[:test_mat_size]
    train_mat = desin_mat[test_mat_size:]
    train_response_vec = response_vec[test_mat_size:]

    coefficients_vecs_lst = []
    train_mat_rows_amount = train_mat.shape[0]
    for i in range(1, 101):
        current_rows_amount = int(i / 100 * train_mat_rows_amount)
        current_train_mat = train_mat[:current_rows_amount]
        current_response_vec = train_response_vec[:current_rows_amount]
        current_coefficients_vec = \
            fit_linear_regression(current_train_mat.T, current_response_vec)[0]
        coefficients_vecs_lst.append(current_coefficients_vec)

    mse_lst = []
    for i in range(0, 100):
        current_coefficients_vec = coefficients_vecs_lst[i]
        current_prediction_vec = predict(test_mat.T, current_coefficients_vec)
        mse_lst.append(mse(current_prediction_vec, test_response_vec))

    mse_array = np.array(mse_lst)
    percentages_array = np.array(range(1, 101))
    plt.plot(percentages_array, mse_array)
    plt.title('MSE values as a function of\nthe percentage taken from the '
              'data set')
    plt.xlabel('Percentage')
    plt.ylabel('MSE Values')
    plt.legend(['MSE Values'])
    plt.show()


def feature_evaluation(path):
    data_df = load_data(path)
    response_vec = np.array(data_df['price'])
    response_vec_std = np.std(response_vec)
    non_categorical_features = ['lat', 'long', 'price', 'year_2014',
                                'year_2015', 'yr_built', 'yr_renovated']
    months_lst = ['month_' + str(i) for i in range(1, 13)]
    features_to_drop = non_categorical_features + months_lst
    data_df = data_df.drop(features_to_drop, axis=1)
    features = data_df.columns.to_list()
    for feature in features:
        current_feature = np.array(data_df[feature])
        current_feature_std = np.std(current_feature)
        feature_mean = np.mean(current_feature)
        response_mean = np.mean(response_vec)
        feature_from_mean = current_feature - feature_mean
        response_from_mean = response_vec - response_mean
        mul = feature_from_mean * response_from_mean
        cor_numerator = np.mean(mul)
        cor_denominator = current_feature_std * response_vec_std
        current_pearson_correlation = cor_numerator / cor_denominator

        plt.scatter(current_feature, response_vec)
        plt.title('house prices as a function of ' + feature + ' values\n\n'
                  + 'Pearson Correlation = ' + str(current_pearson_correlation))
        plt.xlabel(feature + ' values')
        plt.ylabel('house prices')
        plt.legend(['house prices'])
        plt.show()
