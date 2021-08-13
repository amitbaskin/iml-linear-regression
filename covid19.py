import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def fit_linear_regression(design_mat, response_vec):
    singular_values = np.linalg.svd(design_mat, compute_uv=False)
    coefficients_vec = np.linalg.pinv(design_mat.T) @ response_vec
    return coefficients_vec, singular_values


def load_data(path):
    os.chdir(os.path.dirname(path))
    return pd.read_csv(os.path.basename(path))


def add_log_detected_col(path):
    data_df = load_data(path)
    data_df['log_detected'] = np.log(data_df['detected'])
    return data_df


def fit_covid(data_df):
    column_length = len(data_df)
    ones_col = np.ones(column_length)
    data_df.insert(loc=0, column='ones_col', value=ones_col)

    design_mat = data_df[['ones_col', 'day_num']].values.T
    response_vec = data_df['log_detected'].T
    coefficients_vec = fit_linear_regression(design_mat, response_vec)[0]
    day_num_vec = np.array(data_df['day_num'])
    linear_prediction_vec = design_mat.T @ coefficients_vec

    plt.plot(day_num_vec, linear_prediction_vec)
    plt.scatter(day_num_vec, response_vec)
    plt.title('log of the number of detected cases\n'
              'with the corona virus in Israel\n'
              'as a function of the number of days\n'
              'since the first case detected')
    plt.xlabel('number of days')
    plt.ylabel('log of the number of detected')
    plt.legend(['prediction for the log of the number of detected',
                'actual log of the number of detected'])
    plt.show()

    zero_coef = coefficients_vec[0]
    first_coef = coefficients_vec[1]
    exponential_prediction_vec = np.exp(first_coef * day_num_vec) * np.exp(
         zero_coef)

    plt.plot(day_num_vec, exponential_prediction_vec)
    plt.scatter(day_num_vec, np.exp(response_vec))
    plt.title('the number of detected cases\n'
              'with the corona virus in Israel\n'
              'as a function of the number of days\n'
              'since the first case detected')
    plt.xlabel('number of days')
    plt.ylabel('number of detected')
    plt.legend(['prediction for the number of detected',
                'actual number of detected'])
    plt.show()
