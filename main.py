import numpy as np
from fontTools.otlLib import optimize
from sympy import *
from scipy import optimize
import matplotlib.pyplot as plt


def get_payment_dates(d):
    remaining = d - 182.5
    dates = [d]
    while remaining > 0:
        dates.append(remaining)
        remaining = remaining - 182.5
    return dates


def compute_yield(mat, cpn, dp, i, yields):
    payment_dates = get_payment_dates(mat[i])
    p = dp
    n = len(payment_dates)
    for j in range(n - 1):
        cpn_date = payment_dates[n-j-1]
        m = mat[j]
        if m + 20 > cpn_date > m - 20:
            r_interpolate = yields[j]
        elif j == 3:
            r_interpolate = yields[3]
        else:
            r2 = yields[j+1]
            r1 = yields[j]
            d = mat[j+1]
            frac = (d-cpn_date)/365
            r_interpolate = frac*r2 + (1-frac)*r1
        p = p - (cpn/2)*exp(-r_interpolate*cpn_date/365)
    r = -log(p/(100+cpn/2))/(mat[i]/365)
    return r


def compute_forward_price(mat, cpn, dp, i, yields):
    payment_dates = get_payment_dates(mat[i])
    p = dp
    n = len(payment_dates)
    for j in range(n - 1):
        cpn_date = payment_dates[n - j - 1]
        m = mat[j]
        if m + 20 > cpn_date > m - 20:
            r_interpolate = yields[j]
        elif j == 3:
            r_interpolate = yields[3]
        else:
            r2 = yields[j + 1]
            r1 = yields[j]
            d = mat[j + 1]
            frac = (d - cpn_date) / 365
            r_interpolate = frac * r2 + (1 - frac) * r1
        p = p - (cpn/2)*exp(-r_interpolate*cpn_date/365)
    f = -log(p/(100+cpn/2))
    return f


def compute_forward_curve_int(mat, cpns, dirty_prices, yields):
    forwards_int = []
    for i in range(10):
        forwards_int.append(compute_forward_price(mat, cpns[i], dirty_prices[i], i, yields))
    return forwards_int


def compute_forward(f, mat):
    indexes = [2, 4, 6, 8]
    forward = []
    for i in indexes:
        forward.append((f[i]-f[i-1])/((mat[i]-mat[i-1])/365))
    return forward


def compute_yield_curve(mat, cpns, dirty_prices):
    yields = []
    for i in range(10):
        yields.append(compute_yield(mat, cpns[i], dirty_prices[i], i, yields))
    return yields


def get_dirty_prices(cpns, prices, date):
    t_last_cpn = [42947, 42947, 42947, 42947, 42978, 42978, 42978, 42978, 42978, 42978]
    dirty_prices = []
    for i in range(10):
        dirty_prices.append(((date - t_last_cpn[i]) / 365) * cpns[i] + prices[i])
    return dirty_prices


def get_prices(date_index, data):
    index = date_index + 3
    prices = []
    for i in range(len(data)):
        prices.append(data[i][index])
    return prices


def update_r(dp, payoffs, t, r):
    f_r = -dp
    f_prime_r = 0
    for i in range(len(payoffs)):
        f_r += payoffs[i]*exp(-r*(t+i/2))
        f_prime_r += -payoffs[i]*(t+i/2)*exp(-r*(t+i/2))
    return r - (f_r/f_prime_r)


def get_ytm(dp, payoffs, time, iterations):
    r = 0.01
    for i in range(iterations):
        r = update_r(dp, payoffs, time, r)
    return r


def compute_ytms(dirty_prices, payoffs, date):
    ytms = []
    iterations = 20
    for i in range(10):
        if i == 0 or i == 1 or i == 2 or i == 3:
            time = (43131 - date)/365
        else:
            time = (43159 - date)/365
        ytms.append(get_ytm(dirty_prices[i], payoffs[i], time, iterations))
    return ytms


def get_interpolated_yields(y):
    y1 = 22/184*y[1]+162/184*y[2]
    y2 = 50/212*y[3]+162/212*y[4]
    y3 = 131/181*y[5]+50/181*y[6]
    y4 = 131/181*y[7]+50/181*y[8]
    y5 = y[9]
    return [y1, y2, y3, y4, y5]


def get_payoffs(cpns):
    coupon_payments = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    payoffs = []
    for i in range(10):
        payoff = []
        for j in range(i+1):
            if j == i:
                payoff.append(100 + cpns[i]/2)
            else:
                payoff.append(cpns[i]/2)
        payoffs.append(payoff)
    return payoffs


def get_time_to_mat(data, d):
    time_to_mat = []
    for i in range(10):
        time_to_mat.append(data[i][1]-d)
    return time_to_mat


if __name__ == '__main__':
    dates = [43109, 43110, 43111, 43112, 43113, 43116, 43117, 43118, 43119, 43120]
    coupons = [1.5, 0.25, 0.25, 0.25, 2.25, 1.5, 1.25, 0.5, 0.25, 1]
    payoffs = get_payoffs(coupons)
    bond_data = [["22-Feb", 43131, 1.50, 100.07, 100.06, 100.06, 100.05, 100.04, 100.04, 100.03, 100.03, 100.02, 100.02],
                 ["22-Aug", 43312, 0.25, 99.8, 99.79, 99.78, 99.77, 99.76, 99.72, 99.7, 99.7, 99.72, 99.74],
                 ["23-Feb", 43496, 0.25, 99.31, 99.26, 99.29, 99.28, 99.26, 99.18, 99.12, 99.1, 99.15, 99.2],
                 ["23-Aug", 43677, 0.25, 98.81, 98.8, 98.78, 98.76, 98.74, 98.61, 98.55, 98.55, 98.56, 98.61],
                 ["24-Mar", 43890, 2.25, 102.31, 102.31, 102.26, 102.22, 102.18, 101.99, 101.91, 101.9, 101.88, 101.97],
                 ["24-Sep",	44074, 1.50, 100.56, 100.63, 100.62, 100.58, 100.52, 100.29, 100.21, 100.15, 100.17, 100.27],
                 ["25-Mar", 44255, 1.25, 99.64, 99.67, 99.61, 99.59, 99.49, 99.25, 99.12, 99.11, 99.1, 99.26],
                 ["25-Sep", 44439, 0.50, 96.66, 96.72, 96.77, 96.77, 96.68, 96.43, 96.26, 96.15, 96.19, 96.31],
                 ["26-Mar", 44620, 0.25, 95.11, 95.17, 95.12, 95.14, 95.04, 94.75, 94.53, 94.5, 94.52, 94.7],
                 ["26-Sep",	44804, 1.00, 97.66, 97.7, 97.67, 97.69, 97.58, 97.24, 97.02, 96.96, 97, 97.23]]
    # Q4 a)
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    date_names = ["Jan 10", "Jan 11", "Jan 12", "Jan 13", "Jan 14", "Jan 17", "Jan 18", "Jan 19", "Jan 20", "Jan 21"]
    times = [21/365, 203/365, 21/365 + 1, 203/365 + 1, 50/365 + 2, 233/365 + 2, 50/365 + 3, 233/365 + 3, 50/365 + 4, 233/365 + 4]
    ytms = []
    for date_index in range(10):
        date = dates[date_index]
        prices = get_prices(date_index, bond_data)
        dirty_price = get_dirty_prices(coupons, prices, date)
        yields = compute_ytms(dirty_price, payoffs, date)
        ytms.append(get_interpolated_yields(yields))
        ax1.plot(times, yields, label=date_names[date_index])
    plt.xlabel("Number of Years")
    plt.ylabel("Yield")
    plt.suptitle("5 Year Canadian Bond Yield Curve")
    plt.legend(loc='upper left', prop={'size': 6})
    plt.show()
    # Q4 b)
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    for date_index in range(10):
        date = dates[date_index]
        prices = get_prices(date_index, bond_data)
        dirty_price = get_dirty_prices(coupons, prices, date)
        maturities = get_time_to_mat(bond_data, date)
        spot = compute_yield_curve(maturities, coupons, dirty_price)
        ax2.plot(times, spot, label=date_names[date_index])
    plt.xlabel("Number of Years")
    plt.ylabel("Spot Rate")
    plt.suptitle("5 Year Canadian Spot Rate Curve")
    plt.legend(loc='upper left', prop={'size': 6})
    plt.show()
    # Q4 c)
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    f = []
    for date_index in range(10):
        years = [1, 2, 3, 4]
        date = dates[date_index]
        prices = get_prices(date_index, bond_data)
        dirty_price = get_dirty_prices(coupons, prices, date)
        maturities = get_time_to_mat(bond_data, date)
        spot = compute_yield_curve(maturities, coupons, dirty_price)
        forward_int = compute_forward_curve_int(maturities, coupons, dirty_price, spot)
        forwards = compute_forward(forward_int, maturities)
        f.append(forwards)
        ax3.plot(years, forwards, label=date_names[date_index])
    plt.xlabel("Number of Years")
    plt.ylabel("Forward Rate")
    plt.suptitle("5 Year Canadian Forward Rate Curve")
    plt.legend(loc='upper left', prop={'size': 6})
    plt.show()

    # Q5
    r1 = []
    r2 = []
    r3 = []
    r4 = []
    r5 = []
    test1 = []
    test2 = []
    for i in range(len(ytms)-1):
        test1.append(ytms[i][0])
        test2.append(ytms[i][4])
        r1.append(log(ytms[i+1][0]/ytms[i][0]))
        r2.append(log(ytms[i+1][1]/ytms[i][1]))
        r3.append(log(ytms[i+1][2]/ytms[i][2]))
        r4.append(log(ytms[i+1][3]/ytms[i][3]))
        r5.append(log(ytms[i+1][4]/ytms[i][4]))
    X = np.array([r1, r2, r3, r4, r5])
    cov_r = np.cov(X.astype(float))

    f1 = []
    f2 = []
    f3 = []
    f4 = []
    for i in range(len(f)-1):
        f1.append(log(f[i+1][0]/f[i][0]))
        f2.append(log(f[i + 1][1] / f[i][1]))
        f3.append(log(f[i + 1][2] / f[i][2]))
        f4.append(log(f[i + 1][3] / f[i][3]))
    F = np.array([f1, f2, f3, f4])
    cov_f = np.cov(F.astype(float))

    # Q6
    r_eigenvalues, r_eigenvectors = np.linalg.eig(cov_r)
    f_eigenvalues, f_eigenvectors = np.linalg.eig(cov_f)
    print(r_eigenvectors)
    print(r_eigenvalues)
    print(f_eigenvectors)
    print(f_eigenvalues)
    print("Leading eigenvector and value for the log returns of yield")
    print(r_eigenvectors[0])
    print(r_eigenvalues[0])
    print("Leading eigenvector and value for the log returns of forwards")
    print(f_eigenvectors[0])
    print(f_eigenvalues[0])
