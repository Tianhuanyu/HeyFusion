import numpy as np

def generate_sigma_points(x, P, kappa):
    n = len(x)
    sigma_points = np.zeros((2 * n + 1, n))
    sigma_points[0] = x
    U = np.linalg.cholesky((n + kappa) * P)
    for i in range(n):
        sigma_points[i + 1] = x + U[i]
        sigma_points[n + i + 1] = x - U[i]
    return sigma_points

def predict_sigma_points(sigma_points, dt):
    for i in range(len(sigma_points)):
        sigma_points[i][0] += sigma_points[i][1] * dt
    return sigma_points

def predict_mean_and_covariance(sigma_points, Wm, Wc, Q):
    n = sigma_points.shape[1]
    x_pred = np.dot(Wm, sigma_points)
    P_pred = Q.copy()
    for i in range(len(sigma_points)):
        y = sigma_points[i] - x_pred
        P_pred += Wc[i] * np.outer(y, y)
    return x_pred, P_pred

def predict_measurement(sigma_points, Wm, Wc, R):
    z_pred = np.dot(Wm, sigma_points[:, 0])
    P_zz = R.copy()
    for i in range(len(sigma_points)):
        y = sigma_points[i][0] - z_pred
        P_zz += Wc[i] * y * y
    return z_pred, P_zz

def update_state(x_pred, P_pred, z, sigma_points, Wc, z_pred, P_zz, R):
    n = len(x_pred)
    P_xz = np.zeros((n, 1))
    for i in range(len(sigma_points)):
        P_xz += Wc[i] * np.outer(sigma_points[i] - x_pred, sigma_points[i][0] - z_pred)
    K = P_xz / P_zz
    x_updated = x_pred + K * (z - z_pred)
    P_updated = P_pred - K * P_zz * K.T
    return x_updated, P_updated

def unscented_kalman_filter(x, P, z, Q, R, dt, kappa):
    n = len(x)
    Wm = np.full(2 * n + 1, 0.5 / (n + kappa))
    Wc = np.full(2 * n + 1, 0.5 / (n + kappa))
    Wm[0] = kappa / (n + kappa)
    Wc[0] = Wm[0]
    sigma_points = generate_sigma_points(x, P, kappa)
    print("Sigma points:", sigma_points)
    sigma_points = predict_sigma_points(sigma_points, dt)
    x_pred, P_pred = predict_mean_and_covariance(sigma_points, Wm, Wc, Q)
    z_pred, P_zz = predict_measurement(sigma_points, Wm, Wc, R)
    x_updated, P_updated = update_state(x_pred, P_pred, z, sigma_points, Wc, z_pred, P_zz, R)
    return x_updated, P_updated

if __name__ == '__main__':
    x = np.array([0.0, 1.0])
    P = np.eye(2)
    Q = np.eye(2) * 0.1
    R = np.eye(1)
    dt = 1.0
    kappa = 0.0
    z = np.array([0.5])
    x_updated, P_updated = unscented_kalman_filter(x, P, z, Q, R, dt, kappa)
    print("Updated state:", x_updated)
    print("Updated covariance:", P_updated)
