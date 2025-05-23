from matplotlib import pyplot as plt
import numpy as np


## Changes 
# 1. rename unicode variables to readable ascii
# 2. move `distance` to a method 
# 3. Vectorize for loop

def train(input_data, n_max_iterations, width, height):
    sigma_0 = max(width, height) / 2
    alpha_0 = 0.1
    weights = np.random.random((width, height, 3))
    time_const = n_max_iterations / np.log(sigma_0)
    
    x = np.arange(width)
    y = np.arange(height)
    x_coords, y_coords = np.meshgrid(x, y, indexing='ij')
    for t in range(n_max_iterations):
        sigma_t = sigma_0 * np.exp(-t/time_const)
        alpha_t = alpha_0 * np.exp(-t/time_const)
        for vt in input_data:
            bmu = np.argmin(np.sum((weights - vt) ** 2, axis=2))
            bmu_x, bmu_y = np.unravel_index(bmu, (width, height))
                    
            di = distance(bmu_x, bmu_y, x_coords, y_coords)
            theta_t = np.exp(-(di ** 2) / (2*(sigma_t ** 2)))
            weights += alpha_t * theta_t[:, :, np.newaxis] * (vt - weights)

    return weights

def distance(bmu_x, bmu_y, x, y):
    di = np.sqrt(((x - bmu_x) ** 2) + ((y - bmu_y) ** 2))
    return di


if __name__ == '__main__':
    import timeit
    # Generate data
    input_data = np.random.random((10,3))
    start_time = timeit.default_timer()
    image_data = train(input_data, 100, 10, 10)
    end_time = timeit.default_timer()
    print(f"Duration 1: {end_time - start_time}s")
    start_time = end_time
    plt.imsave('100.png', image_data)

    # Generate data
    image_data = train(input_data, 1000, 100, 100)
    end_time = timeit.default_timer()
    print(f"Duration 2: {end_time - start_time}s")

    plt.imsave('1000.png', image_data)