import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle
import scipy.spatial as spatial
from scipy.stats import poisson
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# Function to generate different point patterns
def generate_points(pattern_type, n=100, window_size=1, cluster_std=0.05, min_dist=0.05):
    """
    Generate spatial point patterns.
    
    Parameters:
    -----------
    pattern_type : str
        Type of pattern: 'random', 'clustered', or 'regular'
    n : int
        Number of points to generate
    window_size : float
        Size of the square window (assumes 0 to window_size in both x and y)
    cluster_std : float
        Standard deviation for cluster generation (for clustered pattern)
    min_dist : float
        Minimum distance between points (for regular pattern)
        
    Returns:
    --------
    points : numpy.ndarray
        Array of point coordinates with shape (n, 2)
    """
    if pattern_type == 'random':
        # Complete Spatial Randomness (CSR)
        points = np.random.uniform(0, window_size, size=(n, 2))
        
    elif pattern_type == 'clustered':
        # Clustered pattern (simplified Neyman-Scott process)
        num_clusters = int(n / 10)  # Create approximately 10 points per cluster
        cluster_centers = np.random.uniform(0, window_size, size=(num_clusters, 2))
        
        # Initialize empty array for points
        points = np.zeros((n, 2))
        
        # Generate points around each cluster center
        points_per_cluster = n // num_clusters
        for i in range(num_clusters):
            start_idx = i * points_per_cluster
            end_idx = start_idx + points_per_cluster if i < num_clusters - 1 else n
            num_points = end_idx - start_idx
            
            # Generate points around this cluster center
            cluster_points = np.random.normal(loc=cluster_centers[i], scale=cluster_std, size=(num_points, 2))
            
            # Ensure points are within the window
            cluster_points = np.clip(cluster_points, 0, window_size)
            points[start_idx:end_idx] = cluster_points
            
    elif pattern_type == 'regular':
        # Regular pattern (simple inhibition process)
        points = np.zeros((n, 2))
        
        # Add first point randomly
        points[0] = np.random.uniform(0, window_size, size=2)
        
        # Add remaining points with minimum distance constraint
        current_n = 1
        max_attempts = 1000
        
        while current_n < n:
            # Generate candidate point
            candidate = np.random.uniform(0, window_size, size=2)
            
            # Check distance to all existing points
            valid_point = True
            attempts = 0
            
            while attempts < max_attempts:
                distances = np.sqrt(np.sum((points[:current_n] - candidate)**2, axis=1))
                if np.min(distances) < min_dist:
                    # Too close to an existing point, generate a new candidate
                    candidate = np.random.uniform(0, window_size, size=2)
                    attempts += 1
                else:
                    # Point is valid
                    valid_point = True
                    break
            
            if valid_point:
                points[current_n] = candidate
                current_n += 1
            
            # If we've tried too many times, reduce the minimum distance constraint
            if attempts == max_attempts:
                min_dist *= 0.9
    
    else:
        raise ValueError("Pattern type must be 'random', 'clustered', or 'regular'")
    
    return points

# Function to generate multi-scale pattern
def generate_multiscale_pattern(n=200, window_size=1):
    """
    Generate a point pattern with clustering at multiple scales.
    
    Parameters:
    -----------
    n : int
        Number of points to generate
    window_size : float
        Size of the square window
        
    Returns:
    --------
    points : numpy.ndarray
        Array of point coordinates with shape (n, 2)
    """
    # Create large-scale clusters
    num_large_clusters = 4
    large_cluster_centers = np.array([
        [0.25, 0.25],
        [0.25, 0.75],
        [0.75, 0.25],
        [0.75, 0.75]
    ])
    
    # Initialize points array
    points = np.zeros((n, 2))
    points_per_large_cluster = n // num_large_clusters
    
    for i in range(num_large_clusters):
        start_idx = i * points_per_large_cluster
        end_idx = start_idx + points_per_large_cluster if i < num_large_clusters - 1 else n
        cluster_size = end_idx - start_idx
        
        # Create small-scale clusters within each large cluster
        num_small_clusters = 3
        # Create small cluster centers around the large cluster center
        small_cluster_centers = np.random.normal(
            loc=large_cluster_centers[i], 
            scale=0.1, 
            size=(num_small_clusters, 2)
        )
        # Ensure they're within the window
        small_cluster_centers = np.clip(small_cluster_centers, 0, window_size)
        
        # Assign points to small clusters
        points_per_small_cluster = cluster_size // num_small_clusters
        for j in range(num_small_clusters):
            small_start_idx = start_idx + j * points_per_small_cluster
            small_end_idx = small_start_idx + points_per_small_cluster if j < num_small_clusters - 1 else end_idx
            small_cluster_size = small_end_idx - small_start_idx
            
            # Generate points around small cluster center
            small_cluster_points = np.random.normal(
                loc=small_cluster_centers[j],
                scale=0.02,
                size=(small_cluster_size, 2)
            )
            
            # Ensure points are within the window
            small_cluster_points = np.clip(small_cluster_points, 0, window_size)
            points[small_start_idx:small_end_idx] = small_cluster_points
    
    return points

# Function to calculate Ripley's K function with edge correction
def ripley_k(points, r_values, window_size=1, edge_correction=True):
    """
    Calculate Ripley's K function for a set of points.
    
    Parameters:
    -----------
    points : numpy.ndarray
        Array of point coordinates with shape (n, 2)
    r_values : numpy.ndarray
        Array of radius values at which to evaluate K
    window_size : float
        Size of the square window (assumes 0 to window_size in both x and y)
    edge_correction : bool
        Whether to apply edge correction
        
    Returns:
    --------
    k_values : numpy.ndarray
        K function values for each radius
    l_values : numpy.ndarray
        L function values for each radius
    """
    n = points.shape[0]
    area = window_size ** 2
    intensity = n / area
    
    # Calculate pairwise distances
    distances = spatial.distance.pdist(points)
    distances = spatial.distance.squareform(distances)
    
    k_values = np.zeros_like(r_values)
    
    for i, r in enumerate(r_values):
        # Count points within radius r of each point
        counts = np.sum(distances < r, axis=1) - 1  # Subtract 1 to exclude self
        
        if edge_correction:
            # Apply border edge correction
            # For each point, calculate proportion of circle within window
            proportions = np.ones(n)
            for j in range(n):
                x, y = points[j]
                # Proportion of circle area within window
                # This is a simplified edge correction for a square window
                min_dist_to_border = min(x, y, window_size - x, window_size - y)
                if min_dist_to_border < r:
                    # Calculate proportion of circle inside window
                    # This is an approximation based on distance to border
                    overlap_prop = (min_dist_to_border / r) ** 2
                    proportions[j] = max(0.25, overlap_prop)  # Minimum 25% overlap
            
            # Apply correction
            counts = counts / proportions
        
        # Calculate K
        k_values[i] = area * np.sum(counts) / (n * (n - 1))
    
    # Calculate L function (normalized version of K)
    l_values = np.sqrt(k_values / np.pi) - r_values
    
    return k_values, l_values

def main():
    # Generate point patterns
    window_size = 1
    n_points = 100
    random_points = generate_points('random', n=n_points, window_size=window_size)
    clustered_points = generate_points('clustered', n=n_points, window_size=window_size)
    regular_points = generate_points('regular', n=n_points, window_size=window_size, min_dist=0.07)

    # Define radius values for K function
    r_values = np.linspace(0.01, 0.3, 30)

    # Calculate theoretical K for CSR
    k_theo = np.pi * r_values**2

    # Calculate K and L for each pattern
    k_random, l_random = ripley_k(random_points, r_values, window_size)
    k_clustered, l_clustered = ripley_k(clustered_points, r_values, window_size)
    k_regular, l_regular = ripley_k(regular_points, r_values, window_size)

    # Plot the point patterns
    plt.figure(figsize=(15, 5))

    # Random pattern
    plt.subplot(1, 3, 1)
    plt.scatter(random_points[:, 0], random_points[:, 1], s=10)
    plt.title('Complete Spatial Randomness')
    plt.xlim(0, window_size)
    plt.ylim(0, window_size)
    plt.gca().set_aspect('equal')

    # Clustered pattern
    plt.subplot(1, 3, 2)
    plt.scatter(clustered_points[:, 0], clustered_points[:, 1], s=10)
    plt.title('Clustered Pattern')
    plt.xlim(0, window_size)
    plt.ylim(0, window_size)
    plt.gca().set_aspect('equal')

    # Regular pattern
    plt.subplot(1, 3, 3)
    plt.scatter(regular_points[:, 0], regular_points[:, 1], s=10)
    plt.title('Regular Pattern')
    plt.xlim(0, window_size)
    plt.ylim(0, window_size)
    plt.gca().set_aspect('equal')

    plt.tight_layout()
    plt.savefig('point_patterns.png', dpi=300)
    plt.close()

    # Plot K functions
    plt.figure(figsize=(12, 5))

    # K function
    plt.subplot(1, 2, 1)
    plt.plot(r_values, k_theo, 'k-', label='Theoretical (CSR)')
    plt.plot(r_values, k_random, 'b--', label='Random')
    plt.plot(r_values, k_clustered, 'r--', label='Clustered')
    plt.plot(r_values, k_regular, 'g--', label='Regular')
    plt.xlabel('Distance (r)')
    plt.ylabel('K(r)')
    plt.title('Ripley\'s K Function')
    plt.legend()

    # L function
    plt.subplot(1, 2, 2)
    plt.axhline(y=0, color='k', linestyle='-', label='Theoretical (CSR)')
    plt.plot(r_values, l_random, 'b--', label='Random')
    plt.plot(r_values, l_clustered, 'r--', label='Clustered')
    plt.plot(r_values, l_regular, 'g--', label='Regular')
    plt.xlabel('Distance (r)')
    plt.ylabel('L(r) - r')
    plt.title('Ripley\'s L Function')
    plt.legend()

    plt.tight_layout()
    plt.savefig('ripley_k_plot.png', dpi=300)
    plt.close()

    # Visual aid for interpreting K and L functions
    plt.figure(figsize=(9, 6))

    # Add explanation
    plt.text(0.5, 0.95, 'Interpreting Ripley\'s K and L Functions', ha='center', fontsize=16, fontweight='bold')
    plt.text(0.5, 0.9, 'L(r) - r > 0: Clustering at distance r', ha='center', fontsize=12, color='red')
    plt.text(0.5, 0.85, 'L(r) - r = 0: Complete Spatial Randomness (CSR)', ha='center', fontsize=12, color='black')
    plt.text(0.5, 0.8, 'L(r) - r < 0: Regularity/Dispersion at distance r', ha='center', fontsize=12, color='green')

    # Example visualization
    x = np.linspace(0, 1, 100)
    y_csr = np.zeros_like(x)
    y_clustered = 0.1 * np.exp(-((x - 0.5) ** 2) / 0.05)
    y_regular = -0.05 * np.sin(np.pi * x * 10) - 0.03

    plt.plot(x, y_csr, 'k-', linewidth=2, label='CSR')
    plt.plot(x, y_clustered, 'r-', linewidth=2, label='Clustered')
    plt.plot(x, y_regular, 'g-', linewidth=2, label='Regular')

    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.legend(loc='lower right')
    plt.xlabel('Distance (r)')
    plt.ylabel('L(r) - r')
    plt.title('Schematic Representation of L Function Interpretation')

    # Customize
    plt.grid(alpha=0.3)
    sns.despine()
    plt.tight_layout()
    plt.savefig('ripley_k_interpretation.png', dpi=300)
    plt.close()

    # NEW: Generate and analyze multi-scale clustered pattern
    n_multiscale = 200
    multiscale_points = generate_multiscale_pattern(n=n_multiscale, window_size=window_size)
    
    # Use more r values with wider range to capture both scales
    r_values_multiscale = np.linspace(0.01, 0.5, 50)
    k_theo_multiscale = np.pi * r_values_multiscale**2
    
    # Calculate K and L for multiscale pattern
    k_multiscale, l_multiscale = ripley_k(multiscale_points, r_values_multiscale, window_size)
    
    # Create figure to display the multi-scale analysis
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot the point pattern
    ax[0, 0].scatter(multiscale_points[:, 0], multiscale_points[:, 1], s=10, color='purple')
    ax[0, 0].set_title('Multi-scale Clustered Pattern')
    ax[0, 0].set_xlim(0, window_size)
    ax[0, 0].set_ylim(0, window_size)
    ax[0, 0].set_aspect('equal')
    
    # Plot the K function
    ax[0, 1].plot(r_values_multiscale, k_theo_multiscale, 'k-', label='Theoretical (CSR)')
    ax[0, 1].plot(r_values_multiscale, k_multiscale, 'purple', label='Multi-scale')
    ax[0, 1].set_xlabel('Distance (r)')
    ax[0, 1].set_ylabel('K(r)')
    ax[0, 1].set_title('Ripley\'s K Function')
    ax[0, 1].legend()
    
    # Plot the L function
    ax[1, 0].axhline(y=0, color='k', linestyle='-', label='Theoretical (CSR)')
    ax[1, 0].plot(r_values_multiscale, l_multiscale, 'purple', label='Multi-scale')
    ax[1, 0].set_xlabel('Distance (r)')
    ax[1, 0].set_ylabel('L(r) - r')
    ax[1, 0].set_title('Ripley\'s L Function')
    ax[1, 0].legend()
    
    # Highlight the different scales
    # Find peaks in the L function
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(l_multiscale, height=0.02)
    
    # Create derivative of L function to identify changes in clustering intensity
    l_derivative = np.gradient(l_multiscale, r_values_multiscale)
    
    # Plot the derivative of L function
    ax[1, 1].plot(r_values_multiscale, l_derivative, 'purple')
    ax[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax[1, 1].set_xlabel('Distance (r)')
    ax[1, 1].set_ylabel('d/dr [L(r) - r]')
    ax[1, 1].set_title('Rate of Change in L Function')
    
    # Annotate the scales
    for idx in peaks:
        r = r_values_multiscale[idx]
        l_val = l_multiscale[idx]
        ax[1, 0].plot(r, l_val, 'ro')
        ax[1, 0].annotate(f'Scale: {r:.2f}', xy=(r, l_val), 
                         xytext=(r+0.05, l_val),
                         arrowprops=dict(arrowstyle='->'))
    
    plt.tight_layout()
    plt.savefig('multiscale_analysis.png', dpi=300)
    plt.close()

    print("Analysis complete. Visualizations saved.")

if __name__ == "__main__":
    main() 