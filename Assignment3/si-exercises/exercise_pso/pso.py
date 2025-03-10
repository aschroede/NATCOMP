import numpy as np
from PIL import Image
import random
import matplotlib.pyplot as plt


"""
in this script we will compare PSO with k = 8 and differing number of particles 
to a K-means baseline with k =8 and plot the resulting Sum of Squared Distances (SSD)
over iterations.
"""

def quantize_image(image_array, palette):
    #given array of pixels and palette (Kx3), return quantized pixel array

    M = image_array.shape[0]
    K = palette.shape[0]
    quantized_pixels = np.zeros_like(image_array)
    
    total_error = 0.0
    for i in range(M):
        pixel = image_array[i]
        dists = np.sum((pixel - palette)**2, axis=1)
        nearest_idx = np.argmin(dists)
        quantized_pixels[i] = palette[nearest_idx]
        total_error += dists[nearest_idx]
    
    return quantized_pixels, total_error

def compute_fitness(image_array, palette):
    _, dist = quantize_image(image_array, palette)
    return dist



def run_kmeans(image_array, K=8, max_iter=20, random_state=42):
    np.random.seed(random_state)
    
    M = image_array.shape[0]
    #init centroids
    init_indices = np.random.choice(M, K, replace=False)
    centroids = image_array[init_indices].copy()
    
    sse_curve = []
    
    for it in range(max_iter):
        #assign pixels to centroid
        distances = np.sum((image_array[:, None, :] - centroids[None, :, :])**2, axis=2)
        cluster_labels = np.argmin(distances, axis=1)
        
        #compute sse
        sse = 0.0
        for i in range(M):
            sse += np.sum((image_array[i] - centroids[cluster_labels[i]])**2)
        sse_curve.append(sse)
        
        for k in range(K):
            points_in_cluster = image_array[cluster_labels == k]
            if len(points_in_cluster) > 0:
                centroids[k] = np.mean(points_in_cluster, axis=0)
    return centroids, sse_curve


class PSOColorQuantizer:
    def __init__(self, image_array, K=8, num_particles=20, omega=0.7, alpha1=1.4, alpha2=1.4, max_iter=100, seed=42):
        np.random.seed(seed)
        random.seed(seed)
        
        self.image_array = image_array
        self.M = image_array.shape[0]
        self.K = K
        self.dim = 3 * K
        self.num_particles = num_particles
        
        #hyperparams
        self.omega = omega
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.max_iter = max_iter
        
        #init positions/velocities
        self.positions = []
        self.velocities = []
        
        for _ in range(num_particles):
            pos = np.random.uniform(low=0, high=255, size=self.dim)
            vel = np.zeros(self.dim)
            
            self.positions.append(pos)
            self.velocities.append(vel)
        
        self.positions = np.array(self.positions)
        self.velocities = np.array(self.velocities)
        
        #personal best
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_fitness = np.array([np.inf]*num_particles)
        
        #global best
        self.global_best_position = None
        self.global_best_fitness = np.inf
        
        self._evaluate_initial_population()
    
    def _evaluate_initial_population(self):
        for i in range(self.num_particles):
            fitness_i = self._fitness(self.positions[i])
            if fitness_i < self.personal_best_fitness[i]:
                self.personal_best_fitness[i] = fitness_i
                self.personal_best_positions[i] = self.positions[i].copy()
            if fitness_i < self.global_best_fitness:
                self.global_best_fitness = fitness_i
                self.global_best_position = self.positions[i].copy()
    
    def _fitness(self, particle):
        palette = particle.reshape((self.K, 3))
        return compute_fitness(self.image_array, palette)
    
    def optimize(self):
        best_fitness_over_time = []
        
        for it in range(self.max_iter):
            #update velocities/positions
            for i in range(self.num_particles):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                
                cognitive = self.alpha1 * r1 * (self.personal_best_positions[i] - self.positions[i])
                social    = self.alpha2 * r2 * (self.global_best_position - self.positions[i])
                
                self.velocities[i] = (self.omega * self.velocities[i]) + cognitive + social
                self.positions[i] += self.velocities[i]
                
                #clip to [0,255]
                np.clip(self.positions[i], 0, 255, out=self.positions[i])
            
            #new positions
            for i in range(self.num_particles):
                fitness_i = self._fitness(self.positions[i])
                
                #new personal best
                if fitness_i < self.personal_best_fitness[i]:
                    self.personal_best_fitness[i] = fitness_i
                    self.personal_best_positions[i] = self.positions[i].copy()
                
                #new global best
                if fitness_i < self.global_best_fitness:
                    self.global_best_fitness = fitness_i
                    self.global_best_position = self.positions[i].copy()
            
            best_fitness_over_time.append(self.global_best_fitness)
            # print(f"iter {it}, global best: {self.global_best_fitness:.4f}")
        
        best_palette = self.global_best_position.reshape((self.K, 3))
        return best_palette, best_fitness_over_time


def main():
    img = Image.open("image.png").convert("RGB")
    img_array = np.array(img, dtype=np.float32).reshape((-1, 3))
    
    K = 8
    max_iter_pso = 100
    max_iter_kmeans = 20
    
    print("Running K-means...")
    kmeans_centroids, kmeans_curve = run_kmeans(
        image_array=img_array,
        K=K,
        max_iter=max_iter_kmeans,
        random_state=42
    )
    print("K-means final SSE:", kmeans_curve[-1])
    
    particle_list = [4, 8, 16, 20]
    pso_curves = {}
    
    for p in particle_list:
        print(f"Running PSO with {p} particles...")
        pso_q = PSOColorQuantizer(image_array=img_array, K=K, num_particles=p, omega=0.7, alpha1=1.4, alpha2=1.4, max_iter=max_iter_pso, seed=42)
        best_palette, fitness_curve = pso_q.optimize()
        pso_curves[p] = fitness_curve
        print(f"PSO ({p} particles) final fitness: {fitness_curve[-1]}")
    
    plt.figure(figsize=(8, 6))
    plt.plot(kmeans_curve, label="K-means (K=8)", linewidth=2)
    for p in particle_list:
        plt.plot(pso_curves[p], label=f"PSO (p={p})", alpha=0.8)
    
    plt.title("K-means vs. PSO (Different Particle Counts) Convergence")
    plt.xlabel("Iteration")
    plt.ylabel("Sum of Squared Distances (Fitness)")
    plt.legend()
    plt.grid(True)
    plt.savefig("comparison_convergence.png")
    plt.show()
    
    

    quantized_arr_km, _ = quantize_image(img_array, kmeans_centroids)
    quantized_img_km = quantized_arr_km.reshape((img.height, img.width, 3)).astype(np.uint8)
    out_img_km = Image.fromarray(quantized_img_km, mode="RGB")
    out_img_km.save("kmeans_quantized.png")
    out_img_km.show()
    
    best_palette_20 = pso_curves[20]
    pso_q_20 = PSOColorQuantizer(img_array, K=K, num_particles=20, max_iter=max_iter_pso, seed=42)
    best_palette_20, _ = pso_q_20.optimize()
    quantized_arr_pso20, _ = quantize_image(img_array, best_palette_20)
    quantized_img_pso20 = quantized_arr_pso20.reshape((img.height, img.width, 3)).astype(np.uint8)
    out_img_pso20 = Image.fromarray(quantized_img_pso20, mode="RGB")
    out_img_pso20.save("pso20_quantized.png")
    out_img_pso20.show()

if __name__ == "__main__":
    main()
