import numpy as np
from PIL import Image
import random
import matplotlib.pyplot as plt


"""
in this script we run grid search to find the distances between the images 
quantised by pso after 50 generation of training with different values of
p and k (so different number of particles and clusters/colors for each particle).
from the results we generate a heatmap.
"""
def quantize_image(image_array, palette):
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
    _, ssd = quantize_image(image_array, palette)
    return ssd


class PSOColorQuantizer:
    def __init__(self, image_array, K=8, num_particles=20, omega=0.7, alpha1=1.4, alpha2=1.4, max_iter=50, seed=42):
        np.random.seed(seed)
        random.seed(seed)
        
        self.image_array = image_array
        self.M = image_array.shape[0]
        self.K = K
        self.dim = 3 * K
        self.num_particles = num_particles
        self.omega = omega
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.max_iter = max_iter
        
        #intit positions/velocities
        self.positions = []
        self.velocities = []
        
        for _ in range(num_particles):
            pos = np.random.uniform(low=0, high=255, size=self.dim)
            vel = np.zeros(self.dim)
            
            self.positions.append(pos)
            self.velocities.append(vel)
        
        self.positions = np.array(self.positions)
        self.velocities = np.array(self.velocities)
        
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_fitness = np.array([np.inf]*num_particles)
        
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
        for it in range(self.max_iter):
            for i in range(self.num_particles):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                
                cognitive = self.alpha1 * r1 * (self.personal_best_positions[i] - self.positions[i])
                social    = self.alpha2 * r2 * (self.global_best_position - self.positions[i])
                
                self.velocities[i] = (self.omega * self.velocities[i]) + cognitive + social
                self.positions[i] += self.velocities[i]
                #rgp clipping
                np.clip(self.positions[i], 0, 255, out=self.positions[i])
            
            for i in range(self.num_particles):
                fitness_i = self._fitness(self.positions[i])
                
                if fitness_i < self.personal_best_fitness[i]:
                    self.personal_best_fitness[i] = fitness_i
                    self.personal_best_positions[i] = self.positions[i].copy()
                
                if fitness_i < self.global_best_fitness:
                    self.global_best_fitness = fitness_i
                    self.global_best_position = self.positions[i].copy()
        
        best_palette = self.global_best_position.reshape((self.K, 3))
        return best_palette, self.global_best_fitness



def main():
    img = Image.open("image.png").convert("RGB")
    img_array = np.array(img, dtype=np.float32).reshape((-1, 3))
    
    k_values = [4, 8, 12, 16]
    particle_values = [4, 8, 16, 20] 
    
    results = np.zeros((len(k_values), len(particle_values)))
    
    for i, k in enumerate(k_values):
        for j, num_p in enumerate(particle_values):
            print(f"Running PSO with K={k}, num_particles={num_p}")
            
            pso_q = PSOColorQuantizer(
                image_array=img_array,
                K=k,
                num_particles=num_p,
                omega=0.7,
                alpha1=1.4,
                alpha2=1.4,
                max_iter=50, 
                seed=42
            )
            
            _, final_fitness = pso_q.optimize()
            results[i, j] = final_fitness
            print(f" -> Final fitness: {final_fitness:.2f}")
    
    #heatmap
    plt.figure(figsize=(6,5))
    heatmap = plt.imshow(results, cmap='viridis', origin='upper',  aspect='auto', interpolation='nearest')
    
    plt.colorbar(heatmap, label="Final Best Fitness (SSD)")
    plt.xticks(ticks=np.arange(len(particle_values)), labels=particle_values)
    plt.yticks(ticks=np.arange(len(k_values)), labels=k_values)
    plt.xlabel("Number of Particles")
    plt.ylabel("K (number of clusters)")
    
    plt.title("PSO Final Fitness for Different (K, ParticleCount)")
    plt.tight_layout()
    plt.savefig("pso_2d_heatmap.png")
    plt.show()

if __name__ == "__main__":
    main()
