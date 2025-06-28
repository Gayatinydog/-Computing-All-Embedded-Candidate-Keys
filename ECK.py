import itertools
import random
import time
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
from functools import lru_cache
import math

class CandidateKeyGenerator:
    def __init__(self, attributes, functional_deps):
        self.A = set(attributes)
        self.D0 = [(set(l), set(r)) for l, r in functional_deps]
        self.all_attrs = set(attributes)
        self.fd_dict = self._canonical_form()
        self.stats = {'closure_calls': 0, 'key_checks': 0}
        
    def _canonical_form(self):
        canonical_fds = []
        for L, R in self.D0:
            for attr in R:
                canonical_fds.append((L, {attr}))
        return canonical_fds

    def attribute_closure(self, X):
        self.stats['closure_calls'] += 1
        closure = set(X)
        changed = True
        while changed:
            changed = False
            for L, R in self.fd_dict:
                if L.issubset(closure) and not R.issubset(closure):
                    closure |= R
                    changed = True
        return closure

    def is_key(self, K):
        """ Check whether K is a candidate key """
        self.stats['key_checks'] += 1
        return self.attribute_closure(K) == self.all_attrs

    def minimal_key(self, candidate_key):
        """ Minimize candidate key """
        K_prime = set(candidate_key)
        for attr in sorted(candidate_key, key=lambda x: random.random()):  # 随机顺序
            if self.is_key(K_prime - {attr}):
                K_prime -= {attr}
        return frozenset(K_prime)

    def find_all_keys(self):
        """Lucchasi-Osborn"""
        
        self.stats = {'closure_calls': 0, 'key_checks': 0}
        
        # Find minimal_key
        K0 = self.minimal_key(self.all_attrs)
        keys = {K0}
        queue = deque([K0])
        
        
        while queue:
            K = queue.popleft()
            for L, R in self.fd_dict:
                # 计算 S = L ∪ (K - R)
                S = L | (K - R)
                
                
                if not any(J.issubset(S) for J in keys):
                    new_key = self.minimal_key(S)
                    if new_key not in keys:
                        keys.add(new_key)
                        queue.append(new_key)
        
        return keys, self.stats

class EmbeddedKeyGenerator:
    def __init__(self, attributes, embedded_ucs, embedded_fds):
        self.R = set(attributes)
        self.K = embedded_ucs  # [(E, K)]
        self.F = embedded_fds  # [(E, X, Y)]
        self.prime_closure_cache = {}
        self.stats = {'closure_calls': 0, 'key_checks': 0}
        
    def project_dependencies(self, E):
        proj_fds = []
        # Handle embedded FDs
        for E_fd, X, Y in self.F:
            if E_fd.issubset(E):
                proj_fds.append((X, Y))
        
        # Convert embedded UCs to FDs (转换为FDs)
        for E_uc, K_uc in self.K:
            if E_uc.issubset(E):
                proj_fds.append((K_uc, E_uc))
        
        return proj_fds
    
    def attribute_closure(self, X, E):
        self.stats['closure_calls'] += 1
        proj_fds = self.project_dependencies(E)
        closure = set(X)
        changed = True
        while changed:
            changed = False
            for L, R in proj_fds:
                if L.issubset(closure) and not R.issubset(closure):
                    closure |= R
                    closure &= E
                    changed = True
        return closure

    def key_function(self, E, K):
        self.stats['key_checks'] += 1
        
        # Check directly via Lemma 2.1
        for E_prime, K_prime in self.K:
            if E_prime.issubset(E) and K_prime.issubset(K):
                return True
        
        # attribute_closure in E
        closure = self.attribute_closure(K, E)
        
        # check eUC 
        for E_prime, K_prime in self.K:
            if E_prime.issubset(E) and K_prime.issubset(closure):
                return True
        return False

    def minimal_embedded_key(self, E, K):
        E_prime = set(E)
        K_prime = set(K)
        
        for attr in sorted(E, key=lambda x: random.random()):
            # Case 1: Attempt to remove both embedding attributes and key attributes simultaneously.
            test_key1 = (E_prime - {attr}, K_prime - {attr})
            if self.key_function(*test_key1):
                E_prime -= {attr}
                K_prime -= {attr}
                continue
            
            # Case 2: Attempt removal only if the attribute is in the key set.
            if attr in K_prime:
                test_key2 = (E_prime, K_prime - {attr})
                if self.key_function(*test_key2):
                    K_prime -= {attr}
        
        return (frozenset(E_prime), frozenset(K_prime))

    def find_all_embedded_keys(self):
        self.stats = {'closure_calls': 0, 'key_checks': 0}
        C = set()
        
        # eUCs
        for E_uc, K_uc in self.K:
            min_key = self.minimal_embedded_key(E_uc, K_uc)
            C.add(min_key)
        
        # Handle embedded FDs 
        new_keys = True
        while new_keys:
            new_keys = False
            current_keys = list(C)
            
            for (E, K) in current_keys:
                for (E_fd, X, Y) in self.F:
                    S = E | E_fd
                    T = X | (K - Y)
                    
                    # Validate the new candidate key
                    if not self.key_function(S, T):
                        continue
                    
                    # Check whether it is subsumed by existing keys
                    subsumed = False
                    for (E_exist, K_exist) in C:
                        if E_exist.issubset(S) and K_exist.issubset(T):
                            subsumed = True
                            break
                    
                    if not subsumed:
                        new_key = self.minimal_embedded_key(S, T)
                        if new_key not in C:
                            C.add(new_key)
                            new_keys = True
        
        return C, self.stats

def generate_test_data(num_attrs, num_fds, max_rhs=2):
    """ Generate random test data """
    attributes = [f'A{i}' for i in range(num_attrs)]
    
    # Generate standard functional dependencies
    std_fds = []
    for _ in range(num_fds):
        lhs_size = random.randint(1, min(3, num_attrs-1))
        rhs_size = random.randint(1, max_rhs)
        lhs = set(random.sample(attributes, lhs_size))
        rhs = set(random.sample(attributes, rhs_size))
        std_fds.append((lhs, rhs))
    
    # Generate embedded dependencies
    embedded_ucs = []
    embedded_fds = []
    
    # Randomly create 1–2 eUCs (embedded unary constraints).
    for _ in range(random.randint(1, 2)):
        e_size = random.randint(2, min(4, num_attrs))
        k_size = random.randint(1, e_size-1)
        E_uc = set(random.sample(attributes, e_size))
        K_uc = set(random.sample(list(E_uc), k_size))
        embedded_ucs.append((E_uc, K_uc))
    
    # Create eFDs, which are similar to standard FDs but include an embedding set.
    for _ in range(num_fds):
        e_size = random.randint(1, min(3, num_attrs))
        E_fd = set(random.sample(attributes, e_size))
        lhs = set(random.sample(list(E_fd), min(len(E_fd), random.randint(1, 2))))
        rhs = set(random.sample(list(E_fd), 1))
        embedded_fds.append((E_fd, lhs, rhs))
    
    return {
        'attributes': attributes,
        'std_fds': std_fds,
        'embedded_ucs': embedded_ucs,
        'embedded_fds': embedded_fds
    }

# Revised theoretical model
def calculate_eck_complexity(num_eck_keys, num_edeps, size):
    # 1. Effective iteration factor (0.1–0.3)
    effective_iter_factor = 0.2
    
    # 2. Closure computation complexity (based on average embedding set size)
    avg_embed_size = max(3, size * 0.4)  # Assume an average embedding set size.
    closure_complexity = avg_embed_size ** 2
    
    # 3. Minimization process complexity
    minimization_ops = 1.5 * avg_embed_size  # Non-worst-case 2|R|
    
    return int(
        num_eck_keys * num_edeps * effective_iter_factor *
        (closure_complexity + minimization_ops)
    )

def performance_test(max_size=25, step=5, num_trials=3):
    """ Performance testing framework """
    results = []
    sizes = list(range(5, max_size + 1, step))
    
    for size in sizes:
        ck_times = []
        eck_times = []
        ck_counts = []
        eck_counts = []
        ck_ops = []
        eck_ops = []
        ck_complexity = []
        ck_complexity_mod = []
        eck_complexity = []
        eck_complexity_mod = []
        
        for _ in range(num_trials):
            data = generate_test_data(size, size - 2)
            
            # CK
            start = time.perf_counter()
            ck_gen = CandidateKeyGenerator(data['attributes'], data['std_fds'])
            ck_keys, ck_stats = ck_gen.find_all_keys()
            ck_time = time.perf_counter() - start
            
            # ECK
            start = time.perf_counter()
            eck_gen = EmbeddedKeyGenerator(
                data['attributes'],
                data['embedded_ucs'],
                data['embedded_fds']
            )
            eck_keys, eck_stats = eck_gen.find_all_embedded_keys()
            eck_time = time.perf_counter() - start
            
            # Compute theoretical complexity
            num_ck_keys = len(ck_keys)
            num_eck_keys = len(eck_keys)
            num_fds = len(data['std_fds'])
            num_edeps = len(data['embedded_ucs']) + len(data['embedded_fds'])
            
            # Theoretical complexity of candidate key computation: O(|D[O]|·|K|·|A|·(|K|+|A|))
            ck_theory = num_fds * num_ck_keys * size * (num_ck_keys + size)
            
            # Theoretical complexity of embedded candidate key computation: O(|C|·|D|·|R|·(|C|+|R|))
            eck_theory = num_eck_keys * num_edeps * size * (num_eck_keys + size)
            
            ck_times.append(ck_time)
            eck_times.append(eck_time)
            ck_counts.append(num_ck_keys)
            eck_counts.append(num_eck_keys)
            ck_ops.append(ck_stats['closure_calls'] + ck_stats['key_checks'])
            eck_ops.append(eck_stats['closure_calls'] + eck_stats['key_checks'])
            ck_complexity.append(ck_theory)
            ck_complexity_mod = num_ck_keys * num_fds * size * size  # O(|C||D||R|^2)
            eck_complexity.append(eck_theory)
            eck_complexity_mod = calculate_eck_complexity(num_eck_keys, num_edeps, size)
        
        # mean
        results.append({
            'size': size,
            'ck_time': np.mean(ck_times),
            'ck_time_std': np.std(ck_times),
            'ck_key_count': np.mean(ck_counts),
            'eck_time': np.mean(eck_times),
            'eck_time_std': np.std(eck_times),
            'eck_key_count': np.mean(eck_counts),
            'ck_ops': np.mean(ck_ops),
            'eck_ops': np.mean(eck_ops),
            'ck_complexity': np.mean(ck_complexity),
            'ck_complexity_mod': np.mean(ck_complexity_mod),
            'eck_complexity': np.mean(eck_complexity),
            'eck_complexity_mod': np.mean(eck_complexity_mod)
        })
    
    return results

def expansion_test(attributes, embedded_ucs, embedded_fds, samples_per_size=5):
    """Attribute Expansion Test – Exhaustively test all attribute combinations"""
    results = []
    base_sizes = [3, 5, 7]  # Three different base embedding set sizes
    all_attributes = set(attributes)
    
    for base_size in base_sizes:
        if base_size > len(attributes):
            continue
            
        # Generate all possible combinations of base embedding sets
        base_sets = list(itertools.combinations(all_attributes, base_size))
        if len(base_sets) > samples_per_size:
            base_sets = random.sample(base_sets, samples_per_size)
        
        for base_E in base_sets:
            base_E = set(base_E)
            available = list(all_attributes - base_E)
            
            # Base set test
            eck_gen = EmbeddedKeyGenerator(attributes, embedded_ucs, embedded_fds)
            start = time.perf_counter()
            eck_gen.find_all_embedded_keys()
            base_time = time.perf_counter() - start
            results.append((f'Base {base_size}', base_size, base_time))
            
            # Test all possible expansion combinations
            for extension_size in range(1, len(available) + 1):
                # Generate all possible expansion combinations
                extension_sets = list(itertools.combinations(available, extension_size))
                if len(extension_sets) > samples_per_size:
                    extension_sets = random.sample(extension_sets, samples_per_size)
                
                extension_times = []
                for extension in extension_sets:
                    current_E = base_E | set(extension)
                    
                    # Test the expanded embedding sets
                    start = time.perf_counter()
                    eck_gen.find_all_embedded_keys()
                    exp_time = time.perf_counter() - start
                    extension_times.append(exp_time)
                
                # Record all time samples for each embedding set size
                total_size = len(current_E)
                for time_val in extension_times:
                    results.append((f'Size {base_size}', total_size, time_val))
    
    return results

def plot_performance_comparison(results):
    """ Plot performance comparison """
    plt.figure(figsize=(14, 10))
    
    # Extract data
    sizes = [res['size'] for res in results]
    ck_times = [res['ck_time'] for res in results]
    eck_times = [res['eck_time'] for res in results]
    ck_ops = [res['ck_ops'] for res in results]
    eck_ops = [res['eck_ops'] for res in results]
    
    # Time vs. Size comparison chart
    plt.subplot(2, 2, 1)
    plt.plot(sizes, ck_times, 'bo-', label='Classical Keys (CK)')
    plt.plot(sizes, eck_times, 'ro-', label='Embedded Keys (ECK)')
    plt.xlabel('Number of Attributes')
    plt.ylabel('Execution Time (s)')
    plt.title('Time Complexity Comparison')
    plt.legend()
    plt.grid(True)
    
    # Operations vs. Size comparison chart
    plt.subplot(2, 2, 2)
    plt.plot(sizes, ck_ops, 'bo-', label='CK Operations')
    plt.plot(sizes, eck_ops, 'ro-', label='ECK Operations')
    plt.xlabel('Number of Attributes')
    plt.ylabel('Number of Operations')
    plt.title('Operation Count Comparison')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')  # Logarithmic scale
    
    # Theoretical complexity curve
    plt.subplot(2, 2, 3)
    
    # Extract theoretical complexity values
    ck_complexity = [res['ck_complexity'] for res in results]
    ck_complexity_mod = [res['ck_complexity_mod'] for res in results]
    eck_complexity = [res['eck_complexity'] for res in results]
    eck_complexity_mod = [res['eck_complexity_mod'] for res in results]
    
    # Normalize theoretical complexity for comparison
    if ck_ops and eck_ops:
        ck_norm_factor = ck_ops[0] / ck_complexity[0] if ck_complexity[0] != 0 else 1
        ck_mod_norm_factor = ck_ops[0] / ck_complexity_mod[0] if ck_complexity_mod[0] != 0 else 1
        eck_norm_factor = eck_ops[0] / eck_complexity[0] if eck_complexity[0] != 0 else 1
        eck_mod_norm_factor = eck_ops[0] / eck_complexity_mod[0] if eck_complexity_mod[0] != 0 else 1
        
        ck_complexity_norm = [x * ck_norm_factor for x in ck_complexity]
        ck_complexity_mod_norm = [x * ck_mod_norm_factor for x in ck_complexity_mod]
        eck_complexity_norm = [x * eck_norm_factor for x in eck_complexity]
        eck_complexity_mod_norm = [x * eck_mod_norm_factor for x in eck_complexity_mod]
        
        plt.plot(sizes, ck_ops, 'bo-', label='CK Actual Ops')
        plt.plot(sizes, ck_complexity_norm, 'b--', label='CK Theoretical Ops')
        # plt.plot(sizes, ck_complexity_mod_norm, 'b:', label='CK Theoretical Ops Modified')
        plt.plot(sizes, eck_ops, 'ro-', label='ECK Actual Ops')
        plt.plot(sizes, eck_complexity_norm, 'r--', label='ECK Theoretical Ops')
        # plt.plot(sizes, eck_complexity_mod_norm, 'r:', label='ECK Theoretical Ops Modified')
        
        plt.xlabel('Number of Attributes')
        plt.ylabel('Normalized Operations')
        plt.title('Theoretical vs Actual Operations')
        plt.legend()
        plt.grid(True)
        plt.yscale('log')
    
    # performance comparison
    plt.subplot(2, 2, 4)
    ratios = [eck / ck for ck, eck in zip(ck_times, eck_times)]
    plt.plot(sizes, ratios, 'go-')
    plt.xlabel('Number of Attributes')
    plt.ylabel('ECK Time / CK Time')
    plt.title('Relative Performance (ECK/CK)')
    plt.grid(True)
    plt.axhline(y=1, color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('performance_comparison.png', dpi=300)
    plt.show()


def plot_operation_comparison(results):
    """ Plot operation count comparison """
    plt.figure(figsize=(12, 6))
    
    sizes = [res['size'] for res in results]
    ck_ops = [res['ck_ops'] for res in results]
    eck_ops = [res['eck_ops'] for res in results]
    ck_complexity = [res['ck_complexity'] for res in results]
    eck_complexity = [res['eck_complexity'] for res in results]
    
    # Normalize theoretical complexity for comparison purposes.
    if ck_ops and ck_complexity:
        ck_norm_factor = ck_ops[0] / ck_complexity[0] if ck_complexity[0] != 0 else 1
        eck_norm_factor = eck_ops[0] / eck_complexity[0] if eck_complexity[0] != 0 else 1
        
        ck_complexity_norm = [x * ck_norm_factor for x in ck_complexity]
        eck_complexity_norm = [x * eck_norm_factor for x in eck_complexity]
    
    # CK operation count analysis
    plt.subplot(1, 2, 1)
    plt.plot(sizes, ck_ops, 'bo-', label='Actual Operations')
    
    # CK theoretical complexity
    if ck_ops and ck_complexity:
        plt.plot(sizes, ck_complexity_norm, 'b--', label='Theoretical Complexity')
    
    plt.xlabel('Number of Attributes')
    plt.ylabel('Operations')
    plt.title('CK Operation Count')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    
    # ECK operation count analysis
    plt.subplot(1, 2, 2)
    plt.plot(sizes, eck_ops, 'ro-', label='Actual Operations')
    
    # ECK theoretical complexity
    if eck_ops and eck_complexity:
        plt.plot(sizes, eck_complexity_norm, 'r--', label='Theoretical Complexity')
    
    plt.xlabel('Number of Attributes')
    plt.ylabel('Operations')
    plt.title('ECK Operation Count')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig('operation_comparison.png', dpi=300)
    plt.show()

def plot_time_vs_operations(results):
    """ Plot relationship between time and operations """
    plt.figure(figsize=(12, 6))
    
    ck_times = [res['ck_time'] for res in results]
    eck_times = [res['eck_time'] for res in results]
    ck_ops = [res['ck_ops'] for res in results]
    eck_ops = [res['eck_ops'] for res in results]
    
    # CK time–operation relationship
    plt.subplot(1, 2, 1)
    plt.scatter(ck_ops, ck_times, c='b', s=50)
    plt.xlabel('Number of Operations')
    plt.ylabel('Execution Time (s)')
    plt.title('CK: Time vs Operations')
    
    # Add a linear regression line
    if len(ck_ops) > 1:
        m, b = np.polyfit(ck_ops, ck_times, 1)
        plt.plot(ck_ops, [m*x + b for x in ck_ops], 'b--', 
                label=f'Time = {m:.2e} × Ops + {b:.2e}')
        plt.legend()
    
    # ECK time–operation relationship
    plt.subplot(1, 2, 2)
    plt.scatter(eck_ops, eck_times, c='r', s=50)
    plt.xlabel('Number of Operations')
    plt.ylabel('Execution Time (s)')
    plt.title('ECK: Time vs Operations')
    
    # Add a linear regression line.
    if len(eck_ops) > 1:
        m, b = np.polyfit(eck_ops, eck_times, 1)
        plt.plot(eck_ops, [m*x + b for x in eck_ops], 'r--', 
                label=f'Time = {m:.2e} × Ops + {b:.2e}')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('time_vs_operations.png', dpi=300)
    plt.show()
# example
if __name__ == "__main__":
    print("Starting performance analysis...")

    perf_results = performance_test(max_size=40, step=1, num_trials=5)
    
    print("\nPerformance Results:")
    print(f"{'Size':<6} | {'CK Time (s)':<12} | {'ECK Time (s)':<12} | {'CK Keys':<8} | {'ECK Keys':<8} | {'CK Ops':<10} | {'ECK Ops':<10} | {'CK Comp':<12} | {'ECK Comp':<12}")
    for res in perf_results:
        print(f"{res['size']:<6} | {res['ck_time']:<12.6f} | {res['eck_time']:<12.6f} | {res['ck_key_count']:<8.1f} | {res['eck_key_count']:<8.1f} | {res['ck_ops']:<10.0f} | {res['eck_ops']:<10.0f} | {res['ck_complexity']:<12.0f} | {res['eck_complexity']:<12.0f}")
    
    # Plot performance comparison
    plot_performance_comparison(perf_results)
    plot_operation_comparison(perf_results)
    plot_time_vs_operations(perf_results)
    
    # generate test
    test_data = generate_test_data(15, 20)
    exp_results = expansion_test(
        test_data['attributes'],
        test_data['embedded_ucs'],
        test_data['embedded_fds']
    )
    
    print("\nExpansion Test Results:")
    print(f"{'Set':<10} | {'Size':<6} | {'Time (s)':<10}")
    for desc, size, time_val in exp_results:
        print(f"{desc:<10} | {size:<6} | {time_val:<10.6f}")
    
    # Plot expansion test results
    plot_expansion_results(exp_results)
    
    print("\nPerformance analysis completed. Graphs saved as PNG files.")
