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
        """将函数依赖转换为规范形式 (右侧单属性)"""
        canonical_fds = []
        for L, R in self.D0:
            for attr in R:
                canonical_fds.append((L, {attr}))
        return canonical_fds

    def attribute_closure(self, X):
        """计算属性闭包 X+"""
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
        """找到所有最小候选键 (Lucchasi-Osborn 算法)"""
        # 重置统计
        self.stats = {'closure_calls': 0, 'key_checks': 0}
        
        # 步骤1: 找到初始最小键
        K0 = self.minimal_key(self.all_attrs)
        keys = {K0}
        queue = deque([K0])
        
        # 步骤2: 使用引理4寻找其他键
        while queue:
            K = queue.popleft()
            for L, R in self.fd_dict:
                # 计算 S = L ∪ (K - R)
                S = L | (K - R)
                
                # 检查S是否包含已有键
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
        
        # 计算在 E 上的闭包
        closure = self.attribute_closure(K, E)
        
        # 检查是否存在 eUC 满足条件
        for E_prime, K_prime in self.K:
            if E_prime.issubset(E) and K_prime.issubset(closure):
                return True
        return False

    def minimal_embedded_key(self, E, K):
        E_prime = set(E)
        K_prime = set(K)
        
        for attr in sorted(E, key=lambda x: random.random()):
            # 情况1: 尝试同时移除嵌入属性和键属性
            test_key1 = (E_prime - {attr}, K_prime - {attr})
            if self.key_function(*test_key1):
                E_prime -= {attr}
                K_prime -= {attr}
                continue
            
            # 情况2: 仅当属性在键集中才尝试移除
            if attr in K_prime:
                test_key2 = (E_prime, K_prime - {attr})
                if self.key_function(*test_key2):
                    K_prime -= {attr}
        
        return (frozenset(E_prime), frozenset(K_prime))

    def find_all_embedded_keys(self):
        self.stats = {'closure_calls': 0, 'key_checks': 0}
        C = set()
        
        # 处理初始 eUCs
        for E_uc, K_uc in self.K:
            min_key = self.minimal_embedded_key(E_uc, K_uc)
            C.add(min_key)
        
        # Handle embedded FDs 生成新候选
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
    
    # Generate embedded dependencies (简化版)
    embedded_ucs = []
    embedded_fds = []
    
    # 随机创建1-2个eUC
    for _ in range(random.randint(1, 2)):
        e_size = random.randint(2, min(4, num_attrs))
        k_size = random.randint(1, e_size-1)
        E_uc = set(random.sample(attributes, e_size))
        K_uc = set(random.sample(list(E_uc), k_size))
        embedded_ucs.append((E_uc, K_uc))
    
    # 创建eFDs (与标准FDs类似但带嵌入集)
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
    # 1. 有效迭代因子 (0.1-0.3)
    effective_iter_factor = 0.2
    
    # 2. 闭包计算复杂度 (基于平均嵌入集大小)
    avg_embed_size = max(3, size * 0.4)  # 假设平均嵌入集大小
    closure_complexity = avg_embed_size ** 2
    
    # 3. 最小化过程复杂度
    minimization_ops = 1.5 * avg_embed_size  # 非最坏的 2|R|
    
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
            
            # 测试经典算法 (CK)
            start = time.perf_counter()
            ck_gen = CandidateKeyGenerator(data['attributes'], data['std_fds'])
            ck_keys, ck_stats = ck_gen.find_all_keys()
            ck_time = time.perf_counter() - start
            
            # 测试嵌入式算法 (ECK)
            start = time.perf_counter()
            eck_gen = EmbeddedKeyGenerator(
                data['attributes'],
                data['embedded_ucs'],
                data['embedded_fds']
            )
            eck_keys, eck_stats = eck_gen.find_all_embedded_keys()
            eck_time = time.perf_counter() - start
            
            # 计算理论复杂度
            num_ck_keys = len(ck_keys)
            num_eck_keys = len(eck_keys)
            num_fds = len(data['std_fds'])
            num_edeps = len(data['embedded_ucs']) + len(data['embedded_fds'])
            
            # 候选键理论复杂度: O(|D[O]|·|K|·|A|·(|K|+|A|))
            ck_theory = num_fds * num_ck_keys * size * (num_ck_keys + size)
            
            # 嵌入式候选键理论复杂度: O(|C|·|D|·|R|·(|C|+|R|))
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
        
        # 计算平均值
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
    """属性扩展测试 - 全面测试所有属性组合"""
    results = []
    base_sizes = [3, 5, 7]  # 三种不同的基础嵌入集大小
    all_attributes = set(attributes)
    
    for base_size in base_sizes:
        if base_size > len(attributes):
            continue
            
        # 生成所有可能的基础嵌入集组合
        base_sets = list(itertools.combinations(all_attributes, base_size))
        if len(base_sets) > samples_per_size:
            base_sets = random.sample(base_sets, samples_per_size)
        
        for base_E in base_sets:
            base_E = set(base_E)
            available = list(all_attributes - base_E)
            
            # 基础集合测试
            eck_gen = EmbeddedKeyGenerator(attributes, embedded_ucs, embedded_fds)
            start = time.perf_counter()
            eck_gen.find_all_embedded_keys()
            base_time = time.perf_counter() - start
            results.append((f'Base {base_size}', base_size, base_time))
            
            # 测试所有可能的扩展组合
            for extension_size in range(1, len(available) + 1):
                # 生成所有可能的扩展组合
                extension_sets = list(itertools.combinations(available, extension_size))
                if len(extension_sets) > samples_per_size:
                    extension_sets = random.sample(extension_sets, samples_per_size)
                
                extension_times = []
                for extension in extension_sets:
                    current_E = base_E | set(extension)
                    
                    # 测试扩展后的嵌入集
                    start = time.perf_counter()
                    eck_gen.find_all_embedded_keys()
                    exp_time = time.perf_counter() - start
                    extension_times.append(exp_time)
                
                # 记录每个属性大小的所有时间样本
                total_size = len(current_E)
                for time_val in extension_times:
                    results.append((f'Size {base_size}', total_size, time_val))
    
    return results

def plot_performance_comparison(results):
    """ Plot performance comparison """
    plt.figure(figsize=(14, 10))
    
    # 提取数据
    sizes = [res['size'] for res in results]
    ck_times = [res['ck_time'] for res in results]
    eck_times = [res['eck_time'] for res in results]
    ck_ops = [res['ck_ops'] for res in results]
    eck_ops = [res['eck_ops'] for res in results]
    
    # 时间-规模对比图
    plt.subplot(2, 2, 1)
    plt.plot(sizes, ck_times, 'bo-', label='Classical Keys (CK)')
    plt.plot(sizes, eck_times, 'ro-', label='Embedded Keys (ECK)')
    plt.xlabel('Number of Attributes')
    plt.ylabel('Execution Time (s)')
    plt.title('Time Complexity Comparison')
    plt.legend()
    plt.grid(True)
    
    # 操作次数-规模对比图
    plt.subplot(2, 2, 2)
    plt.plot(sizes, ck_ops, 'bo-', label='CK Operations')
    plt.plot(sizes, eck_ops, 'ro-', label='ECK Operations')
    plt.xlabel('Number of Attributes')
    plt.ylabel('Number of Operations')
    plt.title('Operation Count Comparison')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')  # 对数尺度
    
    # 理论复杂度曲线
    plt.subplot(2, 2, 3)
    
    # 提取理论复杂度值
    ck_complexity = [res['ck_complexity'] for res in results]
    ck_complexity_mod = [res['ck_complexity_mod'] for res in results]
    eck_complexity = [res['eck_complexity'] for res in results]
    eck_complexity_mod = [res['eck_complexity_mod'] for res in results]
    
    # 归一化理论复杂度以便比较
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
    
    # 相对性能对比
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

def plot_expansion_results(results):
    """绘制扩展测试结果 - 箱线图展示不同属性大小的时间分布"""
    plt.figure(figsize=(14, 8))
    
    # 按基础大小分组数据
    grouped_data = {}
    for desc, size, time_val in results:
        if desc.startswith('Size'):
            base_size = int(desc.split(' ')[1])
            if base_size not in grouped_data:
                grouped_data[base_size] = {}
            if size not in grouped_data[base_size]:
                grouped_data[base_size][size] = []
            grouped_data[base_size][size].append(time_val)
    
    # 为每个基础大小创建子图
    fig, axes = plt.subplots(1, len(grouped_data), figsize=(15, 6))
    if len(grouped_data) == 1:
        axes = [axes]
    
    for i, (base_size, size_data) in enumerate(grouped_data.items()):
        ax = axes[i]
        
        # 准备箱线图数据
        sizes_sorted = sorted(size_data.keys())
        box_data = [size_data[size] for size in sizes_sorted]
        
        # 创建箱线图
        box = ax.boxplot(box_data, positions=sizes_sorted, patch_artist=True, widths=0.6)
        
        # 设置箱线图颜色
        colors = ['lightblue', 'lightgreen', 'salmon']
        for patch in box['boxes']:
            patch.set_facecolor(colors[i % len(colors)])
        
        # 添加均值线
        means = [np.mean(times) for times in box_data]
        ax.plot(sizes_sorted, means, 'ko-', label='Mean Time')
        
        ax.set_xlabel('Number of Attributes in Embedded Set')
        ax.set_ylabel('Execution Time (s)')
        ax.set_title(f'Base Size = {base_size} Attributes')
        ax.set_xticks(sizes_sorted)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('expansion_test_boxplots.png', dpi=300)
    plt.show()

def plot_operation_comparison(results):
    """ Plot operation count comparison """
    plt.figure(figsize=(12, 6))
    
    sizes = [res['size'] for res in results]
    ck_ops = [res['ck_ops'] for res in results]
    eck_ops = [res['eck_ops'] for res in results]
    ck_complexity = [res['ck_complexity'] for res in results]
    eck_complexity = [res['eck_complexity'] for res in results]
    
    # 归一化理论复杂度以便比较
    if ck_ops and ck_complexity:
        ck_norm_factor = ck_ops[0] / ck_complexity[0] if ck_complexity[0] != 0 else 1
        eck_norm_factor = eck_ops[0] / eck_complexity[0] if eck_complexity[0] != 0 else 1
        
        ck_complexity_norm = [x * ck_norm_factor for x in ck_complexity]
        eck_complexity_norm = [x * eck_norm_factor for x in eck_complexity]
    
    # CK操作次数分析
    plt.subplot(1, 2, 1)
    plt.plot(sizes, ck_ops, 'bo-', label='Actual Operations')
    
    # CK理论复杂度
    if ck_ops and ck_complexity:
        plt.plot(sizes, ck_complexity_norm, 'b--', label='Theoretical Complexity')
    
    plt.xlabel('Number of Attributes')
    plt.ylabel('Operations')
    plt.title('CK Operation Count')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    
    # ECK操作次数分析
    plt.subplot(1, 2, 2)
    plt.plot(sizes, eck_ops, 'ro-', label='Actual Operations')
    
    # ECK理论复杂度
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
    
    # CK时间-操作关系
    plt.subplot(1, 2, 1)
    plt.scatter(ck_ops, ck_times, c='b', s=50)
    plt.xlabel('Number of Operations')
    plt.ylabel('Execution Time (s)')
    plt.title('CK: Time vs Operations')
    
    # 添加线性回归线
    if len(ck_ops) > 1:
        m, b = np.polyfit(ck_ops, ck_times, 1)
        plt.plot(ck_ops, [m*x + b for x in ck_ops], 'b--', 
                label=f'Time = {m:.2e} × Ops + {b:.2e}')
        plt.legend()
    
    # ECK时间-操作关系
    plt.subplot(1, 2, 2)
    plt.scatter(eck_ops, eck_times, c='r', s=50)
    plt.xlabel('Number of Operations')
    plt.ylabel('Execution Time (s)')
    plt.title('ECK: Time vs Operations')
    
    # 添加线性回归线
    if len(eck_ops) > 1:
        m, b = np.polyfit(eck_ops, eck_times, 1)
        plt.plot(eck_ops, [m*x + b for x in eck_ops], 'r--', 
                label=f'Time = {m:.2e} × Ops + {b:.2e}')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('time_vs_operations.png', dpi=300)
    plt.show()

# 示例用法
if __name__ == "__main__":
    print("Starting performance analysis...")
    
    # 运行性能测试
    perf_results = performance_test(max_size=40, step=1, num_trials=5)
    
    # 打印结果
    print("\nPerformance Results:")
    print(f"{'Size':<6} | {'CK Time (s)':<12} | {'ECK Time (s)':<12} | {'CK Keys':<8} | {'ECK Keys':<8} | {'CK Ops':<10} | {'ECK Ops':<10} | {'CK Comp':<12} | {'ECK Comp':<12}")
    for res in perf_results:
        print(f"{res['size']:<6} | {res['ck_time']:<12.6f} | {res['eck_time']:<12.6f} | {res['ck_key_count']:<8.1f} | {res['eck_key_count']:<8.1f} | {res['ck_ops']:<10.0f} | {res['eck_ops']:<10.0f} | {res['ck_complexity']:<12.0f} | {res['eck_complexity']:<12.0f}")
    
    # Plot performance comparison
    plot_performance_comparison(perf_results)
    plot_operation_comparison(perf_results)
    plot_time_vs_operations(perf_results)
    
    # 运行扩展测试
    test_data = generate_test_data(15, 20)
    exp_results = expansion_test(
        test_data['attributes'],
        test_data['embedded_ucs'],
        test_data['embedded_fds']
    )
    
    # 打印扩展测试结果
    print("\nExpansion Test Results:")
    print(f"{'Set':<10} | {'Size':<6} | {'Time (s)':<10}")
    for desc, size, time_val in exp_results:
        print(f"{desc:<10} | {size:<6} | {time_val:<10.6f}")
    
    # Plot expansion test results
    plot_expansion_results(exp_results)
    
    print("\nPerformance analysis completed. Graphs saved as PNG files.")
