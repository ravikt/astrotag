import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import random
from rs_codec import Generalized_Reed_Solomon
from message_generator import binary_string_to_int_list, int_list_to_binary_string

def introduce_errors(codeword: str, error_positions: List[int]) -> str:
    codeword_list = list(codeword)
    for pos in error_positions:
        codeword_list[pos] = '1' if codeword_list[pos] == '0' else '0'
    return ''.join(codeword_list)

def calculate_metrics(original: str, decoded: str) -> Tuple[float, int, bool]:
    """Calculate BER, Hamming distance, and success"""
    hamming = sum(a != b for a, b in zip(original, decoded))
    ber = hamming / len(original)
    success = (original == decoded)
    return ber, hamming, success

def comprehensive_analysis(message: str, codeword: str, grs_encoder, max_errors: int = 24, trials: int = 100):
    error_counts = range(1, max_errors + 1)
    results = {
        'ber': {'mean': [], 'std': []},
        'hamming': {'mean': [], 'std': []},
        'success_rate': [],
        'position_effect': np.zeros((48, max_errors))
    }
    
    for num_errors in error_counts:
        trial_metrics = []
        
        for _ in range(trials):
            error_positions = random.sample(range(48), num_errors)
            corrupted = introduce_errors(codeword, error_positions)
            
            # Decode
            decoded = grs_encoder.decode(binary_string_to_int_list(corrupted))
            decoded_bits = int_list_to_binary_string(decoded)[:8]
            
            # Calculate metrics
            ber, hamming, success = calculate_metrics(message, decoded_bits)
            trial_metrics.append((ber, hamming, success))
            
            # Track position effects
            for pos in error_positions:
                results['position_effect'][pos, num_errors-1] += (1 if not success else 0)
        
        # Calculate statistics
        bers, hammings, successes = zip(*trial_metrics)
        results['ber']['mean'].append(np.mean(bers))
        results['ber']['std'].append(np.std(bers))
        results['hamming']['mean'].append(np.mean(hammings))
        results['hamming']['std'].append(np.std(hammings))
        results['success_rate'].append(sum(successes) / trials)
    
    # Normalize position effect
    results['position_effect'] /= trials
    
    return error_counts, results

def plot_results(error_counts, results):
    fig = plt.figure(figsize=(15, 10))
    
    # Plot 1: BER
    ax1 = plt.subplot(221)
    ax1.errorbar(error_counts, results['ber']['mean'], yerr=results['ber']['std'], 
                 fmt='o-', capsize=5)
    ax1.set_xlabel('Number of Errors')
    ax1.set_ylabel('Bit Error Rate')
    ax1.set_title('BER vs Number of Errors')
    ax1.grid(True)
    
    # Plot 2: Success Rate
    ax2 = plt.subplot(222)
    ax2.plot(error_counts, results['success_rate'], 'g-', marker='o')
    ax2.set_xlabel('Number of Errors')
    ax2.set_ylabel('Success Rate')
    ax2.set_title('Decoder Success Rate')
    ax2.grid(True)
    
    # Plot 3: Hamming Distance
    ax3 = plt.subplot(223)
    ax3.errorbar(error_counts, results['hamming']['mean'], 
                 yerr=results['hamming']['std'], fmt='r-', capsize=5)
    ax3.set_xlabel('Number of Errors')
    ax3.set_ylabel('Hamming Distance')
    ax3.set_title('Average Hamming Distance')
    ax3.grid(True)
    
    # Plot 4: Position Effect Heatmap
    ax4 = plt.subplot(224)
    im = ax4.imshow(results['position_effect'], aspect='auto', 
                    extent=[1, max(error_counts), 0, 48])
    ax4.set_xlabel('Number of Errors')
    ax4.set_ylabel('Error Position')
    ax4.set_title('Error Position Effect')
    plt.colorbar(im)
    
    plt.tight_layout()
    plt.savefig('grs_analysis.png')
    plt.show()

def print_summary_statistics(error_counts, results):
    max_reliable = max([count for count, rate in zip(error_counts, results['success_rate']) 
                       if rate > 0.95], default=0)
    
    # Find 50% threshold more safely
    threshold_index = next((i for i, rate in enumerate(results['success_rate']) 
                          if rate < 0.5), len(error_counts)-1)
    threshold = error_counts[threshold_index]
    
    print("\nSummary Statistics:")
    print(f"Maximum reliable correction (>95% success): {max_reliable} errors")
    print(f"50% success threshold: {threshold} errors")
    print(f"Average BER at maximum reliable correction: {results['ber']['mean'][0]:.4f}")

# Run analysis
grs_encoder = Generalized_Reed_Solomon(2, 48, 8, 1, 1, None, False, False)
message = "11101111"
codeword = "111011111110111100000000000000001110111111101111"

error_counts, results = comprehensive_analysis(message, codeword, grs_encoder)
plot_results(error_counts, results)
print_summary_statistics(error_counts, results)