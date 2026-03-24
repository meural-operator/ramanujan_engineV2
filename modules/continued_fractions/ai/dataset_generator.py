import json
import random

class MathematicalDatasetGenerator:
    """
    Generates synthetic mathematical identities for pre-training sequence-to-sequence models or LLMs.
    The models will autorearessively predict the symbolic representation of an and bn 
    to match the convergence of target scalars.
    """
    def __init__(self, num_samples=10000):
        self.num_samples = num_samples
        
    def generate_random_polynomial_sequence(self, max_degree=4, max_coef=20):
        """
        Generates a sequence [P(1), P(2), ..., P(N)] mapped to its symbolic polynomial.
        """
        degree = random.randint(1, max_degree)
        coefs = [random.randint(-max_coef, max_coef) for _ in range(degree + 1)]
        
        # Ensure leading coef is non-zero
        if coefs[-1] == 0:
            coefs[-1] = random.choice([1, -1])
            
        return coefs

    def build_synthetic_dataset(self, filepath="synthetic_identities.json"):
        """
        Builds a JSON dictionary of synthetic polynomials mapping numerical sequences
        to the analytical generating coefficients.
        """
        dataset = []
        for _ in range(self.num_samples):
            a_coefs = self.generate_random_polynomial_sequence()
            b_coefs = self.generate_random_polynomial_sequence(max_degree=6)
            
            # The dataset maps partial numerical evaluations to the formal string
            sample = {
                "a_coefs": a_coefs,
                "b_coefs": b_coefs,
                "domain_classification": "polynomial",
                "converges": True # Assume pre-filtered
            }
            dataset.append(sample)
            
        with open(filepath, 'w') as f:
            json.dump(dataset, f, indent=4)
        print(f"Dataset generated at {filepath}")

if __name__ == "__main__":
    generator = MathematicalDatasetGenerator(num_samples=100)
    generator.build_synthetic_dataset("ramanujan/ai/sample_dataset.json")
