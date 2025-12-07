from baseline_verification import main as baseline
from fhe_baseline import main as fhe_baseline
from fhe_jl import main as jl_reduction
from fhe_learned_autoencoder import main as ae_reduction
from fhe_pca import main as pca_reduction
from fhe_rand_proj_gauss import main as grp_reduction
from fhe_rand_proj_sparse import main as srp_reduction
from fhe_rsvd import main as rsvd_reduction


if __name__ == "__main__":
    baseline()
    fhe_baseline(csv_path='results/baseline')

    for i in range(10):
        print(f'--- RUN {i+1} ---\n')
        print('\nRunning JL\n')
        jl_reduction(csv_path='results/jl_results.csv')
        print('\nRunning AE\n')
        ae_reduction(csv_path='results/ae_results.csv')
        print('\nRunning PCA\n')
        pca_reduction(csv_path='results/pca_results.csv')
        print('\nRunning GRP\n')
        grp_reduction(csv_path='results/grp_results.csv')
        print('\nRunning SRP\n')
        srp_reduction(csv_path='results/srp_results.csv')
        print('\nRunning RSVD\n')
        rsvd_reduction(csv_path='results/rsvd_results.csv')
