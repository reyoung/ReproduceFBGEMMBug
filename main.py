from pip import main
import torch
import fbgemm_gpu
import pickle

def main():
    cuda_args = pickle.load(open("./block_bucketize_sparse_features.1.args.pkl", 'rb'))
    cuda_results = torch.ops.fbgemm.block_bucketize_sparse_features(*cuda_args)
    cpu_args = [item.to("cpu") if isinstance(item, torch.Tensor) else item for item in cuda_args]
    cpu_results = torch.ops.fbgemm.block_bucketize_sparse_features(*cpu_args)

    for i, (cuda_result, cpu_result) in enumerate(zip(cuda_results, cpu_results)):
        if isinstance(cuda_result, torch.Tensor):
            print(i, (cuda_result.to("cpu") == cpu_result).all())
        else:
            print(i, cuda_result == cpu_result)
        
    # The bucketized_indices is mismatch.
    # There is a strage value 33445
    print(cuda_results[1].tolist()) 

if __name__ == '__main__':
    main()