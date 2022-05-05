import argparse
import numpy as np
import torch 
from pprint import pprint
from scipy.optimize import linear_sum_assignment
import point_cloud_utils as pcu
import os 
import random
import glob
from tqdm import tqdm

CAT2ID = {
    'car': '02958343',
    'chair': '03001627'
}

CAT2SCALE = {
    '02958343': 0.9,
    '03001627': 0.7
    }


def seed_everything(seed):
    if seed < 0:
        return
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# Borrow from https://github.com/ThibaultGROUEIX/AtlasNet
def distChamfer(a, b):
    x, y = a, b
    _, num_points, _ = x.size()
    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    zz = torch.bmm(x, y.transpose(2, 1))
    diag_ind = torch.arange(0, num_points).to(a).long()
    rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
    ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
    P = (rx.transpose(2, 1) + ry - 2 * zz)
    return P.min(1)[0], P.min(2)[0]


# Import CUDA version of approximate EMD, from https://github.com/zekunhao1995/pcgan-pytorch/
try:
    from metrics.StructuralLosses.nn_distance import nn_distance
    def distChamferCUDA(x, y):
        return nn_distance(x, y)
except:
    print("distChamferCUDA not available; fall back to slower version.")
    def distChamferCUDA(x, y):
        return distChamfer(x, y)

try:
    from metrics.StructuralLosses.match_cost import match_cost
    def emd_approx_cuda(sample, ref):
        B, N, N_ref = sample.size(0), sample.size(1), ref.size(1)
        assert N == N_ref, "Not sure what would EMD do in this case"
        emd = match_cost(sample, ref)  # (B,)
        emd_norm = emd / float(N)  # (B,)
        return emd_norm
except:
    print("emd_approx_cuda not available. Fall back to slower version.")
    def emd_approx_cuda(sample, ref):
        return emd_approx(sample, ref)
        
       
def emd_approx(x, y):
    bs, npts, mpts, dim = x.size(0), x.size(1), y.size(1), x.size(2)
    assert npts == mpts, "EMD only works if two point clouds are equal size"
    dim = x.shape[-1]
    x = x.reshape(bs, npts, 1, dim)
    y = y.reshape(bs, 1, mpts, dim)
    dist = (x - y).norm(dim=-1, keepdim=False)  # (bs, npts, mpts)

    emd_lst = []
    dist_np = dist.cpu().detach().numpy()
    for i in range(bs):
        d_i = dist_np[i]
        r_idx, c_idx = linear_sum_assignment(d_i)
        emd_i = d_i[r_idx, c_idx].mean()
        emd_lst.append(emd_i)
    emd = np.stack(emd_lst).reshape(-1)
    emd_torch = torch.from_numpy(emd).to(x)
    return emd_torch


def EMD_CD(sample_pcs, ref_pcs, batch_size, reduced=True):
    N_sample = sample_pcs.shape[0]
    N_ref = ref_pcs.shape[0]
    assert N_sample == N_ref, "REF:%d SMP:%d" % (N_ref, N_sample)

    cd_lst = []
    emd_lst = []
    iterator = range(0, N_sample, batch_size)

    for b_start in iterator:
        b_end = min(N_sample, b_start + batch_size)
        sample_batch = sample_pcs[b_start:b_end]
        ref_batch = ref_pcs[b_start:b_end]

        dl, dr = distChamferCUDA(sample_batch, ref_batch)
        cd_lst.append(dl.mean(dim=1) + dr.mean(dim=1))

        emd_batch = emd_approx_cuda(sample_batch, ref_batch)
        emd_lst.append(emd_batch)

    if reduced:
        cd = torch.cat(cd_lst).mean()
        emd = torch.cat(emd_lst).mean()
    else:
        cd = torch.cat(cd_lst)
        emd = torch.cat(emd_lst)

    results = {
        'MMD-CD': cd,
        'MMD-EMD': emd,
    }
    return results


def _pairwise_EMD_CD_(sample_pcs, ref_pcs, batch_size, compute_emd):
    N_sample = sample_pcs.shape[0]
    N_ref = ref_pcs.shape[0]
    all_cd = []
    all_emd = []
    iterator = range(N_sample)

    for sample_b_start in tqdm(iterator):
        sample_batch = sample_pcs[sample_b_start]

        cd_lst = []
        emd_lst = []
        for ref_b_start in range(0, N_ref, batch_size):
            ref_b_end = min(N_ref, ref_b_start + batch_size)
            ref_batch = ref_pcs[ref_b_start:ref_b_end]

            batch_size_ref = ref_batch.size(0)
            sample_batch_exp = sample_batch.view(1, -1, 3).expand(batch_size_ref, -1, -1)
            sample_batch_exp = sample_batch_exp.contiguous()

            dl, dr = distChamferCUDA(sample_batch_exp, ref_batch)
            cd_lst.append((dl.mean(dim=1) + dr.mean(dim=1)).view(1, -1))

            if compute_emd:
                emd_batch = emd_approx_cuda(sample_batch_exp, ref_batch)
                emd_lst.append(emd_batch.view(1, -1))

        cd_lst = torch.cat(cd_lst, dim=1)
        all_cd.append(cd_lst)

        if compute_emd:
            emd_lst = torch.cat(emd_lst, dim=1)
            all_emd.append(emd_lst)
        

    all_cd = torch.cat(all_cd, dim=0)  # N_sample, N_ref
    
    if compute_emd:
        all_emd = torch.cat(all_emd, dim=0)  # N_sample, N_ref
    else:
        all_emd = None

    return all_cd, all_emd


# Adapted from https://github.com/xuqiantong/GAN-Metrics/blob/master/framework/metric.py
def knn(Mxx, Mxy, Myy, k, sqrt=False):
    n0 = Mxx.size(0)
    n1 = Myy.size(0)
    label = torch.cat((torch.ones(n0), torch.zeros(n1))).to(Mxx)
    M = torch.cat((torch.cat((Mxx, Mxy), 1), torch.cat((Mxy.transpose(0, 1), Myy), 1)), 0)
    if sqrt:
        M = M.abs().sqrt()
    INFINITY = float('inf')
    val, idx = (M + torch.diag(INFINITY * torch.ones(n0 + n1).to(Mxx))).topk(k, 0, False)

    count = torch.zeros(n0 + n1).to(Mxx)
    for i in range(0, k):
        count = count + label.index_select(0, idx[i])
    pred = torch.ge(count, (float(k) / 2) * torch.ones(n0 + n1).to(Mxx)).float()

    s = {
        'tp': (pred * label).sum(),
        'fp': (pred * (1 - label)).sum(),
        'fn': ((1 - pred) * label).sum(),
        'tn': ((1 - pred) * (1 - label)).sum(),
    }

    s.update({
        'precision': s['tp'] / (s['tp'] + s['fp'] + 1e-10),
        'recall': s['tp'] / (s['tp'] + s['fn'] + 1e-10),
        'acc_t': s['tp'] / (s['tp'] + s['fn'] + 1e-10),
        'acc_f': s['tn'] / (s['tn'] + s['fp'] + 1e-10),
        'acc': torch.eq(label, pred).float().mean(),
    })
    return s


def lgan_mmd_cov(all_dist):
    N_ref = all_dist.size(1)
    min_val_fromsmp, min_idx = torch.min(all_dist, dim=1)
    min_val, _ = torch.min(all_dist, dim=0)
    mmd = min_val.mean()
    mmd_smp = min_val_fromsmp.mean()
    cov = float(min_idx.unique().view(-1).size(0)) / float(N_ref)
    cov = torch.tensor(cov).to(all_dist)
    return {
        'lgan_mmd': mmd,
        'lgan_cov': cov,
        'lgan_mmd_smp': mmd_smp,
    }


def compute_all_metrics(sample_pcs, ref_pcs, batch_size):
    results = {}

    M_rs_cd, M_rs_emd = _pairwise_EMD_CD_(ref_pcs, sample_pcs, batch_size, compute_emd=False)

    res_cd = lgan_mmd_cov(M_rs_cd.t())
    results.update({
        "%s-CD" % k: v for k, v in res_cd.items()
    })

    if M_rs_emd is not None:
        res_emd = lgan_mmd_cov(M_rs_emd.t())
        results.update({
            "%s-EMD" % k: v for k, v in res_emd.items()
        })

    M_rr_cd, M_rr_emd = _pairwise_EMD_CD_(ref_pcs, ref_pcs, batch_size, compute_emd=False)
    M_ss_cd, M_ss_emd = _pairwise_EMD_CD_(sample_pcs, sample_pcs, batch_size, compute_emd=False)

    # 1-NN results
    one_nn_cd_res = knn(M_rr_cd, M_rs_cd, M_ss_cd, 1, sqrt=False)
    results.update({
        "1-NN-CD-%s" % k: v for k, v in one_nn_cd_res.items() if 'acc' in k
    })

    if M_rr_emd is not None:

        one_nn_emd_res = knn(M_rr_emd, M_rs_emd, M_ss_emd, 1, sqrt=False)
        results.update({
            "1-NN-EMD-%s" % k: v for k, v in one_nn_emd_res.items() if 'acc' in k
        })

    return results




def evaluate(args):
    # Set the random seed
    seed_everything(41)

    # Read in and sample the reference data
    # Read the split 
    with open(os.path.join(args.dataset_path,'splits', CAT2ID[args.category], 'test.txt')) as f:
        split_models = f.readlines()
        split_models = [model.rstrip() for model in split_models]


    ref_pcs = []
    for model in split_models:
        data = np.load(os.path.join(args.dataset_path,CAT2ID[args.category], f'{model}.npz'))['vertices']
        ref_pcs.append(np.random.permutation(data)[:args.n_points,:])
    ref_pcs = torch.tensor(ref_pcs).cuda().float() # n_models, n_pts, 3

    # Read in the generated data

    gen_models = glob.glob(os.path.join(args.gen_path, args.category, '*.ply'))
    method = args.gen_path.split(os.sep)[-1] if args.gen_path[-1] != os.sep else args.gen_path.split(os.sep)[-2]
    assert method in ['pointflow', 'ours'], "Method not recognized. Implement the eval code"

    gen_pcs = []
    for model in gen_models:
        if method in ['pointflow']:
            v = pcu.load_mesh_v(model)
            assert v.shape[0] >= args.n_points, "Not enough points were genrated"
            gen_pcs.append(np.random.permutation(v)[:args.n_points,:])

        else:
            v, f = pcu.load_mesh_vf(model)
            f_idx, bc = pcu.sample_mesh_random(v, f, num_samples=args.n_points)
            v_sampled = pcu.interpolate_barycentric_coords(f, f_idx, bc, v)
            gen_pcs.append(v_sampled)

    gen_pcs = torch.tensor(gen_pcs).cuda().float()

    assert ref_pcs.shape[0] == gen_pcs.shape[0], "The number of generated models does not correspond the test set size"

    # Compute metrics
    results = compute_all_metrics(gen_pcs, ref_pcs, args.batch_size)
    results = {k: (v.cpu().detach().item()
                    if not isinstance(v, float) else v) for k, v in results.items()}
    pprint(results)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="path to the preprocessed shapenet dataset")
    parser.add_argument("--gen_path", type=str, required=True, help="path to the generated models")
    parser.add_argument("--category", type=str, required=True, help="category to evaluate")
    parser.add_argument("--n_points", type=int, default=2048, help="Number of points used for evaluation")
    parser.add_argument("--batch_size", type=int, default=50, help="batch size")
    args = parser.parse_args()

    evaluate(args)
