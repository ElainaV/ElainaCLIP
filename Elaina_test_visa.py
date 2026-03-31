import argparse, os, random, numpy as np
from tqdm import tqdm
from tabulate import tabulate
from scipy.ndimage import gaussian_filter

import torch
import torch.nn.functional as F

import AnomalyCLIP_lib
from Elaina_prompt_ensemble import AnomalyCLIP_PromptLearner
from dataset_mvtec import Dataset
from utils   import get_transform
from logger  import get_logger
from visualization import visualizer
from metrics import image_level_metrics, pixel_level_metrics

def setup_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark    = False

def test(args):
    logger = get_logger(args.save_path)

    if torch.cuda.is_available():
        if args.gpu is not None:
            device = torch.device(f"cuda:{args.gpu}")
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    design = {
        "Prompt_length": args.n_ctx,
        "learnabel_text_embedding_depth": args.depth,
        "learnabel_text_embedding_length": args.t_n_ctx,
        "freeze_dynamic": True
    }

    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px",
                                    device=device,
                                    design_details=design)
    model.eval()

    prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), design)
    ckpt = torch.load(args.checkpoint_path, map_location="cpu")
    prompt_learner.load_state_dict(ckpt["prompt_learner"], strict=True)
    prompt_learner.to(device).eval()
    model.to(device)

    model.visual.DAPM_replace(DPAM_layer=20)

    preprocess, target_tf = get_transform(args)
    test_set  = Dataset(root=args.data_path,
                        transform=preprocess,
                        target_transform=target_tf,
                        dataset_name=args.dataset)
    assert args.batch_size == 1, "batch_size 必须为 1 以逐图生成 DynamicPrompt"
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=1,
                                              shuffle=False)
    obj_list = test_set.obj_list

    results = {obj: {"gt_sp": [], "pr_sp": [],
                     "imgs_masks": [], "anomaly_maps": []}
               for obj in obj_list}

    for items in tqdm(test_loader, desc="Test"):
        img  = items["img"].to(device)
        key  = items["cls_name"][0].strip().lower()
        gt_m = items["img_mask"]
        gt_m[gt_m > .5], gt_m[gt_m <= .5] = 1, 0

        results[key]["imgs_masks"].append(gt_m)
        results[key]["gt_sp"].extend(items["anomaly"].cpu())

        with torch.no_grad():

            img_feat, patch_feats = model.encode_image(
                img, args.features_list, DPAM_layer=20)
            img_feat = F.normalize(img_feat, dim=-1)

            prompts, tok_ids, comp_txt = prompt_learner(
                cls_id=None, image_embedding=img_feat)
            txt_feat = model.encode_text_learn(
                prompts, tok_ids, comp_txt).float()
            txt_feat = F.normalize(txt_feat, dim=-1)
            txt_feat = torch.stack(torch.chunk(txt_feat, 2, 0), dim=1)  # [1,2,768]

            probs     = (img_feat @ txt_feat.permute(0, 2, 1) / 0.07).softmax(-1)
            score_abn = probs[:, 0, 1]           # abnormal 概率
            results[key]["pr_sp"].extend(score_abn.cpu())

            amap_stack = []
            for fi, feat in enumerate(patch_feats):
                if fi >= args.feature_map_layer[0]:
                    feat = F.normalize(feat, dim=-1)
                    sim, _ = AnomalyCLIP_lib.compute_similarity(feat, txt_feat[0])
                    sim_map = AnomalyCLIP_lib.get_similarity_map(
                        sim[:, 1:, :], args.image_size)
                    amap_stack.append((sim_map[..., 1] + 1 - sim_map[..., 0]) / 2.)

            amap_gpu = torch.stack(amap_stack).sum(dim=0)            # [H,W] on GPU
            amap_np  = gaussian_filter(amap_gpu.cpu().numpy(), sigma=args.sigma)
            results[key]["anomaly_maps"].append(torch.from_numpy(amap_np))

            # 可视化
            visualizer(items["img_path"], amap_np,
                       args.image_size, args.save_path, key)

    tbl, p_aurocs, p_aupros, i_aurocs, i_aps = [], [], [], [], []
    for obj in obj_list:
        res = results[obj]
        res["imgs_masks"]   = torch.cat(res["imgs_masks"])
        res["anomaly_maps"] = torch.stack(res["anomaly_maps"]).numpy()

        p_auroc = pixel_level_metrics(results, obj, "pixel-auroc")
        p_aupro = pixel_level_metrics(results, obj, "pixel-aupro")
        i_auroc = image_level_metrics(results, obj, "image-auroc")
        i_ap    = image_level_metrics(results, obj, "image-ap")

        tbl.append([obj,
                    f"{p_auroc*100:.1f}",
                    f"{p_aupro*100:.1f}",
                    f"{i_auroc*100:.1f}",
                    f"{i_ap*100:.1f}"])
        p_aurocs.append(p_auroc); p_aupros.append(p_aupro)
        i_aurocs.append(i_auroc); i_aps.append(i_ap)

    tbl.append(["mean",
                f"{np.mean(p_aurocs)*100:.1f}",
                f"{np.mean(p_aupros)*100:.1f}",
                f"{np.mean(i_aurocs)*100:.1f}",
                f"{np.mean(i_aps)*100:.1f}"])

    logger.info("\n%s", tabulate(
        tbl,
        headers=["object", "pix_AUROC", "pix_AUPRO", "img_AUROC", "img_AP"],
        tablefmt="pipe"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Elaina-AnomalyCLIP-Test", add_help=True)

    parser.add_argument("--data_path",  type=str,
        default="/home/wyh/Data/visa")

    parser.add_argument("--save_path",  type=str,
        default="./submittest/results/visa")
    parser.add_argument("--checkpoint_path", type=str,
        default="./checkpoints/epoch_20.pth")

    parser.add_argument("--dataset", type=str, default="visa")
    parser.add_argument("--depth",   type=int, default=9)
    parser.add_argument("--n_ctx",   type=int, default=16)
    parser.add_argument("--t_n_ctx", type=int, default=8)
    parser.add_argument("--features_list", type=int, nargs="+",
                        default=[6, 12, 18, 24])
    parser.add_argument("--feature_map_layer", type=int, nargs="+",
                        default=[0, 1, 2, 3])

    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--sigma",      type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed",       type=int, default=111)

    parser.add_argument("--gpu", type=int, default=None,
        help="选择使用的GPU编号，例如 0 表示 cuda:0；默认自动选择可用GPU")

    args = parser.parse_args()
    setup_seed(args.seed)
    test(args)
