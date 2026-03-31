import os, random, argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F

parser = argparse.ArgumentParser("Elaina-Prompt training", add_help=False)
parser.add_argument("--gpu", type=int, default=0, help="使用哪张GPU显卡")
args_tmp = parser.parse_known_args()[0]
os.environ["CUDA_VISIBLE_DEVICES"] = str(args_tmp.gpu)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import AnomalyCLIP_lib
from dataset_mvtec import Dataset
from utils   import get_transform
from logger  import get_logger
from losselaina    import FocalLoss, BinaryDiceLoss, loss_disentangle, loss_disentangle_cos
from Elaina_prompt_ensemble_oto import AnomalyCLIP_PromptLearner

def setup_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

def train(args):
    logger = get_logger(args.save_path)
    preprocess, target_tf = get_transform(args)

    design = {
        "Prompt_length":                    args.n_ctx,
        "learnabel_text_embedding_depth":   args.depth,
        "learnabel_text_embedding_length":  args.t_n_ctx,
        "freeze_dynamic":                   False
    }

    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px",
                                    device=device,
                                    design_details=design)
    model.eval()
    model.visual.DAPM_replace(DPAM_layer=20)
    train_set = Dataset(args.train_data_path, preprocess, target_tf, args.dataset)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True)

    prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), design)
    prompt_learner.to(device)
    model.to(device)

    optimizer   = torch.optim.Adam(prompt_learner.parameters(),
                                   lr=args.learning_rate,
                                   betas=(0.5, 0.999))
    loss_focal  = FocalLoss()
    loss_dice   = BinaryDiceLoss()

    for ep in tqdm(range(args.epoch), desc="Epoch"):
        model.eval()
        prompt_learner.train()
        reg_loss_log, img_loss_log, dis_loss_log, con_loss_log = [], [], [], []
        for batch in tqdm(train_loader, leave=False):
            img   = batch["img"].to(device)
            label = batch["anomaly"].long().to(device)
            gt    = (batch["img_mask"].squeeze() > .5).float().to(device)

            with torch.no_grad():
                img_feat, patches = model.encode_image(
                    img, args.features_list, DPAM_layer=20)
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

            prompts, tok_ids, comp_txt = prompt_learner(
                cls_id=None, image_embedding=img_feat[0:1])
            # ------------------------------------------------
            txt_feat = model.encode_text_learn(
                prompts, tok_ids, comp_txt).float()
            txt_feat = torch.stack(torch.chunk(txt_feat, 2, 0), dim=1)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

            prob = (img_feat.unsqueeze(1) @
                    txt_feat.permute(0, 2, 1))[:, 0] / 0.07
            img_loss = F.cross_entropy(prob, label)
            img_loss_log.append(img_loss.item())

            reg_loss = 0.0
            for i, patch in enumerate(patches):
                if i >= args.feature_map_layer[0]:
                    patch = patch / patch.norm(dim=-1, keepdim=True)
                    sim, _ = AnomalyCLIP_lib.compute_similarity(patch, txt_feat[0])
                    sim_map = AnomalyCLIP_lib.get_similarity_map(
                        sim[:, 1:], args.image_size).permute(0, 3, 1, 2)
                    reg_loss += loss_focal(sim_map, gt)
                    reg_loss += loss_dice(sim_map[:, 1], gt)
                    reg_loss += loss_dice(sim_map[:, 0], 1 - gt)
            reg_loss_log.append(reg_loss.item())

            dis_loss = 0.0
            con_loss = 0.0
            learnable_ctx = prompt_learner.learn_ctx
            dyn_ctx = prompt_learner.dyn_ctx
            prompts_dict = {
                "learnable": prompts[:, 1 : 1 + learnable_ctx, :],  # [C, L1, D]
                "dynamic":   prompts[:, 1 + learnable_ctx : 1 + learnable_ctx + dyn_ctx, :]  # [C, L2, D]
            }

            prompt_tokens = prompts_dict["learnable"]
            prototypes = prompt_tokens.mean(dim=1)
            patch_feat = patches[0]
            patch_feat = patch_feat / patch_feat.norm(dim=-1, keepdim=True)
            con_feats = patch_feat.mean(dim=1)
            sim_matrix = torch.matmul(con_feats, prototypes.t()) / 0.07
            targets = label.long()

            con_loss = F.cross_entropy(sim_matrix, targets)
            con_loss = 1.0 * con_loss
            con_loss_log.append(con_loss.item())

            dis_loss += loss_disentangle(prompts_dict["dynamic"])
            dis_loss = dis_loss * 0.0003
            dis_loss_log.append(dis_loss.item())
            
            # ----- 反向传播 -----
            optimizer.zero_grad()
            total_loss = img_loss + reg_loss + con_loss + dis_loss
            (total_loss).backward()
            optimizer.step()

        # ---------- 日志 ----------
        if (ep + 1) % args.print_freq == 0:
            logger.info(f"epoch [{ep+1}/{args.epoch}]  "
                        f"region_loss:{np.mean(reg_loss_log):.4f}  "
                        f"img_loss:{np.mean(img_loss_log):.4f}  "
                        f"con_loss:{np.mean(con_loss_log):.4f}  "
                        f"dis_loss:{np.mean(dis_loss_log):.4f}  ")

        # ---------- 保存 ----------
        if (ep + 1) % args.save_freq == 0:
            ckpt = os.path.join(args.save_path, f"epoch_{ep+1}.pth")
            torch.save({"prompt_learner": prompt_learner.state_dict()}, ckpt)


# ============= CLI =============
if __name__ == "__main__":
    # ---------- 新增：合并前面已声明的 parser ----------
    p = argparse.ArgumentParser("Elaina-Prompt training", parents=[parser])

    # 数据 & 保存
    p.add_argument("--train_data_path", type=str,
                   default="/home/wyh/Data/mvtec")
    p.add_argument("--save_path", type=str,  default="./checkpoints/")

    # 模型 / Prompt
    p.add_argument("--dataset",   type=str, default="mvtec")
    p.add_argument("--depth",     type=int, default=9)              
    p.add_argument("--n_ctx",     type=int, default=16)             # 控制 learnable_prompt + Dynamic_prompt 的总长度
    p.add_argument("--t_n_ctx",   type=int, default=8)
    p.add_argument("--features_list", type=int, nargs="+",
                   default=[6, 12, 18, 24])
    p.add_argument("--feature_map_layer", type=int, nargs="+",
                   default=[0, 1, 2, 3])

    # 训练超参
    p.add_argument("--epoch",      type=int, default=20)
    p.add_argument("--learning_rate", type=float, default=5e-4)
    p.add_argument("--batch_size", type=int, default=3)
    p.add_argument("--image_size", type=int, default=700)
    p.add_argument("--print_freq", type=int, default=1)
    p.add_argument("--save_freq",  type=int, default=1)
    p.add_argument("--seed",       type=int, default=111)

    args = p.parse_args()
    setup_seed(args.seed)
    train(args)
