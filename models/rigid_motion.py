import torch

import torch
import torch.nn as nn
import torch.nn.functional as F
class AdaptivePointGenerator(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128, max_deviation=0.5):
        super(AdaptivePointGenerator, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.max_deviation = max_deviation  # 最大偏差，用于约束生成点的位置

    def forward(self, mean, std, num_points):
        # 初始化正态分布的随机点
        z = torch.randn((mean.shape[0], num_points, mean.shape[-1]), device=mean.device)  # [B, num_points, input_dim]
        
        # 初始点乘以标准差再加上均值
        x = z * std.unsqueeze(1) + mean.unsqueeze(1)  # [B, num_points, input_dim]
        # 通过网络逐步调整点的位置
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.tanh(self.fc3(x))
        
        # 调整后的点，并约束偏差
        generated_points = x * std.unsqueeze(1) + mean.unsqueeze(1)
        
        # 对生成的点进行限制，确保不偏离目标均值太远
        min_limit = mean.unsqueeze(1) - self.max_deviation * std.unsqueeze(1)
        max_limit = mean.unsqueeze(1) + self.max_deviation * std.unsqueeze(1)
        generated_points = torch.max(torch.min(generated_points, max_limit), min_limit)
        
        return generated_points
def generate_additional_points(in_valid_cp2, in_valid_vel2, generator=None):
    # 计算现有点的均值和方差，只考虑非零位置
    valid_mask = in_valid_cp2.sum(dim=-1) != 0  # [M, N]
    
    # 计算均值和标准差
    sum_cp = in_valid_cp2.sum(dim=1)  # [M, 3]
    count_cp = valid_mask.sum(dim=1, keepdim=True)  # [M, 1]
    mean_cp = sum_cp / count_cp  # [M, 3]

    # 标准差计算
    sum_sq_diff_cp = ((in_valid_cp2 - mean_cp.unsqueeze(1)) ** 2).sum(dim=1)
    std_cp = torch.sqrt(sum_sq_diff_cp / count_cp) if count_cp.max().item() > 1 else torch.ones_like(mean_cp)  # [M, 3]

    sum_vel = in_valid_vel2.sum(dim=1)  # [M]
    count_vel = valid_mask.sum(dim=1)  # [M]
    mean_vel = sum_vel / count_vel  # [M]

    sum_sq_diff_vel = ((in_valid_vel2 - mean_vel.unsqueeze(1)) ** 2).sum(dim=1)
    std_vel = torch.sqrt(sum_sq_diff_vel / count_vel) if count_vel.max().item() > 1 else torch.ones_like(mean_vel)  # [M]

    # 扩展速度的维度为3D以进行处理
    mean_vel_expanded = mean_vel.unsqueeze(-1).repeat(1, 3)  # [M, 3]
    std_vel_expanded = std_vel.unsqueeze(-1).repeat(1, 3)  # [M, 3]

    # 生成新的点云位置
    generated_cp = generator(mean_cp, std_cp, 8)

    # 生成新的速度
    generated_vel_expanded = generator(mean_vel_expanded, std_vel_expanded, 8)

    # 从3D向量恢复为标量速度
    generated_vel = generated_vel_expanded.mean(dim=-1)  # [M, N]

    # 仅填充原始为0的部分
    new_cp = in_valid_cp2.clone()
    new_vel = in_valid_vel2.clone()

    zero_mask = (new_cp.sum(dim=-1) == 0)  # [M, N]
    zero_mask_vel = (new_vel == 0)  # [M, N]

    new_cp[zero_mask] = generated_cp[zero_mask]
    new_vel[zero_mask_vel] = generated_vel[zero_mask_vel]

    return new_cp, new_vel
def find_seeds_with_sufficient_neighbors(distances, R, min_count=8):
    # 计算在半径 R 内的布尔掩码
    within_radius = distances < R  # [B, N, N]

    # 统计每个种子点在半径 R 内的邻居数量
    neighbor_count = torch.sum(within_radius, dim=-1)  # [B, N]

    # 生成足够邻居和不足邻居的掩码
    sufficient_neighbor_idx_mask = neighbor_count >= min_count  # [B, N]
    insufficient_neighbor_idx_mask = ~sufficient_neighbor_idx_mask  # [B, N]

    # 直接获取满足条件的邻居索引（通过布尔掩码选择）
    sufficient_neighbor_idx = torch.nonzero(sufficient_neighbor_idx_mask, as_tuple=False)
    insufficient_neighbor_idx = torch.nonzero(insufficient_neighbor_idx_mask, as_tuple=False)

    return sufficient_neighbor_idx_mask, sufficient_neighbor_idx, insufficient_neighbor_idx, within_radius
def custom_cluster_pc1(pc1, pc2, ft1, ft2, vel1, vel2, mask1, R=3, min_count=8, generator=None):
    # print(f'径向速度是{vel1[0]}')
    device = pc1.device
    B, N, _ = pc1.shape
    # 设置特殊的占位值，确保不影响计算
    placeholder_value = 1000

    # 处理位置和速度，将mask1为False的点设置为特殊值
    masked_pc1 = torch.where(mask1.unsqueeze(-1), pc1, torch.full_like(pc1, placeholder_value))  # [B, N, 3]
    masked_vel1 = torch.where(mask1, vel1, torch.full_like(vel1, placeholder_value))  # [B, N]

    target_num_points = 8

    # 计算距离
    diff = pc1.unsqueeze(2) - masked_pc1.unsqueeze(1)  # [B, N, N, 3]
    distances = torch.norm(diff, dim=-1)  # [B, N, N]
    # print(distances[0][44])
    sufficient_neighbor_idx_mask, sufficient_neighbor_idx, insufficient_neighbor_idx, within_radius = find_seeds_with_sufficient_neighbors(distances, R, min_count=min_count)
    # print(insufficient_neighbor_idx)
    # 初始化最终输出张量
    final_cluster_coords = torch.zeros(B, N, min_count, 3, device=device)  # [B, N, 8, 3]
    final_cluster_vel = torch.zeros(B, N, min_count, device=device)  # [B, N, 8]
    # print(f'0位置的掩码{within_radius[0][15]}')
    if sufficient_neighbor_idx.numel() > 0:
        b_sufficient = sufficient_neighbor_idx[:, 0]
        n_sufficient = sufficient_neighbor_idx[:, 1]

        # 获取邻居掩码
        neighbors_mask = within_radius[b_sufficient, n_sufficient]  # [857, N]

        # 选出对应的邻居点的速度和坐标
        valid_vel1 = masked_vel1[b_sufficient]  # [857, N]
        # print(f'valid_vel1的shape是{valid_vel1[0][15]}')
        valid_vel1[~neighbors_mask] = -100 # 将不符合条件的点的速度设为-1

        seeds_vel = vel1[b_sufficient, n_sufficient].unsqueeze(1)  # [857, 1]

        # 计算速度差
        velocity_diff = torch.abs(valid_vel1.unsqueeze(1) - seeds_vel.unsqueeze(2))  # [857, 1, N]

        # 选择速度差最小的8个点
        topk_diff, topk_indices = torch.topk(velocity_diff, k=min_count, dim=2, largest=False)  # [857, 1, 8]
        topk_indices = topk_indices.squeeze(1)  # [857, 8]

        # 获取对应的速度和坐标
        selected_vel1 = valid_vel1.gather(1, topk_indices)  # [857, 8]
        selected_pc1 = masked_pc1[b_sufficient].gather(1, topk_indices.unsqueeze(-1).expand(-1, -1, 3))  # [857, 8, 3]

            # 将结果放入最终张量中
        final_cluster_vel[b_sufficient, n_sufficient] = selected_vel1
        final_cluster_coords[b_sufficient, n_sufficient] = selected_pc1
        # print('第一阶段结束')

    if insufficient_neighbor_idx.numel() > 0:
        b_insufficient = insufficient_neighbor_idx[:, 0]
        n_insufficient = insufficient_neighbor_idx[:, 1]

        valid_pc1_list = []
        valid_vel1_list = []

        for b_idx, n_idx in zip(b_insufficient, n_insufficient):
            valid_pc1 = pc1[b_idx][within_radius[b_idx, n_idx]]
            valid_vel1 = vel1[b_idx][within_radius[b_idx, n_idx]]

            # 直接填充邻居点数量到8个
            pad_size = target_num_points - valid_pc1.shape[0]
            
            valid_pc1 = F.pad(valid_pc1, (0, 0, 0, pad_size), mode='constant', value=0)  # 填充 [N, 3] 维度
            valid_vel1 = F.pad(valid_vel1, (0, pad_size), mode='constant', value=0)  # 填充 [N] 维度
            valid_pc1_list.append(valid_pc1)

            valid_vel1_list.append(valid_vel1)

        # 将处理后的有效点列表转换为张量
        valid_pc1_tensor = torch.stack(valid_pc1_list)  # [M, target_num_points, 3]
        valid_vel1_tensor = torch.stack(valid_vel1_list)  # [M, target_num_points]

        # 生成额外的点使其满足 min_count
        generated_cp, generated_vel = generate_additional_points(
            valid_pc1_tensor, valid_vel1_tensor, generator=generator)

        # 将生成的点加入到最终的张量中
        final_cluster_coords[b_insufficient, n_insufficient] = generated_cp
        final_cluster_vel[b_insufficient, n_insufficient] = generated_vel
    # print(f'初步聚类坐标{pc1[0][within_radius[0,44]]}')
    # print(f'初步聚类速度{vel1[0][within_radius[0,44]]}')
    # print(f'最终聚类坐标{final_cluster_coords[0][44]}')
    # print(f'最终聚类速度{final_cluster_vel[0][44]}')
    # print(f'原坐标{pc1[0][44]}')
    # print(f'原速度{vel1[0][44]}')
   
    return final_cluster_coords, final_cluster_vel