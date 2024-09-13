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
        self.fc_z1=nn.Linear(256,hidden_dim)
        self.fc_z2=nn.Linear(hidden_dim,8)
        self.bn_z = nn.BatchNorm1d(8)

    def forward(self, mean, std, num_points,gfeat):

       
            # 初始化正态分布的随机点
        if gfeat is None:
            z = torch.randn((mean.shape[0], num_points, mean.shape[-1]), device=mean.device)  # [B, num_points, input_dim]
            z = torch.sigmoid(z)  # 将 z 限制在 0 到 1 之间
           
        else:
            gfeat = gfeat.unsqueeze(1).expand(-1, 3, -1)  # [B, 3, 256]
            # 通过全连接层将 gfeat 映射到更高的维度
            z = self.relu(self.fc_z1(gfeat))  # [3790, 8, hidden_dim]
            z = self.relu(self.fc_z2(z))  # [3790, 8, input_dim]
            z=z.permute(0,2,1)
            z = torch.sigmoid(z)  # 将 z 限制在 0 到 1 之间

        # # 初始点乘以标准差再加上均值
        # x = z * std.unsqueeze(1) + mean.unsqueeze(1)  # [B, num_points, input_dim]
        # # 通过网络逐步调整点的位置
        # x = self.relu(self.fc1(x))
        # x = self.relu(self.fc2(x))
        # x = self.tanh(self.fc3(x))
        
        # 调整后的点，并约束偏差
        generated_points = z* std.unsqueeze(1) + mean.unsqueeze(1)
        
        # 对生成的点进行限制，确保不偏离目标均值太远
        min_limit = mean.unsqueeze(1) - self.max_deviation * std.unsqueeze(1)
        max_limit = mean.unsqueeze(1) + self.max_deviation * std.unsqueeze(1)
        generated_points = torch.max(torch.min(generated_points, max_limit), min_limit)
        
        return generated_points
def generate_additional_points(in_valid_cp2, in_valid_vel2, generator=None,gfeat=None):
    # 计算现有点的均值和方差，只考虑非零位置
    valid_mask = in_valid_cp2.sum(dim=-1) != 0  # [M, N]
    # 计算均值和标准差
    sum_cp = in_valid_cp2.sum(dim=1)  # [M, 3]
    count_cp = valid_mask.sum(dim=1, keepdim=True)  # [M, 1]
    mean_cp = sum_cp / count_cp  # [M, 3]
   


    # 标准差计算
    diff_cp = (in_valid_cp2 - mean_cp.unsqueeze(1))  # [M, N, 3]
    masked_diff_cp = diff_cp * valid_mask.unsqueeze(-1)  # 仅保留有效位置的差值
    sum_sq_diff_cp = (masked_diff_cp ** 2).sum(dim=1)  # [M, 3]
    std_cp = torch.sqrt(sum_sq_diff_cp / count_cp) if count_cp.max().item() > 1 else torch.ones_like(mean_cp)  # [M, 3]

    # 找到 std_cp 为 0 的位置，并给它加上 0-1 的小噪声
    zero_mask_cp = (std_cp == 0)  # 找到 std_cp 中为 0 的位置
    random_noise_cp = torch.rand_like(std_cp) * zero_mask_cp  # 生成在 0-1 之间的随机噪声
    std_cp = std_cp + random_noise_cp  # 给 std_cp 为 0 的地方加上随机噪声


    sum_vel = in_valid_vel2.sum(dim=1)  # [M]
    count_vel = valid_mask.sum(dim=1)  # [M]
    mean_vel = sum_vel / count_vel  # [M]

    
    diff_vel = (in_valid_vel2 - mean_vel.unsqueeze(1))  # [M, N]
    masked_diff_vel = diff_vel * valid_mask  # 仅保留有效位置的差值
    sum_sq_diff_vel = (masked_diff_vel ** 2).sum(dim=1)  # [M]
    # 计算标准差
    std_vel = torch.sqrt(sum_sq_diff_vel / count_vel) if count_vel.max().item() > 1 else torch.ones_like(mean_vel)  # [M]

    # 找到 std_vel 为 0 的位置，并给它加上 0-1 的小噪声
    zero_mask = (std_vel == 0)
    random_noise = torch.rand_like(std_vel) * zero_mask  # 生成在 0-1 之间的随机噪声
    std_vel = std_vel + random_noise  # 给 std_vel 为 0 的地方加上随机噪声

    # 扩展速度的维度为3D以进行处理
    mean_vel_expanded = mean_vel.unsqueeze(-1).repeat(1, 3)  # [M, 3]
    std_vel_expanded = std_vel.unsqueeze(-1).repeat(1, 3)  # [M, 3]
 
    # 生成新的点云位置
    generated_cp = generator(mean_cp, std_cp, 8,gfeat)

    # 生成新的速度
    generated_vel_expanded = generator(mean_vel_expanded, std_vel_expanded, 8,gfeat)

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
def custom_cluster_pc1(pc1, pc2, ft1, ft2, vel1, vel2, mask1, R=3, min_count=8, generator=None,gfeat=None):
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

        # 过滤有效邻居点的速度
        valid_vel1 = masked_vel1[b_sufficient]  # [857, N]
        # print(valid_vel1.shape)
        valid_vel1[~neighbors_mask] = -100  # 无效点设为-100

        seeds_vel = vel1[b_sufficient, n_sufficient].unsqueeze(1)  # [857, 1]

        # 计算速度差
        velocity_diff = torch.abs(valid_vel1.unsqueeze(1) - seeds_vel.unsqueeze(2))  # [857, 1, N]
        # print(f'速度差的shape是{velocity_diff.shape}')

        # 设定速度差异的阈值
        velocity_threshold = 3
        valid_velocity_mask = velocity_diff < velocity_threshold
        # print(f'mask的shape是{valid_velocity_mask.shape}')
        valid_velocity_count = valid_velocity_mask.sum(dim=2)
        # print(f'count的shape是{valid_velocity_count.shape}')

        # 批量执行 topk 操作和 gather
        topk_indices_all = torch.zeros((b_sufficient.shape[0], min_count), device=device, dtype=torch.long)  # [857, 8]

        # 只对满足条件的点计算 topk
        valid_indices = valid_velocity_count >= min_count
        valid_indices = valid_indices.squeeze()  # 将 [2327, 1] 转为 [2327]
       
        topk_indices_all[valid_indices] = torch.topk(velocity_diff[valid_indices], k=min_count, dim=2, largest=False)[1].squeeze(1)
        

        # 对满足条件的点批量执行 gather 操作
       # 正确的 gather 操作
        selected_vel1 = valid_vel1.gather(1, topk_indices_all)  # 直接使用二维索引 [2327, 8]
        
        selected_pc1 = masked_pc1[b_sufficient].gather(1, topk_indices_all.unsqueeze(-1).expand(-1, -1, 3))  # [857, 8, 3]

        # 更新结果到 final_cluster_vel 和 final_cluster_coords
        final_cluster_vel[b_sufficient, n_sufficient] = selected_vel1
        final_cluster_coords[b_sufficient, n_sufficient] = selected_pc1

        # 处理不满足条件的点，将它们加入 insufficient_neighbor_idx
        insufficient_mask = valid_velocity_count < min_count
        insufficient_mask = insufficient_mask.squeeze()  # 去掉多余的维度
        b_insufficient_add = b_sufficient[insufficient_mask]
        n_insufficient_add = n_sufficient[insufficient_mask]
        if b_insufficient_add.numel() > 0:
            new_insufficient_idx = torch.stack([b_insufficient_add, n_insufficient_add], dim=1)
            insufficient_neighbor_idx = torch.cat([insufficient_neighbor_idx, new_insufficient_idx], dim=0)

       
    if insufficient_neighbor_idx.numel() > 0:
        # print(insufficient_neighbor_idx.shape)
        # # 假设 insufficient_neighbor_idx 是一个二维张量，形状为 [B, 2]
        # target_value = torch.tensor([3, 88], device=insufficient_neighbor_idx.device)

        # # 使用 torch.where 找到与 [3, 88] 匹配的索引
        # matching_idx = torch.where((insufficient_neighbor_idx == target_value).all(dim=1))[0]

        # # 检查是否找到了匹配的索引
        # if matching_idx.numel() > 0:
        #     print(f'找到的 [3, 88] 在 insufficient_neighbor_idx 中的序号是: {matching_idx.item()}')
        # else:
        #     print('[3, 88] 不存在于 insufficient_neighbor_idx 中')
       
        b_insufficient = insufficient_neighbor_idx[:, 0]
        n_insufficient = insufficient_neighbor_idx[:, 1]

        valid_pc1_list = []
        valid_vel1_list = []
        valid_gfeat_list = []  # 用于收集每个点的 gfeat

        for b_idx, n_idx in zip(b_insufficient, n_insufficient):
            valid_pc1 = pc1[b_idx][within_radius[b_idx, n_idx]]
            valid_vel1 = vel1[b_idx][within_radius[b_idx, n_idx]]



             # 获取该点的原始径向速度
            root_vel = vel1[b_idx, n_idx]

            # 计算速度差异
            velocity_diff = torch.abs(valid_vel1 - root_vel)

            # 设定速度差异的阈值，例如设置为0.1
            velocity_threshold = 3
            valid_mask = velocity_diff < velocity_threshold

            # 只保留与 root_vel 速度接近的点
            valid_pc1 = valid_pc1[valid_mask]
            valid_vel1 = valid_vel1[valid_mask]
               # 处理 gfeat 并添加到列表中
            
            if gfeat is not None:
                
                current_gfeat = gfeat[b_idx]
               
                valid_gfeat_list.append(current_gfeat)  # 将当前 gfeat 添加到列表中
                
            # 直接填充邻居点数量到8个
            pad_size = target_num_points - valid_pc1.shape[0]
            
            valid_pc1 = F.pad(valid_pc1, (0, 0, 0, pad_size), mode='constant', value=0)  # 填充 [N, 3] 维度
            valid_vel1 = F.pad(valid_vel1, (0, pad_size), mode='constant', value=0)  # 填充 [N] 维度
            valid_pc1_list.append(valid_pc1)
            valid_vel1_list.append(valid_vel1)

        # 将处理后的有效点列表转换为张量
        valid_pc1_tensor = torch.stack(valid_pc1_list)  # [M, target_num_points, 3]
        valid_vel1_tensor = torch.stack(valid_vel1_list)  # [M, target_num_points]

         # 将 gfeat 列表堆叠为张量
        if gfeat is not None and len(valid_gfeat_list) > 0:
            valid_gfeat_tensor = torch.stack(valid_gfeat_list)  # [M, gfeat_dim]，具体维度根据 gfeat 的形状
            print(valid_gfeat_tensor.shape)  # 获取对应的 gfeat
        else:
            valid_gfeat_tensor = None  # 如果没有 gfeat，则设置为 None
        # 生成额外的点使其满足 min_count
        generated_cp, generated_vel = generate_additional_points(
            valid_pc1_tensor, valid_vel1_tensor, generator=generator,gfeat=valid_gfeat_tensor)

        # 将生成的点加入到最终的张量中
        final_cluster_coords[b_insufficient, n_insufficient] = generated_cp
        final_cluster_vel[b_insufficient, n_insufficient] = generated_vel
        # print(f'初步聚类坐标{pc1[3][within_radius[3,88]]}')
        # print(f'初步聚类速度{vel1[3][within_radius[3,88]]}')
        # print(f'最终聚类坐标{final_cluster_coords[3][88]}')
        # print(f'最终聚类速度{final_cluster_vel[3][88]}')
        # print(f'原坐标{pc1[3][88]}')
        # print(f'原速度{vel1[3][88]}')

    return final_cluster_coords,final_cluster_vel