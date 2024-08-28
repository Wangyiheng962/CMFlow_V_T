import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np
from lib import pointnet2_utils as pointutils
from models.rigid_motion import custom_cluster_pc1
def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    dist = torch.maximum(dist,torch.zeros(dist.size()).cuda())
    return dist


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def index_points_group(points, knn_idx):
    """
    Input:
        points: input points data, [B, N, C]
        knn_idx: sample index data, [B, N, K]
    Return:
        new_points:, indexed points data, [B, N, K, C]
    """
    points_flipped = points.permute(0, 2, 1).contiguous()
    new_points = pointutils.grouping_operation(points_flipped, knn_idx.int()).permute(0, 2, 3, 1)

    return new_points

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)
    return group_idx

def rigid_velocity_regression(final_cluster_coords, final_cluster_vel, vel1, pc1):
    B, N, K, _ = final_cluster_coords.shape  # [B, N, K, 3]

    # 归一化径向向量
    u = final_cluster_coords / torch.norm(final_cluster_coords, dim=-1, keepdim=True)  # [B, N, K, 3]

    # 构造矩阵 A 和向量 b
    A = u  # [B, N, K, 3]
    b_vec = final_cluster_vel  # [B, N, K]

    # 初次计算 A^T * A 和 A^T * b
    ATA = torch.matmul(A.transpose(-1, -2), A)  # [B, N, 3, 3]
    ATb = torch.matmul(A.transpose(-1, -2), b_vec.unsqueeze(-1))  # [B, N, 3, 1]

    # 正则化系数以避免数值不稳定
    epsilon = 1e-6
    ATA += epsilon * torch.eye(ATA.size(-1), device=ATA.device).unsqueeze(0)

    # 初次计算未加权的最小二乘解 v_world
    v_world, _ = torch.solve(ATb, ATA)
    v_world = v_world.squeeze(-1)  # [B, N, 3]

    # 验证步骤（未加权）
    u_pc1 = pc1 / torch.norm(pc1, dim=-1, keepdim=True)  # [B, N, 3]
    reconstructed_vel_unweighted = torch.sum(v_world * u_pc1, dim=-1)  # [B, N]

    # 计算与原始 vel1 的误差（未加权）
    error_unweighted = torch.abs(reconstructed_vel_unweighted - vel1)  # [B, N]
    mean_error_unweighted = torch.mean(error_unweighted)  # [1]
    max_error_unweighted = torch.max(error_unweighted)  # [1]

    print(f"Unweighted - Mean error: {mean_error_unweighted.item()}, Max error: {max_error_unweighted.item()}")

    # # 计算初始残差
    # residual = b_vec - torch.sum(A * v_world.unsqueeze(-2), dim=-1)  # [B, N, K]
    
    # # 设置残差阈值
    # residual_threshold = 1.0

    # # 计算权重：使用残差的倒数（加一个小量以避免除零），当残差大于阈值时
    # residual_weights = torch.where(
    #     torch.abs(residual) > residual_threshold,
    #     1.0 / (torch.abs(residual) + epsilon),
    #     torch.ones_like(residual)  # 如果残差小于或等于阈值，权重设为1
    # )  # [B, N, K]
    
    # # 加权 A 和 b
    # weighted_A = A * residual_weights.unsqueeze(-1)  # [B, N, K, 3]
    # weighted_b = b_vec * residual_weights  # [B, N, K]

    # # 重新计算加权的 A^T * A 和 A^T * b
    # ATA_weighted = torch.matmul(weighted_A.transpose(-1, -2), weighted_A)  # [B, N, 3, 3]
    # ATb_weighted = torch.matmul(weighted_A.transpose(-1, -2), weighted_b.unsqueeze(-1))  # [B, N, 3, 1]

    # # 正则化加权的 ATA 矩阵
    # ATA_weighted += epsilon * torch.eye(ATA_weighted.size(-1), device=ATA_weighted.device).unsqueeze(0)

    # # 计算加权最小二乘解 v_world_weighted
    # v_world_weighted, _ = torch.solve(ATb_weighted, ATA_weighted)
    # v_world_weighted = v_world_weighted.squeeze(-1)  # [B, N, 3]

    # # 验证步骤（加权）
    # reconstructed_vel_weighted = torch.sum(v_world_weighted * u_pc1, dim=-1)  # [B, N]
    
    # # 计算与原始 vel1 的误差（加权）
    # error_weighted = torch.abs(reconstructed_vel_weighted - vel1)  # [B, N]
    # mean_error_weighted = torch.mean(error_weighted)  # [1]
    # max_error_weighted = torch.max(error_weighted)  # [1]

    # print(f"Weighted - Mean error: {mean_error_weighted.item()}, Max error: {max_error_weighted.item()}")

    return v_world


class MultiScaleEncoder(nn.Module):
    def __init__(self, radius, nsample, in_channel, mlp, mlp2):
        super(MultiScaleEncoder, self).__init__()

        self.ms_ls = nn.ModuleList()
        num_sas = len(radius)
        for l in range(num_sas):
            self.ms_ls.append(PointLocalFeature(radius[l], \
                                    nsample[l],in_channel=in_channel, mlp=mlp, mlp2=mlp2))
                
    def forward(self, xyz, features):
        
        new_features = torch.zeros(0).cuda()
        
        for i, sa in enumerate(self.ms_ls):
            new_features = torch.cat((new_features,sa(xyz,features)),dim=1)
            
        return new_features
        
    
class PointLocalFeature(nn.Module):
    def __init__(self, radius, nsample, in_channel, mlp, mlp2):
        super(PointLocalFeature, self).__init__()
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.mlp2_convs = nn.ModuleList()
        self.mlp2_bns = nn.ModuleList()
        last_channel = in_channel+3
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1, bias = False))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
            
        last_channel = mlp[-1]
        for out_channel in mlp2:
            self.mlp2_convs.append(nn.Conv2d(last_channel, out_channel, 1, bias = False))
            self.mlp2_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
    
        self.queryandgroup = pointutils.QueryAndGroup(radius, nsample)

    def forward(self, xyz, points):
  
        device = xyz.device
        B, C, N = xyz.shape
        xyz_t = xyz.permute(0, 2, 1).contiguous()
        new_points = self.queryandgroup(xyz_t, xyz_t, points)
        
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, -1)[0].unsqueeze(2)
        
        for i, conv in enumerate(self.mlp2_convs):
            bn = self.mlp2_bns[i]
            new_points =  F.relu(bn(conv(new_points)))
        new_points = new_points.squeeze(2)
        
        return new_points

class FeatureCorrelator(nn.Module):
    def __init__(self, nsample, in_channel, mlp, bn = False, use_leaky = True):
        super(FeatureCorrelator, self).__init__()
        self.nsample = nsample
        self.bn = bn
        self.w_xyz = nn.Parameter(torch.tensor(0.4))
        self.w_vel = nn.Parameter(torch.tensor(0.2))
        self.w_points = nn.Parameter(torch.tensor(0.4))
        self.mlp_convs = nn.ModuleList()
        if bn:
            self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            if bn:
                self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        self.weightnet1 = WeightNet(3, last_channel)
        self.weightnet2 = WeightNet(3, last_channel)

        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(0.1, inplace=True)


    def forward(self, xyz1, xyz2, points1, points2,vel1,vel2,mask1,mask2,generator):
        """
        Cost Volume layer for Flow Estimation
        Input:
            xyz1: input points position data, [B, C, N1]
            xyz2: input points position data, [B, C, N2]
            points1: input points data, [B, D, N1]
            points2: input points data, [B, D, N2]
        Return:
            new_points: upsample points feature data, [B, D', N1]
        """
        # import ipdb; ipdb.set_trace()
        # print("xyz1 shape:", xyz1.shape, "device:", xyz1.device)
        # print("xyz2 shape:", xyz2.shape, "device:", xyz2.device)
        # print("points1 shape:", points1.shape, "device:", points1.device)
        # print("points2 shape:", points2.shape, "device:", points2.device)
        # print("vel1 shape:", vel1.shape, "device:", vel1.device)
        # print("vel2 shape:", vel2.shape, "device:", vel2.device)
        # print("mask1 shape:", mask1.shape, "device:", mask1.device)
        # print("mask2 shape:", mask2.shape, "device:", mask2.device)
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points1 = points1.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)
        final_cluster_coords, final_cluster_vel = custom_cluster_pc1(
        pc1=xyz1, 
        pc2=xyz2, 
        ft1=points1, 
        ft2=points2, 
        vel1=vel1, 
        vel2=vel2, 
        mask1=mask1, 
        R=3, 
        min_count=8,
        generator=generator)
        rigid_velocities = rigid_velocity_regression(final_cluster_coords, final_cluster_vel, vel1,xyz1)
        print(rigid_velocities.shape)
        # 假设权重 w_xyz, w_points, w_vel 是在类的初始化中定义的
        weighted_xyz1 = self.w_xyz * xyz1  # [B, N, C]
        weighted_points1 = self.w_points * points1  # [B, N, C]
        weighted_vel1 = self.w_vel * vel1.unsqueeze(-1)  # [B, N, 1]

        weighted_xyz2 = self.w_xyz * xyz2  # [B, N, C]
        weighted_points2 = self.w_points * points2  # [B, N, C]
        weighted_vel2 = self.w_vel * vel2.unsqueeze(-1)  # [B, N, 1]

        # 拼接加权后的特征
        features1 = torch.cat((weighted_xyz1, weighted_points1), dim=-1)  # [B, N, C + C + 1]
        features2 = torch.cat((weighted_xyz2, weighted_points2), dim=-1)  # [B, N, C + C + 1]
        knn_idx = knn_point(self.nsample, features2, features1) # B, N1, nsample
        neighbor_xyz = index_points_group(xyz2, knn_idx)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)

        grouped_points2 = index_points_group(points2, knn_idx) # B, N1, nsample, D2
        grouped_points1 = points1.view(B, N1, 1, D1).repeat(1, 1, self.nsample, 1)
        new_points = torch.cat([grouped_points1, grouped_points2, direction_xyz], dim = -1) # B, N1, nsample, D1+D2+3
        new_points = new_points.permute(0, 3, 2, 1) # [B, D1+D2+3, nsample, N1]
        for i, conv in enumerate(self.mlp_convs):
            if self.bn:
                bn = self.mlp_bns[i]
                new_points =  self.relu(bn(conv(new_points)))
            else:
                new_points =  self.relu(conv(new_points))

        # weighted sum
        weights = self.weightnet1(direction_xyz.permute(0, 3, 2, 1)) # B C nsample N1 

        point_to_patch_cost = torch.sum(weights * new_points, dim = 2) # B C N

        # Patch to Patch Cost
        knn_idx = knn_point(self.nsample, rigid_velocities, rigid_velocities) # B, N1, nsample
        neighbor_xyz = index_points_group(xyz1, knn_idx)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)

        # weights for group cost
        weights = self.weightnet2(direction_xyz.permute(0, 3, 2, 1)) # B C nsample N1 
        grouped_point_to_patch_cost = index_points_group(point_to_patch_cost.permute(0, 2, 1), knn_idx) # B, N1, nsample, C
        patch_to_patch_cost = torch.sum(weights * grouped_point_to_patch_cost.permute(0, 3, 2, 1), dim = 2) # B C N

        return patch_to_patch_cost,rigid_velocities

            
class FlowHead(nn.Module):
    def __init__(self, in_channel, mlp):
        super(FlowHead, self).__init__()
        self.sf_mlp = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.sf_mlp.append(nn.Sequential(nn.Conv2d(last_channel, out_channel, 1, bias=False),
                                                 nn.BatchNorm2d(out_channel),
                                                 nn.ReLU(inplace=False)))
            last_channel = out_channel
            
        self.conv2 = nn.Conv2d(mlp[-1], 3, 1, bias=False)
        
    def forward(self, feat):
 
        feat = feat.unsqueeze(3)
        for conv in self.sf_mlp:
            feat = conv(feat)
        
        output = self.conv2(feat)
        
        return output.squeeze(3)

class MotionHead(nn.Module):
    def __init__(self, in_channel, mlp):
        super(MotionHead, self).__init__()
        self.sf_mlp = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.sf_mlp.append(nn.Sequential(nn.Conv2d(last_channel, out_channel, 1, bias=False),
                                                 nn.BatchNorm2d(out_channel),
                                                 nn.ReLU(inplace=False)))
            last_channel = out_channel
            
        self.conv2 = nn.Conv2d(mlp[-1], 1, 1, bias=False)
        self.m = nn.Sigmoid()
        
    def forward(self, feat):
 
        feat = feat.unsqueeze(3)
        for conv in self.sf_mlp:
            feat = conv(feat)
        
        output = self.m(self.conv2(feat))
        
        return output.squeeze(3)
    
class WeightNet(nn.Module):

    def __init__(self, in_channel, out_channel, hidden_unit = [8, 8], bn = False):
        super(WeightNet, self).__init__()

        self.bn = bn
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        if hidden_unit is None or len(hidden_unit) == 0:
            self.mlp_convs.append(nn.Conv2d(in_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
        else:
            self.mlp_convs.append(nn.Conv2d(in_channel, hidden_unit[0], 1))
            self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[0]))
            for i in range(1, len(hidden_unit)):
                self.mlp_convs.append(nn.Conv2d(hidden_unit[i - 1], hidden_unit[i], 1))
                self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[i]))
            self.mlp_convs.append(nn.Conv2d(hidden_unit[-1], out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
        
    def forward(self, localized_xyz):
        #xyz : BxCxKxN

        weights = localized_xyz
        for i, conv in enumerate(self.mlp_convs):
            if self.bn:
                bn = self.mlp_bns[i]
                weights =  F.relu(bn(conv(weights)))
            else:
                weights = F.relu(conv(weights))

        return weights


class FlowDecoder(nn.Module):
    def __init__(self, fc_inch):
        super(FlowDecoder, self).__init__()
        ## multi-scale flow embeddings propogation
        # different scale share the same mlps hyper-parameters
        ep_radius = [2.0, 4.0, 8.0, 16.0]
        ep_nsamples = [4, 8, 16, 32]
        ep_inch = fc_inch * 2 + 3
        ep_mlps = [fc_inch, int(fc_inch/2), int(fc_inch/8)]
        ep_mlp2s = [int(fc_inch/8), int(fc_inch/8), int(fc_inch/8)]
        num_eps = len(ep_radius)
        self.mse = MultiScaleEncoder(ep_radius, ep_nsamples, in_channel=ep_inch, \
                                         mlp = ep_mlps, mlp2 = ep_mlp2s)
        ## scene flow predictor
        sf_inch = num_eps * ep_mlp2s[-1]*2
        sf_mlps = [int(sf_inch/2), int(sf_inch/4), int(sf_inch/8)]
        self.fp = FlowPredictor(in_channel=sf_inch, mlp=sf_mlps)
        
    def forward(self, pc1, feature1, pc1_features, cor_features):
        
        embeddings = torch.cat((feature1, pc1_features, cor_features),dim=1)
        ## multi-scale flow embeddings propogation
        prop_features = self.mse(pc1,embeddings)
        gfeat = torch.max(prop_features,-1)[0].unsqueeze(2).expand(prop_features.size()[0],prop_features.size()[1],pc1.size()[2])
        final_features = torch.cat((prop_features, gfeat),dim=1)
        
        ## initial scene flow prediction
        output = self.fp(final_features)
        
        return output
    

class Decoder(nn.Module):
    def __init__(self, fc_inch):
        super(Decoder, self).__init__()
        ## multi-scale flow embeddings propogation
        # different scale share the same mlps hyper-parameters
        ep_radius = [2.0, 4.0, 8.0, 16.0]
        ep_nsamples = [4, 8, 16, 32]
        ep_inch = fc_inch * 2 + 3
        ep_mlps = [fc_inch, int(fc_inch/2), int(fc_inch/8)]
        ep_mlp2s = [int(fc_inch/8), int(fc_inch/8), int(fc_inch/8)]
        num_eps = len(ep_radius)
        self.mse = MultiScaleEncoder(ep_radius, ep_nsamples, in_channel=ep_inch, \
                                         mlp = ep_mlps, mlp2 = ep_mlp2s)
        ## scene flow predictor
        sf_inch = num_eps * ep_mlp2s[-1]*2
        sf_mlps = [int(sf_inch/2), int(sf_inch/4), int(sf_inch/8)]
        self.fp = FlowPredictor(in_channel=sf_inch, mlp=sf_mlps)
        self.mp = MotionPredictor(in_channel=sf_inch, mlp=sf_mlps)
        
        
    def forward(self, pc1, feature1, pc1_features, cor_features):
        
        embeddings = torch.cat((feature1, pc1_features, cor_features),dim=1)
        ## multi-scale flow embeddings propogation
        prop_features = self.mse(pc1,embeddings)
        gfeat = torch.max(prop_features,-1)[0].unsqueeze(2).expand(prop_features.size()[0],prop_features.size()[1],pc1.size()[2])
        final_features = torch.cat((prop_features, gfeat),dim=1)
        
        ## initial scene flow prediction
        output = self.fp(final_features)
        static_cls = self.mp(final_features)
        
        return output, static_cls
    
        
class FlowPredictor(nn.Module):
    def __init__(self, in_channel, mlp):
        super(FlowPredictor, self).__init__()
        self.sf_mlp = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.sf_mlp.append(nn.Sequential(nn.Conv2d(last_channel, out_channel, 1, bias=False),
                                                 nn.BatchNorm2d(out_channel),
                                                 nn.ReLU(inplace=False)))
            last_channel = out_channel
            
        self.conv2 = nn.Conv2d(mlp[-1], 3, 1, bias=False)
        
    def forward(self, feat):
 
        feat = feat.unsqueeze(3)
        for conv in self.sf_mlp:
            feat = conv(feat)
        
        output = self.conv2(feat)
        
        return output.squeeze(3)

class MotionPredictor(nn.Module):
    def __init__(self, in_channel, mlp):
        super(MotionPredictor, self).__init__()
        self.sf_mlp = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.sf_mlp.append(nn.Sequential(nn.Conv2d(last_channel, out_channel, 1, bias=False),
                                                 nn.BatchNorm2d(out_channel),
                                                 nn.ReLU(inplace=False)))
            last_channel = out_channel
            
        self.conv2 = nn.Conv2d(mlp[-1], 1, 1, bias=False)
        self.m = nn.Sigmoid()
        
    def forward(self, feat):
 
        feat = feat.unsqueeze(3)
        for conv in self.sf_mlp:
            feat = conv(feat)
        
        output = self.m(self.conv2(feat))
        
        return output.squeeze(3)

