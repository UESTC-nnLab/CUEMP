import math
import sys
sys.path.append('/home/honsen/honsen/VO_SLAM/DROID-SLAM')
import torch
import torch.nn as nn
import torch.nn.functional as F
from droid_slam.modules.extractor import BasicEncoder
from droid_slam.modules.corr import CorrBlock
from droid_slam.modules.gru import ConvGRU, KAN_bias_GRU
from droid_slam.modules.clipping import GradientClip
from droid_slam.geom.ba import BA
import droid_slam.geom.projective_ops as pops
from droid_slam.geom.graph_utils import graph_to_edge_list, keyframe_indicies
from torch_scatter import scatter_mean
def cvx_upsample(data, mask):
    """ upsample pixel-wise transformation field """
    batch, ht, wd, dim = data.shape
    data = data.permute(0, 3, 1, 2)
    mask = mask.view(batch, 1, 9, 8, 8, ht, wd)
    mask = torch.softmax(mask, dim=2)

    up_data = F.unfold(data, [3,3], padding=1)
    up_data = up_data.view(batch, dim, 9, 1, 1, ht, wd)

    up_data = torch.sum(mask * up_data, dim=2)
    up_data = up_data.permute(0, 4, 2, 5, 3, 1)
    up_data = up_data.reshape(batch, 8*ht, 8*wd, dim)

    return up_data

def upsample_disp(disp, mask):
    batch, num, ht, wd = disp.shape
    disp = disp.view(batch*num, ht, wd, 1)
    mask = mask.view(batch*num, -1, ht, wd)
    return cvx_upsample(disp, mask).view(batch, num, 8*ht, 8*wd)


class GraphAgg(nn.Module):
    def __init__(self):
        super(GraphAgg, self).__init__()
        self.conv1 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        self.eta = nn.Sequential(
            nn.Conv2d(128, 1, 3, padding=1),
            GradientClip(),
            nn.Softplus())

        self.upmask = nn.Sequential(
            nn.Conv2d(128, 8*8*9, 1, padding=0))

    def forward(self, net, ii):
        batch, num, ch, ht, wd = net.shape
        net = net.view(batch*num, ch, ht, wd)

        _, ix = torch.unique(ii, return_inverse=True)
        net = self.relu(self.conv1(net))

        net = net.view(batch, num, 128, ht, wd)
        net = scatter_mean(net, ix, dim=1)
        net = net.view(-1, 128, ht, wd)

        net = self.relu(self.conv2(net))

        eta = self.eta(net).view(batch, -1, ht, wd)
        upmask = self.upmask(net).view(batch, -1, 8*8*9, ht, wd)

        return .01 * eta, upmask

class UpdateModule(nn.Module):
    def __init__(self):
        super(UpdateModule, self).__init__()
        cor_planes = 2 * (2*3 + 1)**2

        self.corr_encoder = nn.Sequential(
            nn.Conv2d(cor_planes, 128, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True))

        self.flow_encoder = nn.Sequential(
            nn.Conv2d(4, 128, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True))

        self.weight = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 2, 3, padding=1),
            GradientClip(),
            nn.Sigmoid())

        self.delta = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 2, 3, padding=1),
            GradientClip())

        self.gru = ConvGRU(128, 128+128+64)
        self.agg = GraphAgg()

    def forward(self, net, inp, corr, flow=None, ii=None, jj=None):
        """ RaftSLAM update operator """

        batch, num, ch, ht, wd = net.shape

        if flow is None:
            flow = torch.zeros(batch, num, 4, ht, wd, device=net.device)

        output_dim = (batch, num, -1, ht, wd)
        net = net.view(batch*num, -1, ht, wd)
        inp = inp.view(batch*num, -1, ht, wd)        
        corr = corr.view(batch*num, -1, ht, wd)
        flow = flow.view(batch*num, -1, ht, wd)

        corr = self.corr_encoder(corr)
        flow = self.flow_encoder(flow)
        net = self.gru(net, inp, corr, flow)

        ### update variables ###
        delta = self.delta(net).view(*output_dim)
        weight = self.weight(net).view(*output_dim)

        delta = delta.permute(0,1,3,4,2)[...,:2].contiguous()
        weight = weight.permute(0,1,3,4,2)[...,:2].contiguous()

        net = net.view(*output_dim)

        if ii is not None:
            eta, upmask = self.agg(net, ii.to(net.device))
            return net, delta, weight, eta, upmask

        else:
            return net, delta, weight

class VariancePredictor(nn.Module):
    def __init__(self, input_dim=128):
        """
        从上下文特征中预测每个像素采样时的方差。
        input_dim: 输入特征图的维度 (来自 cnet 的 inp)
        """
        super(VariancePredictor, self).__init__()
        
        # 定义一个简单的网络来预测方差
        self.map = nn.Sequential(
            nn.Conv2d(input_dim, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 3, padding=1) # 输出单通道的方差图
        )

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, inp):
        """
        inp: 上下文特征图, 形状 (B*N, C, H, W)
        返回: variance_map, 形状 (B*N, H, W, 1)
        """
        # 预测的方差值需要被约束在一个合理的范围内
        # 使用 sigmoid 将输出映射到 (0, 1)，然后缩放和平移
        # 例如，将方差限制在 [0.5, 5.5] 的范围内
        variance = self.map(inp)
        variance = torch.sigmoid(variance) * 10.0 + 0.0005
        
        # 调整形状以匹配采样器的输入
        # (B*N, 1, H, W) -> (B*N, H, W, 1)
        variance_map = variance.permute(0, 2, 3, 1).contiguous()
        
        return variance_map

def per_Corr_Normalization(tensor, dims):
    mean = tensor.mean(dim=dims, keepdim=True)
    std = tensor.std(dim=dims, keepdim=True) + 1e-5
    return (tensor - mean) / std

class AdaptiveDilation(nn.Module):
    def __init__(self, in_channels, r, scale=4):
        """
        初始化自适应空洞模块
        Args:
            in_channels (int): 输入特征图的通道数，例如 256
            r (int): 采样网格的半径 (radius)。例如 r=3 会生成一个 7x7 的网格。
        """
        super(AdaptiveDilation, self).__init__()
        
        self.scale = scale
        
        # 1. 根据半径 r 动态生成采样网格
        # 创建从 -r 到 r 的坐标轴
        coords = torch.arange(-r, r + 1, dtype=torch.float32)
        # 使用 meshgrid 创建坐标网格, indexing='ij' 使得 y 在前, x 在后
        y_coords, x_coords = torch.meshgrid(coords, coords, indexing='ij')
        
        # 将坐标堆叠并展平为 [y1, x1, y2, x2, ...] 的形式
        # 最终形状为 (2 * (2r+1)^2,)
        offset_grid = torch.stack([y_coords, x_coords], dim=-1).view(-1)
        
        # 2. 将预处理后的网格注册为 buffer，并移至 __init__ 以提高效率
        # 原始网格形状: [1, 2*(2r+1)^2, 1, 1]
        dilated_offset = offset_grid[None, ..., None, None]
        
        # 预处理：
        # a. 调整形状为 [1, num_points, 2, 1, 1], 其中 num_points = (2r+1)^2
        # b. 交换 x, y 坐标顺序，以匹配常见的 grid_sample 格式 [..., (x, y)]
        # c. 调整回 [1, 2*num_points, 1, 1]
        num_points = (2*r + 1)**2
        processed_offset = dilated_offset.view(1, num_points, 2, 1, 1)
        processed_offset = processed_offset[:, :, [1, 0], :, :] # 交换 x, y
        processed_offset = processed_offset.view(1, -1, 1, 1)
        
        self.register_buffer('dilated_offset', processed_offset)

        # 定义卷积层，输入通道数现在是参数
        out_channels = 2 * num_points # 2 * (2*r+1)^2
        self.ofsMap = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.ofs_residual = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.mask1 = nn.Conv2d(in_channels, out_channels, 1, padding=0)

    def forward(self, t):
        # --- 偏移量和掩码的生成 ---
        n,b,c,h,w = t.size()
        t = t.view(n*b, c, h, w)
         
        mask = torch.sigmoid(self.mask1(t))
        
        offset_fine = self.ofsMap(t)
        
        offset_res = self.ofs_residual(t)
                
        # --- 偏移量融合与缩放 ---
        # 学习到的缩放因子被限制在 (0, 2) 范围内
        scale_first = torch.sigmoid(offset_fine) * self.scale
        scale_res = torch.sigmoid(offset_res) * self.scale
        
        # 融合多尺度缩放因子
        final_scale_first = scale_first
        final_scale_sec = (scale_res + scale_first) / 2
        
        # --- 应用缩放和掩码 ---
        # dilated_offset 已经是预处理好的了，直接使用
        # 乘以 dilation_offset/2 表示coarse offset的缩放范围更小
        offset1 = (final_scale_first * self.dilated_offset) * mask
        offset2 = (final_scale_sec * self.dilated_offset / 2) * mask
        
        # --- 格式化输出 ---
        # 转换为 [B, H, W, C] 格式，通常用于 F.grid_sample
        offset1 = offset1.permute(0, 2, 3, 1)
        offset2 = offset2.permute(0, 2, 3, 1)

        return [offset1, offset2]


    @torch.cuda.amp.autocast(enabled=True)
    def forward_infer(self, t):
        b, c, h, w = t.size()

        # --- 偏移量和掩码的生成 ---
        mask = torch.sigmoid(self.mask1(t))

        offset_fine = self.ofsMap(t)

        t1 = F.avg_pool2d(t, kernel_size=2, stride=2)
        offset_coarse = self.ofs_residual(t1)
        offset_coarse = F.interpolate(offset_coarse, size=(h, w), mode='bilinear', align_corners=False)

        # --- 偏移量融合与缩放 ---
        # 学习到的缩放因子被限制在 (0, 2) 范围内
        scale_fine = torch.sigmoid(per_Corr_Normalization(offset_fine, [1, 2, 3])) * 2
        scale_coarse = torch.sigmoid(per_Corr_Normalization(offset_coarse, [1, 2, 3])) * 2

        # 融合多尺度缩放因子
        final_scale_fine = scale_fine
        final_scale_coarse = (scale_coarse + scale_fine) / 2

        # --- 应用缩放和掩码 ---
        # dilated_offset 已经是预处理好的了，直接使用
        # 乘以 dilation_offset/2 表示coarse offset的缩放范围更小
        offset1 = (final_scale_fine * self.dilated_offset) * mask
        offset2 = (final_scale_coarse * self.dilated_offset / 2) * mask

        # --- 格式化输出 ---
        # 转换为 [B, H, W, C] 格式，通常用于 F.grid_sample
        offset1 = offset1.permute(0, 2, 3, 1)
        offset2 = offset2.permute(0, 2, 3, 1)

        return [offset1, offset2]

class DroidNet(nn.Module):
    def __init__(self):
        super(DroidNet, self).__init__()
        self.fnet = BasicEncoder(output_dim=128, norm_fn='instance') #feature network
        self.cnet = BasicEncoder(output_dim=256, norm_fn='none')    #context network
        self.defDliation = AdaptiveDilation(128, 3)
        self.variance_predictor = VariancePredictor(input_dim=128)
        self.update = UpdateModule()

    def extract_features(self, images):
        """ run feeature extraction networks """

        # normalize images
        images = images[:, :, [2,1,0]] / 255.0
        mean = torch.as_tensor([0.485, 0.456, 0.406], device=images.device)
        std = torch.as_tensor([0.229, 0.224, 0.225], device=images.device)
        images = images.sub_(mean[:, None, None]).div_(std[:, None, None])

        fmaps = self.fnet(images)
        net = self.cnet(images)
        net, inp = net.split([128,128], dim=2)
        net = torch.tanh(net)
        inp = torch.relu(inp)
        return fmaps, net, inp


    def forward(self, Gs, images, disps, intrinsics, graph, num_steps=12, fixedp=2):
        """ Estimates SE3 or Sim3 between pair of frames """

        ii, jj, _ = graph_to_edge_list(graph)

        ii = ii.to(device=images.device, dtype=torch.long)
        jj = jj.to(device=images.device, dtype=torch.long)

        fmaps, net, inp = self.extract_features(images)
        net, inp = net[:,ii], inp[:,ii]

        corr_fn = CorrBlock(self.defDliation, fmaps[:,ii], fmaps[:,jj], num_levels=2, radius=3)

        ht, wd = images.shape[-2:]
        coords0 = pops.coords_grid(ht//8, wd//8, device=images.device)
        
        coords1, _ = pops.projective_transform(Gs, disps, intrinsics, ii, jj)
        target = coords1.clone()
       
        Gs_list, disp_list, residual_list = [], [], []

        b, n, c, h, w = inp.shape
        
        variance_map = self.variance_predictor(inp.view(b*n, c, h, w))
        
        cur_ofs = None
        
        for step in range(num_steps):
            Gs = Gs.detach()
            disps = disps.detach()
            coords1 = coords1.detach()
            target = target.detach()

            # extract motion features

            resd = target - coords1
            flow = coords1 - coords0 #coords1 is pij , coords0 is pi
            
            corr,_ = corr_fn(coords1, variance_map, cur_ofs) #

            motion = torch.cat([flow, resd], dim=-1)
            motion = motion.permute(0,1,4,2,3).clamp(-64.0, 64.0)

            net, delta, weight, eta, upmask = \
                self.update(net, inp, corr, motion, ii, jj)

            target = coords1 + delta

            cur_ofs = self.defDliation(net.view(b * n, -1, h, w))  # b*n, 2r+1, 2r+1
            
            for i in range(2):
                Gs, disps = BA(target, weight, eta, Gs, disps, intrinsics, ii, jj, fixedp=2)

            coords1, valid_mask = pops.projective_transform(Gs, disps, intrinsics, ii, jj)
            residual = (target - coords1)

            Gs_list.append(Gs)
            disp_list.append(upsample_disp(disps, upmask))
            residual_list.append(valid_mask * residual)

        return Gs_list, disp_list, residual_list#, loss


def flow_to_image(flow):
    """
    将光流场 (H, W, 2) 转换为彩色RGB图像 (H, W, 3)。
    """
    if flow.ndim != 3 or flow.shape[2] != 2:
        raise ValueError(f"输入光流的形状应为 [H, W, 2]，但得到的是 {flow.shape}")

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return rgb
def visualize_flow_grid(flow_tensor, graph_edges, save_path):
    import matplotlib.pyplot as plt
    """
    将一批光流场可视化为一个网格图并保存。
    """
    if not isinstance(flow_tensor, torch.Tensor):
        raise TypeError("输入必须是PyTorch张量。")

    flow_np = flow_tensor.cpu().detach().numpy()
    num_edges, h, w, _ = flow_np.shape
    
    ii, jj, _ = graph_to_edge_list(graph)
    
    grid_cols = math.ceil(math.sqrt(num_edges))
    grid_rows = math.ceil(num_edges / grid_cols)
    
    fig, axs = plt.subplots(grid_rows, grid_cols, 
                            figsize=(grid_cols * 3, grid_rows * 3), 
                            dpi=120)
    # 如果只有一个子图，axs不是数组，将其转换为数组以便处理
    if num_edges == 1:
        axs = np.array([axs])
    axs_flat = axs.flatten()
    
    for i in range(num_edges):
        ax = axs_flat[i]
        single_flow = flow_np[i]
        flow_img = flow_to_image(single_flow)
        ax.imshow(flow_img)
        title = f"Flow: {ii[i].item()} -> {jj[i].item()}"
        ax.set_title(title, fontsize=8)
        ax.axis('off')

    for i in range(num_edges, len(axs_flat)):
        axs_flat[i].axis('off')
        
    fig.suptitle('Final Optimized Optical Flow', fontsize=16, weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f"优化后的光流可视化图像已保存至: {save_path}")

if __name__ == "__main__":
    from collections import OrderedDict
    import os
    import cv2
    import numpy as np
    import lietorch
    
    with torch.no_grad():
        adaModel = AdaptiveDilation(128,3).cuda()
        inps = torch.randn(15,128,48,64).cuda()
        inps1 = torch.randn(15,128,48,64).cuda()
        asd = adaModel(inps)
        asd1 = adaModel(inps1)
    # --- 1. 定义测试参数 ---
    NUM_FRAMES = 20
    IMG_HEIGHT, IMG_WIDTH = 384, 512
    # 请将您的模型权重路径替换到这里
    WEIGHTS_PATH = "/home/honsen/gitDCGU/ckpoints/checkpoint_name_150000.pth" 
    # ==================== 主要修改点在这里 ====================
    # 请将您的图片文件夹路径替换到这里
    IMAGE_DIR = "/home/honsen/tartan/test/tartanair-test-mono-release/mono/ME001" # "/home/honsen/gitDCGU/test_img"
    # ========================================================
    NUM_STEPS = 6 # 前向传播的迭代次数

    # 设置运行设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"将在设备上运行: {device}")

    # --- 2. 从指定目录加载图像 ---
    print(f"正在从目录 '{IMAGE_DIR}' 加载图像...")
    
    # 检查目录是否存在
    if not os.path.isdir(IMAGE_DIR):
        raise FileNotFoundError(f"错误: 图像目录 '{IMAGE_DIR}' 不存在。")

    # 获取所有图片文件并排序
    supported_formats = ('.png', '.jpg', '.jpeg', '.bmp')
    try:
        image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(supported_formats)])
    except FileNotFoundError:
        raise FileNotFoundError(f"无法访问目录 '{IMAGE_DIR}'。请检查路径和权限。")

    # 检查图片数量是否足够
    if len(image_files) < NUM_FRAMES:
        raise ValueError(f"目录 '{IMAGE_DIR}' 中只有 {len(image_files)} 张图片，但需要 {NUM_FRAMES} 张。")

    # 读取前 NUM_FRAMES 张图片
    image_list = []
    for i in range(NUM_FRAMES):
        img_path = os.path.join(IMAGE_DIR, image_files[i])
        # cv2.imread 默认以 BGR 格式读取, 这正好符合 DroidNet 的期望
        img = cv2.imread(img_path)
        if img is None:
            raise IOError(f"无法读取图片: {img_path}")
            
        # 缩放到模型所需尺寸
        img_resized = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        image_list.append(img_resized)

    # 将图片列表堆叠成一个Numpy数组
    # 形状变为 [Num_Frames, Height, Width, Channels]
    images_np = np.stack(image_list, axis=0)

    # 添加Batch维度并转换为PyTorch张量
    # 最终形状为 [1, Num_Frames, Height, Width, 3], dtype=uint8
    dummy_images = torch.from_numpy(images_np).unsqueeze(0).to(device)
    
    print(f"成功加载 {dummy_images.shape[1]} 帧图像。最终张量形状: {dummy_images.shape}")

    # --- 后续的虚拟数据生成 (内参、位姿等) 保持不变 ---
    # 相机内参: [Batch, Num_Frames, 4] -> (fx, fy, cx, cy)
    dummy_intrinsics = torch.tensor([
        [320.0, 320.0, IMG_WIDTH / 2, IMG_HEIGHT / 2]
    ]).unsqueeze(0).repeat(1, NUM_FRAMES, 1).to(device)

    # 深度/视差图: [Batch, Num_Frames, Height/8, Width/8]
    dummy_disps = torch.ones(1, NUM_FRAMES, IMG_HEIGHT // 8, IMG_WIDTH // 8).to(device)

    # 位姿 (Poses): [Batch, Num_Frames, 7] -> (tx, ty, tz, qx, qy, qz, qw)
    identity_pose = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    dummy_Gs = identity_pose.unsqueeze(0).unsqueeze(0).repeat(1, NUM_FRAMES, 1).to(device)
    
    dummy_Gs = lietorch.SE3(dummy_Gs)
    
    # 图结构 (Graph): 定义帧之间的连接关系
    graph = OrderedDict()
    for i in range(NUM_FRAMES):
        graph[i] = [j for j in range(NUM_FRAMES) if i != j and abs(i - j) <= 2]

    # --- 3. 初始化并加载模型 ---
    print("正在初始化 DroidNet...")
    droid_model = DroidNet().to(device)

    # 加载预训练权重
    if os.path.exists(WEIGHTS_PATH):
        print(f"正在从 '{WEIGHTS_PATH}' 加载权重...")
        try:
            # 复制您代码中处理权重字典的逻辑
            state_dict = torch.load(WEIGHTS_PATH)
            droid_dict = OrderedDict([(k.replace("module.", ""), v) for (k, v) in state_dict.items()])
            
            # 兼容性处理 (根据您的代码)
            droid_dict["update.weight.2.weight"] = droid_dict["update.weight.2.weight"][:2]
            droid_dict["update.weight.2.bias"] = droid_dict["update.weight.2.bias"][:2]
            droid_dict["update.delta.2.weight"] = droid_dict["update.delta.2.weight"][:2]
            droid_dict["update.delta.2.bias"] = droid_dict["update.delta.2.bias"][:2]

            droid_model.load_state_dict(droid_dict)
            print("权重加载成功！")
        except Exception as e:
            print(f"权重加载失败: {e}")
            print("模型将使用随机初始化的权重运行。")
    else:
        print(f"警告: 权重文件 '{WEIGHTS_PATH}' 不存在。")
        print("模型将使用随机初始化的权重运行。")

    # 设置为评估模式
    droid_model.eval()

    # --- 4. 执行模型的前向传播 ---
    print(f"\n开始执行模型前向传播 (num_steps = {NUM_STEPS})...")
    with torch.no_grad(): # 在评估模式下，关闭梯度计算
        # DroidNet的输入图像格式是 [B, N, H, W, C] 且为 BGR uint8
        # 我们需要将其转换为 [B*N, C, H, W] 并进行归一化，这部分在extract_features中完成
        # 因此，直接传入 [B, N, H, W, C] 格式的张量即可
        images_for_model = dummy_images.permute(0, 1, 4, 2, 3).float() # B, N, C, H, W
        
        # DroidNet内部的extract_features需要 uint8 BGR, HWC格式
        # 但它的forward函数接收的是一个已经处理好的images张量，我们直接构造一个符合内部需求的
        # [N, C, H, W]
        images_for_forward = images_for_model#.squeeze(0) # N, C, H, W

        Gs_list, disp_list, residual_list, flows = droid_model(
            Gs=dummy_Gs, 
            images=images_for_forward, # 传入 [N, C, H, W]
            disps=dummy_disps, 
            intrinsics=dummy_intrinsics, 
            graph=graph, 
            num_steps=NUM_STEPS
        )
        
    visualize_flow_grid(flows[-1].squeeze(0),graph, "/home/honsen/gitDCGU/flow_visualization.png")
    
    
    def visualize_disparity_map(
    disparity_tensor, 
    save_path,
    cmap='plasma' # 'plasma' 或 'inferno' 对于视差图效果很好
    ):
        import matplotlib.pyplot as plt
        """
        将一个逆深度/视差张量可视化为带颜色条的热力图，并保存。

        参数:
        disparity_tensor (torch.Tensor): 输入的单帧视差图张量，形状为 [H, W]。
        save_path (str): 可视化结果图像的保存路径。
        cmap (str): 用于绘制热力图的颜色图。
        """
        if not isinstance(disparity_tensor, torch.Tensor):
            raise TypeError("输入必须是一个PyTorch张量。")

        disp_np = disparity_tensor.cpu().detach().squeeze().numpy()
        if disp_np.ndim != 2:
            raise ValueError(f"输入张量在压缩后应为2维，但得到的是 {disp_np.ndim} 维。")

        # 自动确定颜色范围
        vmin = np.percentile(disp_np, 5)  # 忽略一些最小值噪点
        vmax = np.percentile(disp_np, 95) # 忽略一些最大值噪点

        fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
        
        im = ax.imshow(disp_np, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.axis('off')
        
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Disparity / Inverse Depth', weight='bold', fontsize=12)
        
        ax.set_title('Inverse Depth Visualization', fontsize=16, weight='bold')
        
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        print(f"逆深度图已保存至: {save_path}")
        # ==================== 可视化逆深度的核心逻辑 ====================
    # --- 4. 可视化最终优化好的逆深度图 ---
    if len(disp_list) > 0:
        print(f"\n正在可视化最后一步优化好的逆深度图 (共 {NUM_FRAMES} 帧)...")
        
        # 从返回的列表中，取出最后一次迭代的结果
        # 这个张量的形状是 [Batch, Num_Frames, Height, Width]
        final_disps_tensor = disp_list[-1]
        
        # 移除批次维度
        final_disps_tensor_no_batch = final_disps_tensor.squeeze(0) # -> [Num_Frames, H, W]
        
        # 遍历每一帧的逆深度图并进行可视化
        for i in range(NUM_FRAMES):
            # 提取第 i 帧的逆深度图
            disp_map_single_frame = final_disps_tensor_no_batch[i] # -> [H, W]
            
            # 定义保存路径，为每一帧生成一个带编号的文件
            save_path = os.path.join("/home/honsen/gitDCGU/disps_dir", f"disparity_frame_{i:04d}.png")
            
            # 调用可视化函数
            visualize_disparity_map(
                disparity_tensor=disp_map_single_frame,
                save_path=save_path
            )
            
    print("\n所有逆深度图可视化完成。")
    # ==============================================================
