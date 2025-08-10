import torch
from model.clip import clip

# 正确加载CLIP模型
try:
    # 尝试加载JIT模型
    model = torch.jit.load(clip._download(clip._MODELS['ViT-B-16']), map_location="cpu").eval()
    state_dict = model.state_dict()
    print('JIT模型加载成功')
except RuntimeError:
    # 如果失败，尝试加载state_dict
    state_dict = torch.load(clip._download(clip._MODELS['ViT-B-16']), map_location="cpu")

model_path1 = 'test_output_wizMandD/exp_cctv_ir_cctv_rgb/ViT-B-16_60.pth'
state_dict1 = torch.load(model_path1)

# 输出结果到文件
with open('state_dict.txt', 'w') as f:
    f.write('CLIP模型state_dict:')
    f.write(str(state_dict.keys()))
    f.write('训练模型state_dict:')
    f.write(str(state_dict1.keys()))


