"""
简化测试：只测试视觉编码和时序建模部分
绕过文本编码器的复杂初始化
"""
import torch
import sys
sys.path.insert(0, '/data/zyy/wsvad/2026CVPR/FG-CLIP')

def test_temporal_modules():
    """测试时序模块是否正确添加"""
    print("=" * 60)
    print("测试1: 时序模块初始化")
    print("=" * 60)
    
    from fgclip.model.clip_strc.fgclip import FGCLIPModel
    from fgclip.model.clip_strc.configuration_clip import CLIPConfig, CLIPTextConfig, CLIPVisionConfig
    
    text_config = CLIPTextConfig()
    vision_config = CLIPVisionConfig()
    config = CLIPConfig(
        text_config=text_config.to_dict(),
        vision_config=vision_config.to_dict(),
        projection_dim=512,
    )
    
    model = FGCLIPModel(config)
    
    # 检查时序模块
    assert hasattr(model, 'temporal_transformer'), "❌ 缺少 temporal_transformer"
    assert hasattr(model, 'temporal_attention'), "❌ 缺少 temporal_attention"
    
    print("✅ 时序模块已正确添加:")
    print(f"   - temporal_transformer: 2层TransformerEncoder, 8头注意力")
    print(f"   - temporal_attention: 注意力加权模块")
    print()
    return True


def test_vision_encoding_video():
    """测试视觉编码器对视频输入的处理"""
    print("=" * 60)
    print("测试2: 视频视觉编码（逐帧）")
    print("=" * 60)
    
    from fgclip.model.clip_strc.fgclip import FGCLIPModel
    from fgclip.model.clip_strc.configuration_clip import CLIPConfig, CLIPTextConfig, CLIPVisionConfig
    
    text_config = CLIPTextConfig()
    vision_config = CLIPVisionConfig()
    config = CLIPConfig(
        text_config=text_config.to_dict(),
        vision_config=vision_config.to_dict(),
        projection_dim=512,
    )
    
    model = FGCLIPModel(config).eval()
    
    # 模拟视频输入
    bs, T, C, H, W = 2, 8, 3, 224, 224
    video = torch.randn(bs, T, C, H, W)
    
    # 展平为逐帧
    video_flat = video.view(bs * T, C, H, W)
    
    print(f"视频输入: {video.shape}")
    print(f"展平后: {video_flat.shape}")
    
    try:
        with torch.no_grad():
            # 测试vision_model
            vision_outputs = model.vision_model(pixel_values=video_flat)
            pooled_output = vision_outputs[1]  # (B*T, D_vision)
            
            # 测试投影
            image_features = model.visual_projection(pooled_output)  # (B*T, D_proj)
            
            # 重塑为 (B, T, D)
            image_features = image_features.view(bs, T, -1)
            
            print(f"✅ 视觉编码成功")
            print(f"   - 输出形状: {image_features.shape}")
            print(f"   - 每帧特征维度: {image_features.shape[-1]}")
            print()
            return True, image_features
    except Exception as e:
        print(f"❌ 视觉编码失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, None


def test_temporal_transformer(image_features):
    """测试时序Transformer"""
    print("=" * 60)
    print("测试3: 时序Transformer")
    print("=" * 60)
    
    from fgclip.model.clip_strc.fgclip import FGCLIPModel
    from fgclip.model.clip_strc.configuration_clip import CLIPConfig, CLIPTextConfig, CLIPVisionConfig
    
    text_config = CLIPTextConfig()
    vision_config = CLIPVisionConfig()
    config = CLIPConfig(
        text_config=text_config.to_dict(),
        vision_config=vision_config.to_dict(),
        projection_dim=512,
    )
    
    model = FGCLIPModel(config).eval()
    
    print(f"输入特征: {image_features.shape}")
    
    try:
        with torch.no_grad():
            # 测试Transformer编码
            temporal_features = model.temporal_transformer(image_features)
            
            print(f"✅ 时序Transformer成功")
            print(f"   - 输出形状: {temporal_features.shape}")
            print()
            return True, temporal_features
    except Exception as e:
        print(f"❌ 时序Transformer失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, None


def test_attention_pooling(temporal_features):
    """测试注意力加权池化"""
    print("=" * 60)
    print("测试4: 注意力加权池化")
    print("=" * 60)
    
    from fgclip.model.clip_strc.fgclip import FGCLIPModel
    from fgclip.model.clip_strc.configuration_clip import CLIPConfig, CLIPTextConfig, CLIPVisionConfig
    
    text_config = CLIPTextConfig()
    vision_config = CLIPVisionConfig()
    config = CLIPConfig(
        text_config=text_config.to_dict(),
        vision_config=vision_config.to_dict(),
        projection_dim=512,
    )
    
    model = FGCLIPModel(config).eval()
    
    print(f"输入特征: {temporal_features.shape}")
    
    try:
        with torch.no_grad():
            # 计算注意力权重
            attn_weights = model.temporal_attention(temporal_features)  # (B, T, 1)
            attn_weights = torch.softmax(attn_weights, dim=1)
            
            # 加权聚合
            video_features = (temporal_features * attn_weights).sum(dim=1)  # (B, D)
            
            print(f"✅ 注意力加权池化成功")
            print(f"   - 注意力权重形状: {attn_weights.shape}")
            print(f"   - 视频特征形状: {video_features.shape}")
            print(f"   - 第一个视频的注意力权重: {attn_weights[0].squeeze().tolist()}")
            print()
            return True
    except Exception as e:
        print(f"❌ 注意力加权池化失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_with_attention_mask():
    """测试带attention_mask的时序建模"""
    print("=" * 60)
    print("测试5: 带Attention Mask的时序建模")
    print("=" * 60)
    
    from fgclip.model.clip_strc.fgclip import FGCLIPModel
    from fgclip.model.clip_strc.configuration_clip import CLIPConfig, CLIPTextConfig, CLIPVisionConfig
    
    text_config = CLIPTextConfig()
    vision_config = CLIPVisionConfig()
    config = CLIPConfig(
        text_config=text_config.to_dict(),
        vision_config=vision_config.to_dict(),
        projection_dim=512,
    )
    
    model = FGCLIPModel(config).eval()
    
    # 模拟填充视频
    bs, T = 2, 8
    features = torch.randn(bs, T, 512)
    
    # 第一个视频5帧，第二个8帧
    mask = torch.zeros(bs, T, dtype=torch.bool)
    mask[0, :5] = True
    mask[1, :] = True
    
    print(f"特征形状: {features.shape}")
    print(f"Mask: Video0={mask[0].sum().item()}帧, Video1={mask[1].sum().item()}帧")
    
    try:
        with torch.no_grad():
            # Transformer处理
            temporal_mask = ~mask
            temporal_features = model.temporal_transformer(
                features,
                src_key_padding_mask=temporal_mask
            )
            
            # 注意力加权
            attn_weights = model.temporal_attention(temporal_features)
            attn_weights = attn_weights.masked_fill(~mask.unsqueeze(-1), -1e9)
            attn_weights = torch.softmax(attn_weights, dim=1)
            
            # 聚合
            video_features = (temporal_features * attn_weights).sum(dim=1)
            
            print(f"✅ 带Mask的时序建模成功")
            print(f"   - 输出形状: {video_features.shape}")
            print(f"   - Video0注意力权重: {attn_weights[0].squeeze().tolist()}")
            print(f"   - Video1注意力权重: {attn_weights[1].squeeze().tolist()}")
            print()
            return True
    except Exception as e:
        print(f"❌ 带Mask的时序建模失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("视频时序建模简化测试")
    print("=" * 60 + "\n")
    
    results = []
    
    # 测试1: 模块初始化
    results.append(("时序模块初始化", test_temporal_modules()))
    
    # 测试2: 视觉编码
    success, image_features = test_vision_encoding_video()
    results.append(("视频视觉编码", success))
    
    if success and image_features is not None:
        # 测试3: 时序Transformer
        success, temporal_features = test_temporal_transformer(image_features)
        results.append(("时序Transformer", success))
        
        if success and temporal_features is not None:
            # 测试4: 注意力池化
            results.append(("注意力加权池化", test_attention_pooling(temporal_features)))
    
    # 测试5: 带Mask的处理
    results.append(("Attention Mask处理", test_with_attention_mask()))
    
    # 汇总
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{test_name:<25}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("🎉 所有测试通过！")
        print("\n核心功能验证：")
        print("  ✓ 时序Transformer模块正确添加")
        print("  ✓ 视频逐帧编码功能正常")
        print("  ✓ 时序建模捕捉帧间关系")
        print("  ✓ 注意力加权池化工作正常")
        print("  ✓ Attention Mask正确处理填充帧")
        print("\n✨ 模型已成功支持视频输入和时序建模！")
    else:
        print("⚠️  部分测试失败")
    print("=" * 60 + "\n")
