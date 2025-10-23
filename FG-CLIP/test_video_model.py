"""
测试视频模型的时序建模和动态 bbox 功能
"""
import torch
import sys
sys.path.insert(0, '/data/zyy/wsvad/2026CVPR/FG-CLIP')

def create_test_config():
    """创建测试用的配置"""
    from fgclip.model.clip_strc.configuration_clip import CLIPConfig, CLIPTextConfig, CLIPVisionConfig
    
    # 创建文本配置 - 使用默认的77个位置编码
    text_config = CLIPTextConfig(
        vocab_size=49408,
        hidden_size=512,
        intermediate_size=2048,
        num_hidden_layers=12,
        num_attention_heads=8,
        max_position_embeddings=77,  # 标准CLIP长度
        hidden_act="quick_gelu",
        layer_norm_eps=1e-5,
        dropout=0.0,
        attention_dropout=0.0,
        initializer_range=0.02,
        initializer_factor=1.0,
    )
    
    # 创建视觉配置
    vision_config = CLIPVisionConfig(
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        image_size=224,
        patch_size=32,
        hidden_act="quick_gelu",
        layer_norm_eps=1e-5,
        dropout=0.0,
        attention_dropout=0.0,
        initializer_range=0.02,
        initializer_factor=1.0,
    )
    
    # 创建CLIP配置 - 需要传入dict格式
    config = CLIPConfig(
        text_config=text_config.to_dict(),  # 转为dict
        vision_config=vision_config.to_dict(),  # 转为dict
        projection_dim=512,
        logit_scale_init_value=2.6592,
    )
    
    return config


def create_test_model():
    """创建并初始化测试模型"""
    from fgclip.model.clip_strc.fgclip import FGCLIPModel
    
    config = create_test_config()
    model = FGCLIPModel(config)
    
    # 注意：FG-CLIP需要调用resize_postion_embeding和copy_weight
    # 但这需要已经加载预训练权重，在测试环境中我们跳过
    # model.resize_postion_embeding()
    # model.copy_weight()
    
    return model


def test_temporal_modeling():
    """测试时序建模模块"""
    print("=" * 60)
    print("测试1: 时序建模模块初始化")
    print("=" * 60)
    
    from fgclip.model.clip_strc.fgclip import FGCLIPModel
    
    config = create_test_config()
    model = FGCLIPModel(config)
    
    # 检查时序模块是否正确添加
    assert hasattr(model, 'temporal_transformer'), "❌ 缺少 temporal_transformer"
    assert hasattr(model, 'temporal_attention'), "❌ 缺少 temporal_attention"
    
    print("✅ 时序模块已正确添加:")
    print(f"   - temporal_transformer: {type(model.temporal_transformer)}")
    print(f"   - temporal_attention: {type(model.temporal_attention)}")
    print()


def test_video_forward():
    """测试视频输入的前向传播"""
    print("=" * 60)
    print("测试2: 视频输入前向传播")
    print("=" * 60)
    
    from fgclip.model.clip_strc.fgclip import FGCLIPModel
    
    config = create_test_config()
    model = FGCLIPModel(config).eval()
    
    # 模拟视频输入
    batch_size = 2
    num_frames = 8
    channels = 3
    height = 224
    width = 224
    
    video_tensor = torch.randn(batch_size, num_frames, channels, height, width)
    text_short = torch.randint(0, 1000, (batch_size, 77))
    text_long = torch.randint(0, 1000, (batch_size, 77))
    video_attention_mask = torch.ones(batch_size, num_frames, dtype=torch.bool)
    
    print(f"输入视频形状: {video_tensor.shape}")
    print(f"视频注意力掩码形状: {video_attention_mask.shape}")
    
    try:
        with torch.no_grad():
            output = model(
                image=video_tensor,
                text_short=text_short,
                text_long=text_long,
                video_attention_mask=video_attention_mask,
                add_box_loss=False,
                use_hard_neg=False,
                return_loss=True
            )
        
        print("✅ 视频前向传播成功")
        print(f"   - 输出 loss: {output.loss}")
        print()
        return True
    except Exception as e:
        print(f"❌ 视频前向传播失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_image_backward_compatibility():
    """测试图像输入的向后兼容性"""
    print("=" * 60)
    print("测试3: 图像输入向后兼容性")
    print("=" * 60)
    
    from fgclip.model.clip_strc.fgclip import FGCLIPModel
    
    config = create_test_config()
    model = FGCLIPModel(config).eval()
    
    # 模拟图像输入 (B, C, H, W)
    batch_size = 2
    image_tensor = torch.randn(batch_size, 3, 224, 224)
    text_short = torch.randint(0, 1000, (batch_size, 77))
    text_long = torch.randint(0, 1000, (batch_size, 77))
    
    print(f"输入图像形状: {image_tensor.shape}")
    
    try:
        with torch.no_grad():
            output = model(
                image=image_tensor,
                text_short=text_short,
                text_long=text_long,
                add_box_loss=False,
                use_hard_neg=False,
                return_loss=True
            )
        
        print("✅ 图像前向传播成功 (向后兼容)")
        print(f"   - 输出 loss: {output.loss}")
        print()
        return True
    except Exception as e:
        print(f"❌ 图像前向传播失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_video_with_bbox():
    """测试视频 + bbox 的前向传播"""
    print("=" * 60)
    print("测试4: 视频 + BBox Loss")
    print("=" * 60)
    
    from fgclip.model.clip_strc.fgclip import FGCLIPModel
    
    config = create_test_config()
    model = FGCLIPModel(config).eval()
    
    batch_size = 2
    num_frames = 8
    max_anns = 4
    
    video_tensor = torch.randn(batch_size, num_frames, 3, 224, 224)
    text_short = torch.randint(0, 1000, (batch_size, 77))
    text_long = torch.randint(0, 1000, (batch_size, 77))
    video_attention_mask = torch.ones(batch_size, num_frames, dtype=torch.bool)
    
    # BBox 信息: (B, max_anns, 4) - 归一化坐标
    box_infos = torch.rand(batch_size, max_anns, 4)
    box_texts = torch.randint(0, 1000, (batch_size * max_anns, 77))
    box_nums = torch.tensor([2, 1], dtype=torch.long)  # 第一个视频2个box，第二个1个box
    
    print(f"输入视频形状: {video_tensor.shape}")
    print(f"BBox 信息形状: {box_infos.shape}")
    print(f"BBox 数量: {box_nums}")
    
    try:
        with torch.no_grad():
            output = model(
                image=video_tensor,
                text_short=text_short,
                text_long=text_long,
                video_attention_mask=video_attention_mask,
                box_infos=box_infos,
                box_texts=box_texts,
                box_nums=box_nums,
                add_box_loss=True,
                use_hard_neg=False,
                return_loss=True
            )
        
        print("✅ 视频 + BBox 前向传播成功")
        print(f"   - 输出 loss: {output.loss}")
        print()
        return True
    except Exception as e:
        print(f"❌ 视频 + BBox 前向传播失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_attention_mask():
    """测试注意力掩码的正确性"""
    print("=" * 60)
    print("测试5: 注意力掩码处理")
    print("=" * 60)
    
    from fgclip.model.clip_strc.fgclip import FGCLIPModel
    
    config = create_test_config()
    model = FGCLIPModel(config).eval()
    
    batch_size = 2
    num_frames = 8
    
    video_tensor = torch.randn(batch_size, num_frames, 3, 224, 224)
    text_short = torch.randint(0, 1000, (batch_size, 77))
    text_long = torch.randint(0, 1000, (batch_size, 77))
    
    # 模拟填充: 第一个视频有5帧,第二个视频有8帧
    video_attention_mask = torch.zeros(batch_size, num_frames, dtype=torch.bool)
    video_attention_mask[0, :5] = True
    video_attention_mask[1, :] = True
    
    print(f"视频注意力掩码:")
    print(f"  Video 0: {video_attention_mask[0].sum().item()} 有效帧 / {num_frames} 总帧")
    print(f"  Video 1: {video_attention_mask[1].sum().item()} 有效帧 / {num_frames} 总帧")
    
    try:
        with torch.no_grad():
            output = model(
                image=video_tensor,
                text_short=text_short,
                text_long=text_long,
                video_attention_mask=video_attention_mask,
                add_box_loss=False,
                use_hard_neg=False,
                return_loss=True
            )
        
        print("✅ 注意力掩码处理成功")
        print(f"   - 输出 loss: {output.loss}")
        print()
        return True
    except Exception as e:
        print(f"❌ 注意力掩码处理失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("视频模型时序建模功能测试")
    print("=" * 60 + "\n")
    
    results = []
    
    # 运行所有测试
    results.append(("时序模块初始化", test_temporal_modeling()))
    results.append(("视频前向传播", test_video_forward()))
    results.append(("图像向后兼容", test_image_backward_compatibility()))
    results.append(("视频+BBox", test_video_with_bbox()))
    results.append(("注意力掩码", test_attention_mask()))
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{test_name:<20}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("🎉 所有测试通过！模型修改正确。")
    else:
        print("⚠️ 部分测试失败，请检查错误信息。")
    print("=" * 60 + "\n")
