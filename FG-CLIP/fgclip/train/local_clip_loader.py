"""
本地CLIP加载器 - 不依赖HuggingFace/OpenAI在线下载

用于离线环境，使用项目自带的CLIP实现和本地权重文件
"""

import os
import torch
from typing import Dict, Any
import sys

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 导入本地CLIP
from fgclip.model.clip import clip


class LocalCLIPWrapper:
    """
    包装本地CLIP，提供与HuggingFace兼容的接口
    
    这样可以在train_fgclip.py中无缝替换：
        from local_clip_loader import LocalCLIPWrapper
        processor = LocalCLIPWrapper.get_image_processor(model_path)
        tokenizer = LocalCLIPWrapper.get_tokenizer()
    """
    
    @staticmethod
    def get_available_models():
        """列出可用的模型"""
        return clip.available_models()
    
    @staticmethod
    def load_model(model_name_or_path: str, device: str = "cuda"):
        """
        加载CLIP模型
        
        Args:
            model_name_or_path: 模型名称或本地路径
                - 如果是 "ViT-B/32", "ViT-B/16" 等：从本地缓存加载
                - 如果是文件路径 (.pt): 直接加载
            device: 设备 ("cuda" 或 "cpu")
        
        Returns:
            model: CLIP模型
            preprocess: 图像预处理函数
        """
        print(f"[LocalCLIP] Loading model: {model_name_or_path}")
        
        # 检查是否是本地文件
        if os.path.isfile(model_name_or_path):
            print(f"[LocalCLIP] Loading from local file: {model_name_or_path}")
            model, preprocess = clip.load(model_name_or_path, device=device, jit=False)
        else:
            # 尝试从本地缓存加载
            # 标准化模型名称
            if model_name_or_path == "openai/clip-vit-base-patch32":
                model_name = "ViT-B/32"
            elif model_name_or_path == "openai/clip-vit-base-patch16":
                model_name = "ViT-B/16"
            elif model_name_or_path == "openai/clip-vit-large-patch14":
                model_name = "ViT-L/14"
            else:
                model_name = model_name_or_path
            
            # 检查本地缓存
            cache_dir = os.path.expanduser("~/.cache/clip")
            print(f"[LocalCLIP] Looking for cached model: {model_name}")
            print(f"[LocalCLIP] Cache directory: {cache_dir}")
            
            try:
                model, preprocess = clip.load(model_name, device=device, jit=False, download_root=cache_dir)
                print(f"[LocalCLIP] ✓ Model loaded successfully")
            except Exception as e:
                print(f"[LocalCLIP] ✗ Failed to load model: {e}")
                print(f"[LocalCLIP] Available models: {clip.available_models()}")
                print(f"[LocalCLIP] Please download the model first or provide a local .pt file")
                raise
        
        return model, preprocess
    
    @staticmethod
    def get_tokenizer():
        """
        获取CLIP tokenizer（兼容HuggingFace接口）
        
        Returns:
            一个类似HuggingFace tokenizer的对象
        """
        class CLIPTokenizerWrapper:
            """包装CLIP tokenizer使其兼容HuggingFace接口"""
            
            def __init__(self):
                # CLIP tokenizer是全局的，已经在clip.py中初始化
                pass
            
            def __call__(self, texts, max_length=77, padding="max_length", 
                        truncation=True, return_tensors="pt"):
                """
                Tokenize文本（兼容HuggingFace接口）
                
                Args:
                    texts: str或list[str]
                    max_length: 最大长度（CLIP默认77）
                    padding: padding策略
                    truncation: 是否截断
                    return_tensors: 返回类型
                
                Returns:
                    一个类似HuggingFace BatchEncoding的对象，支持 .input_ids 和 ["input_ids"] 两种访问方式
                """
                if isinstance(texts, str):
                    texts = [texts]
                
                # 使用CLIP的tokenize函数
                tokens = clip.tokenize(
                    texts, 
                    context_length=max_length, 
                    truncate=truncation
                )
                
                # 创建一个同时支持字典访问和属性访问的对象
                class TokenizerOutput:
                    def __init__(self, input_ids):
                        self.input_ids = input_ids
                    
                    def __getitem__(self, key):
                        if key == "input_ids":
                            return self.input_ids
                        raise KeyError(f"Key {key} not found")
                
                return TokenizerOutput(tokens)
            
            def from_pretrained(self, *args, **kwargs):
                """兼容HuggingFace的from_pretrained（但实际不做任何事）"""
                return self
        
        return CLIPTokenizerWrapper()
    
    @staticmethod
    def get_image_processor(model_name_or_path: str = None):
        """
        获取图像预处理器（兼容HuggingFace接口）
        
        Returns:
            一个类似HuggingFace CLIPImageProcessor的对象
        """
        class CLIPImageProcessorWrapper:
            """包装CLIP preprocess使其兼容HuggingFace接口"""
            
            def __init__(self, image_size=224):
                # CLIP的标准预处理
                from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
                try:
                    from torchvision.transforms import InterpolationMode
                    BICUBIC = InterpolationMode.BICUBIC
                except ImportError:
                    from PIL import Image
                    BICUBIC = Image.BICUBIC
                
                self.transform = Compose([
                    Resize(image_size, interpolation=BICUBIC),
                    CenterCrop(image_size),
                    lambda image: image.convert("RGB"),
                    ToTensor(),
                    Normalize(
                        (0.48145466, 0.4578275, 0.40821073), 
                        (0.26862954, 0.26130258, 0.27577711)
                    ),
                ])
            
            def preprocess(self, image, return_tensors="pt"):
                """
                预处理图像（兼容HuggingFace接口）
                
                Args:
                    image: PIL.Image
                    return_tensors: 返回类型
                
                Returns:
                    {"pixel_values": torch.Tensor}
                """
                tensor = self.transform(image)
                
                if return_tensors == "pt":
                    return {"pixel_values": tensor.unsqueeze(0)}
                else:
                    return {"pixel_values": tensor}
            
            def from_pretrained(self, *args, **kwargs):
                """兼容HuggingFace的from_pretrained（但实际不做任何事）"""
                return self
        
        return CLIPImageProcessorWrapper()


def test_local_clip():
    """测试本地CLIP加载"""
    print("=" * 60)
    print("Testing Local CLIP Loader")
    print("=" * 60)
    
    # 1. 测试tokenizer
    print("\n1. Testing Tokenizer...")
    tokenizer = LocalCLIPWrapper.get_tokenizer()
    test_texts = ["A man walking", "A woman running"]
    tokens = tokenizer(test_texts, max_length=77, truncation=True)
    print(f"   ✓ Tokenized {len(test_texts)} texts")
    print(f"   Token shape: {tokens['input_ids'].shape}")
    
    # 2. 测试image processor
    print("\n2. Testing Image Processor...")
    processor = LocalCLIPWrapper.get_image_processor()
    from PIL import Image
    import numpy as np
    dummy_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    processed = processor.preprocess(dummy_image)
    print(f"   ✓ Processed image")
    print(f"   Processed shape: {processed['pixel_values'].shape}")
    
    # 3. 测试模型加载（如果有缓存的话）
    print("\n3. Testing Model Loading...")
    try:
        model, preprocess = LocalCLIPWrapper.load_model("ViT-B/32", device="cpu")
        print(f"   ✓ Model loaded successfully")
        print(f"   Model type: {type(model)}")
    except Exception as e:
        print(f"   ✗ Model loading failed (expected if no cached model): {e}")
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)


if __name__ == "__main__":
    test_local_clip()
