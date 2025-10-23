"""
Hard Negative Caption生成模块
用于UCF-Crime数据集的Hard Negative对比学习

策略：跨类别语义混淆 + 细粒度描述差异
"""

import random
from typing import List, Dict

# UCF-Crime 13类异常行为的Hard Negative模板
HARD_NEGATIVE_TEMPLATES = {
    "Abuse": {
        "positive": [
            "A person physically attacking another person",
            "A person fighting with another person",
            "A person assaulting someone violently",
        ],
        "hard_negatives": [
            # 来自其他类别的混淆描述
            "A person stealing items from a store",           # Shoplifting
            "A person running away from the scene quickly",   # Burglary
            "A person damaging property with tools",          # Vandalism
            "A police officer arresting a suspect",            # Arrest
            "A person threatening someone with a weapon",      # Assault
            "A person breaking into a building",               # Burglary
            "A person throwing objects at property",           # Vandalism
            "A person driving a car recklessly",               # RoadAccidents
            "A person shooting at a target",                   # Shooting
            "A person forcibly taking items from someone",     # Robbery
        ]
    },
    "Arrest": {
        "positive": [
            "A police officer arresting a suspect",
            "A person being handcuffed by police",
            "A law enforcement officer detaining someone",
        ],
        "hard_negatives": [
            "A person fighting with another person",           # Abuse
            "A person escaping from a building",               # Burglary
            "A person stealing from a store",                  # Shoplifting
            "A person being questioned peacefully",            # Normal
            "A person shaking hands with an officer",          # Normal
            "A person running away from police",               # Burglary
            "A person committing vandalism",                   # Vandalism
            "A person assaulting someone",                     # Assault
            "A person shooting at a target",                   # Shooting
            "A person forcibly taking property",               # Robbery
        ]
    },
    "Arson": {
        "positive": [
            "A person setting fire to property",
            "A person deliberately starting a fire",
            "A person igniting flammable materials",
        ],
        "hard_negatives": [
            "A person using a lighter normally",               # Normal
            "A person cooking with fire",                      # Normal
            "A person damaging property with tools",           # Vandalism
            "A person spraying graffiti",                      # Vandalism
            "A person smoking a cigarette",                    # Normal
            "A person holding a torch",                        # Normal
            "A person breaking windows",                       # Vandalism
            "A person throwing objects",                       # Abuse
            "A person stealing items",                         # Shoplifting
            "A car emitting smoke normally",                   # Normal
        ]
    },
    "Assault": {
        "positive": [
            "A person attacking someone with a weapon",
            "A person threatening someone violently",
            "A person striking another person with force",
        ],
        "hard_negatives": [
            "A person fighting without weapons",               # Abuse
            "A person shooting at a target",                   # Shooting
            "A person forcibly taking items",                  # Robbery
            "A person arresting someone",                      # Arrest
            "A person running away quickly",                   # Burglary
            "A person arguing loudly",                         # Normal
            "A person demonstrating combat moves",             # Normal
            "A person damaging property",                      # Vandalism
            "A person stealing from someone",                  # Shoplifting
            "A person breaking into a building",               # Burglary
        ]
    },
    "Burglary": {
        "positive": [
            "A person breaking into a building",
            "A person forcing entry through a window",
            "A person stealing items from a house",
        ],
        "hard_negatives": [
            "A person opening a door normally",                # Normal
            "A person looking through a window",               # Normal
            "A person stealing from a store",                  # Shoplifting
            "A person running away from scene",                # Escape
            "A person damaging a door",                        # Vandalism
            "A person forcibly taking items",                  # Robbery
            "A police officer entering a building",            # Arrest
            "A person carrying items normally",                # Normal
            "A person fixing a window",                        # Normal
            "A person setting fire to property",               # Arson
        ]
    },
    "Explosion": {
        "positive": [
            "A sudden explosion with smoke and debris",
            "A violent blast destroying objects",
            "A detonation causing damage",
        ],
        "hard_negatives": [
            "A person setting fire to objects",                # Arson
            "A person shooting at a target",                   # Shooting
            "A car accident with smoke",                       # RoadAccidents
            "A person damaging property",                      # Vandalism
            "A fireworks display",                             # Normal
            "A person throwing objects",                       # Abuse
            "A controlled demolition",                         # Normal
            "A car engine backfiring",                         # Normal
            "A person breaking glass",                         # Vandalism
            "A person using power tools",                      # Normal
        ]
    },
    "Fighting": {
        "positive": [
            "Multiple people fighting physically",
            "A group engaged in violent combat",
            "People punching and kicking each other",
        ],
        "hard_negatives": [
            "A person attacking another person",               # Abuse
            "A person assaulting with a weapon",               # Assault
            "A person being arrested forcefully",              # Arrest
            "People playing sports roughly",                   # Normal
            "People dancing energetically",                    # Normal
            "A person damaging property",                      # Vandalism
            "A person forcibly taking items",                  # Robbery
            "People arguing loudly",                           # Normal
            "A person demonstrating martial arts",             # Normal
            "A person stealing from someone",                  # Shoplifting
        ]
    },
    "RoadAccidents": {
        "positive": [
            "A car crashing into another vehicle",
            "A vehicle collision on the road",
            "A car accident with visible damage",
        ],
        "hard_negatives": [
            "A person driving recklessly",                     # Normal/Stealing
            "A car being stolen",                              # Stealing
            "A car explosion",                                 # Explosion
            "A car parked normally",                           # Normal
            "A car driving fast on highway",                   # Normal
            "A car being vandalized",                          # Vandalism
            "A person breaking car window",                    # Burglary
            "A car being set on fire",                         # Arson
            "A police car chasing someone",                    # Arrest
            "A car with flat tire",                            # Normal
        ]
    },
    "Robbery": {
        "positive": [
            "A person forcibly taking items from someone",
            "A person threatening and stealing property",
            "A violent theft with intimidation",
        ],
        "hard_negatives": [
            "A person stealing from a store secretly",         # Shoplifting
            "A person breaking into a building",               # Burglary
            "A person assaulting someone",                     # Assault
            "A person being arrested",                         # Arrest
            "A person handing items to another",               # Normal
            "A person shopping normally",                      # Normal
            "A person running away quickly",                   # Burglary
            "A person fighting physically",                    # Fighting
            "A person receiving a package",                    # Normal
            "A person damaging property",                      # Vandalism
        ]
    },
    "Shooting": {
        "positive": [
            "A person shooting a gun at someone",
            "A person firing a weapon",
            "A person aiming and shooting",
        ],
        "hard_negatives": [
            "A person assaulting with a weapon",               # Assault
            "A person holding a gun normally",                 # Normal
            "A police officer drawing a weapon",               # Arrest
            "A person at a shooting range",                    # Normal
            "A person threatening with a weapon",              # Assault
            "An explosion occurring",                          # Explosion
            "A person forcibly taking items",                  # Robbery
            "A person hunting animals",                        # Normal
            "A person carrying a gun legally",                 # Normal
            "A person cleaning a firearm",                     # Normal
        ]
    },
    "Shoplifting": {
        "positive": [
            "A person secretly stealing items from a store",
            "A person hiding merchandise in clothing",
            "A person taking items without paying",
        ],
        "hard_negatives": [
            "A person shopping normally in a store",           # Normal
            "A person examining products closely",             # Normal
            "A person breaking into a building",               # Burglary
            "A person forcibly taking items",                  # Robbery
            "A person being arrested in a store",              # Arrest
            "A person trying on clothes",                      # Normal
            "A person carrying shopping bags",                 # Normal
            "A person running out of a store",                 # Escape
            "A person damaging store property",                # Vandalism
            "A person arguing with staff",                     # Normal
        ]
    },
    "Stealing": {
        "positive": [
            "A person taking items that don't belong to them",
            "A person stealing property secretly",
            "A person taking objects without permission",
        ],
        "hard_negatives": [
            "A person shopping in a store",                    # Normal
            "A person forcibly taking items",                  # Robbery
            "A person breaking into a place",                  # Burglary
            "A person hiding items secretly",                  # Shoplifting
            "A person moving furniture",                       # Normal
            "A person borrowing items",                        # Normal
            "A person being arrested",                         # Arrest
            "A person organizing belongings",                  # Normal
            "A person delivering packages",                    # Normal
            "A person damaging property",                      # Vandalism
        ]
    },
    "Vandalism": {
        "positive": [
            "A person deliberately damaging property",
            "A person breaking windows or doors",
            "A person spray painting graffiti illegally",
        ],
        "hard_negatives": [
            "A person breaking into a building",               # Burglary
            "A person setting fire to property",               # Arson
            "A person fixing broken items",                    # Normal
            "A person cleaning walls",                         # Normal
            "A person doing construction work",                # Normal
            "A person throwing objects at people",             # Abuse
            "A person creating street art legally",            # Normal
            "A person demolishing with permission",            # Normal
            "A person forcibly entering",                      # Burglary
            "A person being arrested for damage",              # Arrest
        ]
    },
    # 正常行为（用于对比）
    "Normal": {
        "positive": [
            "A person walking normally on the street",
            "A person shopping in a store peacefully",
            "A person driving a car normally",
        ],
        "hard_negatives": [
            "A person running away quickly",                   # Burglary/Escape
            "A person fighting with another",                  # Fighting
            "A person stealing items secretly",                # Shoplifting
            "A person breaking into a building",               # Burglary
            "A person damaging property",                      # Vandalism
            "A person being arrested",                         # Arrest
            "A person setting fire to objects",                # Arson
            "A person shooting a weapon",                      # Shooting
            "A person forcibly taking items",                  # Robbery
            "A person assaulting someone",                     # Assault
        ]
    },
}


def generate_hard_negatives(
    category: str, 
    original_caption: str = None,
    num_negatives: int = 10,
    include_positive: bool = True
) -> List[str]:
    """
    为给定类别生成Hard Negative captions
    
    Args:
        category: UCF-Crime类别名称
        original_caption: 原始正样本caption（如果为None，则从模板中随机选一个）
        num_negatives: 需要的Hard Negative数量
        include_positive: 是否在返回结果中包含正样本（用于构造11候选的场景）
    
    Returns:
        List[str]: Hard Negative captions列表
        - 如果include_positive=True: [正样本, neg1, neg2, ..., neg10] (共11个)
        - 如果include_positive=False: [neg1, neg2, ..., neg10] (共10个)
    """
    # 获取模板
    if category not in HARD_NEGATIVE_TEMPLATES:
        # 如果类别不存在，使用Normal类别的hard negatives
        category = "Normal"
    
    template = HARD_NEGATIVE_TEMPLATES[category]
    
    # 选择正样本caption
    if original_caption is None:
        positive_caption = random.choice(template["positive"])
    else:
        positive_caption = original_caption
    
    # 采样Hard Negatives（随机打乱后取前num_negatives个）
    available_negatives = template["hard_negatives"].copy()
    random.shuffle(available_negatives)
    selected_negatives = available_negatives[:num_negatives]
    
    # 构造返回结果
    if include_positive:
        # 正样本在第0个位置（labels=0）
        return [positive_caption] + selected_negatives
    else:
        return selected_negatives


def batch_generate_hard_negatives(
    categories: List[str],
    original_captions: List[str] = None,
    num_negatives: int = 10
) -> List[List[str]]:
    """
    批量生成Hard Negatives（用于batch处理）
    
    Args:
        categories: 类别列表 ['Abuse', 'Arrest', ...]
        original_captions: 对应的原始captions（可选）
        num_negatives: 每个样本的Hard Negative数量
    
    Returns:
        List[List[str]]: 每个样本的11个候选caption
    """
    results = []
    for i, category in enumerate(categories):
        original = original_captions[i] if original_captions else None
        candidates = generate_hard_negatives(
            category=category,
            original_caption=original,
            num_negatives=num_negatives,
            include_positive=True
        )
        results.append(candidates)
    return results


# ========== 测试代码 ==========
if __name__ == "__main__":
    # 测试单个类别
    print("=" * 80)
    print("测试: Abuse类别的Hard Negatives")
    print("=" * 80)
    candidates = generate_hard_negatives("Abuse", num_negatives=10, include_positive=True)
    for i, caption in enumerate(candidates):
        label = "✅ 正样本" if i == 0 else f"❌ Hard Neg {i}"
        print(f"{label}: {caption}")
    
    print("\n" + "=" * 80)
    print("测试: 批量生成")
    print("=" * 80)
    batch_categories = ["Abuse", "Arrest", "Shooting"]
    batch_results = batch_generate_hard_negatives(batch_categories)
    for cat, cands in zip(batch_categories, batch_results):
        print(f"\n【{cat}】")
        print(f"  正样本: {cands[0]}")
        print(f"  Hard Negs: {cands[1:3]}... (共{len(cands)-1}个)")
