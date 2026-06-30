"""
配置多分类任务的辅助脚本

用法：在 finetune 脚本中调用此函数，为包含 'multistage' 的任务设置正确的类别数
"""

def setup_multistage_config(config):
    """
    自动检测并设置多分类任务的类别数（包括EHRBert的输入特征）
    
    Args:
        config: 配置字典
    
    Returns:
        修改后的配置字典
    """
    cls_label_names = config.get('cls_label_names', [])
    if not cls_label_names:
        print("⚠️  警告: cls_label_names 为空，跳过多分类配置")
        return config
    
    cls_num_classes = []
    
    for label_name in cls_label_names:
        # 检测是否为 multistage 任务
        if 'multistage' in label_name.lower():
            # COPD multistage: 4个类别 (0,1,2,3对应I-IV级)，-1为无COPD不参与训练
            cls_num_classes.append(4)
            print(f"✓ 检测到输出多分类任务: {label_name} -> 4类 (0-3对应I-IV级)")
        else:
            # 默认二分类
            cls_num_classes.append(2)
    
    config['cls_num_classes'] = cls_num_classes
    
    # 设置 category_num_classes（用于 EHRBert 的 input_ids 映射）
    # 默认所有输入特征都是二值的
    n_category_feats = config.get('n_category_feats', 0)
    category_num_classes = [2] * n_category_feats
    
    # 如果使用EHR信息，需要为输入的诊断特征设置正确的类别数
    if config.get('use_info_EHR', False):
        # 检查 cls_label_names 中的 multistage 特征
        # 这些特征会作为输入特征添加到 category_feats 中
        category_num_classes = category_num_classes + cls_num_classes[:len(cls_num_classes)//2]
    if not config.get('use_info_LAB', True):
        category_num_classes = category_num_classes[n_category_feats:]
    
    config['category_num_classes'] = category_num_classes
    
    print(f"\n📊 任务配置总结:")
    print(f"  输出任务: {len(cls_label_names)} 个")
    print(f"    - 二分类: {cls_num_classes.count(2)} 个")
    print(f"    - 多分类: {len([c for c in cls_num_classes if c > 2])} 个")
    print(f"  输入特征: {n_category_feats} 个分类特征")
    multi_class_inputs = [i for i, c in enumerate(category_num_classes) if c > 2]
    if multi_class_inputs:
        print(f"    - 多分类输入特征位置: {multi_class_inputs}")
    
    return config

