import joblib

# 加载我们之前保存的标签编码器
label_encoder_path = "outputs/models/source_xgb_baseline_label_encoder.pkl"
le = joblib.load(label_encoder_path)

# 打印出编码器记录的所有类别
# .classes_ 这个属性会按照 数字0, 1, 2, 3... 的顺序，返回对应的文本标签
print("标签编码对应关系 (索引号 -> 文本标签):")
for index, class_name in enumerate(le.classes_):
    print(f"数字 {index} 对应 -> {class_name}")