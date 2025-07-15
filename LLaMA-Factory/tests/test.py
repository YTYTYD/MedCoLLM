import pandas as pd
import json

# 读取 CSV 文件
csv_path = 'All_Data.csv'  # 修改为你的路径
df = pd.read_csv(csv_path)

# 假设最后一列是标签
features = df.columns[1:-1]
label_col = df.columns[-1]

json_data = []

for _, row in df.iterrows():
    label = row[label_col]

    # 跳过标签为空的行（包括 NaN 和空字符串）
    if pd.isna(label) or str(label).strip() == "":

        continue

    input_str = ", ".join([f"{col}: {row[col]}" for col in features])
    
    json_data.append({
        "instruction": "分类任务：请判断此人的类别。",
        "input": input_str,
        "output": str(label)
    })

# 写入 JSON 文件
with open("formatted_dataset.json", "w", encoding="utf-8") as f:
    json.dump(json_data, f, ensure_ascii=False, indent=2)

print("✅ 转换完成，保存为 formatted_dataset.json")
