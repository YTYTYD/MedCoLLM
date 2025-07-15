import pandas as pd
import requests
import json
from tqdm import tqdm
import re
# 本地 API 地址
API_URL = "http://localhost:8000/v1/chat/completions"
HEADERS = {"Content-Type": "application/json", "Authorization": "Bearer dummy"}
INSTRUCTION = (
    "分类任务：请判断此人的类别。"
    "请你只从【1, 2, 3, 4】中选择一个数字作为类别，不要输出其他内容。"
)


# 读取表格
df = pd.read_excel("./2024年数据.xlsx")

# 去除标签为空的行（最后一列）
label_col = df.columns[-1]
df = df.dropna(subset=[label_col])
# 随机抽取 20% 作为测试集
df = df.sample(frac=0.2, random_state=42).reset_index(drop=True)
true_labels = df[label_col].tolist()

# 填充其他列的缺失值为 "nan"
df = df.fillna("nan")


def extract_prediction(content: str) -> int:
    """
    提取模型输出末尾类似 '</think>\n\n2' 的预测标签值。
    """
    match = re.search(r"</think>\s*\n*\s*(\d(?:\.0)?)\s*$", content)
    if match:
        return int(float(match.group(1)))  # 转为整数
    else:
        return -1  # 无法提取时标记错误
# 构造输入 prompt
def build_input(row):
    return ", ".join([f"{col}: {row[col]}" for col in df.columns[:-1]])

# 存放预测值
pred_labels = []

# 遍历行并请求模型
for idx, row in tqdm(df.iterrows(), total=len(df)):
    input_text = build_input(row)
    payload = {
        "model": "/root/autodl-tmp/saves/deepseek1",
        "messages": [
            {"role": "user", "content": f"{INSTRUCTION}\n{input_text}"}
        ],
        "temperature": 0.6,
        "top_p": 0.7
    }
    data=json.dumps(payload)

   
    try:
        response = requests.post(API_URL, headers=HEADERS, data=json.dumps(payload))
        result = response.json()

        # output_text = result["choices"][0]["message"]["content"].strip()
        # pred = int(float(output_text.split()[0]))  # 提取数字标签
        content = result["choices"][0]["message"]["content"]
        # print(content)
        pred = extract_prediction(content)
        # print(pred)
    except Exception as e:
        print(f"[ERROR] 第 {idx} 行失败：{e}")
        pred = -1  # 标记错误

 
    pred_labels.append(pred)


# 写入预测值与准确性
df["预测值"] = pred_labels
df["是否正确"] = df["预测值"] == df[label_col]
accuracy = (df["是否正确"].sum() / len(df)) * 100

print(f"\n✅ 样本数: {len(df)}, 准确率: {accuracy:.2f}%")

# 保存完整预测结果表
df.to_excel("./deepseek_预测结果_完整1.xlsx", index=False)

# 保存真实标签与预测标签的纯净版文件
label_df = pd.DataFrame({
    "真实标签": df[label_col].astype(int),
    "预测标签": df["预测值"].astype(int)
})
label_df.to_csv("./deepseek_真实vs预测1.csv", index=False)

print("📁 文件已保存：")
print("▶ /mnt/data/deepseek_预测结果_完整.xlsx")
print("▶ /mnt/data/deepseek_真实vs预测.csv")
