import os
import sys
import yaml
import subprocess
import copy

# WSL 的主目錄路徑（假設你放在 /home/使用者名稱/hsnu/DDAD-main）
# WSL 路徑寫法
BASE_DIR = "/mnt/d/1128"

CONFIG_FILE = os.path.join(BASE_DIR, "config.yaml")
MAIN_SCRIPT = os.path.join(BASE_DIR, "main.py")

# MVTec3D-AD 10 類別清單（官方資料集類別）
# MVTec AD 15 類別清單（官方資料集類別）
# MVTec AD 2 (MVTec2) 官方 8 類別清單
# MVTec 3D-AD 官方 10 類別清單
categories = [
    "01Gorilla",
    "02Unicorn",
    "03Mallard",
    "04Turtle",
    "05Whale",
    "06Bird",
    "07Owl",
    "08Sabertooth",
    "09Swan",
    "10Sheep",
    "11Pig",
    "12Zalika",
    "13Pheonix",
    "14Elephant",
    "15Parrot",
    "16Cat",
    "17Scorpion",
    "18Obesobeso",
    "19Bear",
    "20Puppy",
]


















def main():
    # 讀取原始配置文件
    with open(CONFIG_FILE, 'r') as f:
        config = yaml.safe_load(f)
    
    # 逐一跑每個類別
    for category in categories:
        print("=" * 50)
        print(f"開始處理類別: {category}")
        print("=" * 50)
        
        # 複製並更新 category 欄位
        category_config = copy.deepcopy(config)
        category_config['data']['category'] = category
        
        # 暫存配置
        tmp_config_path = os.path.join("/tmp", f"config_{category}.yaml")
        with open(tmp_config_path, 'w') as f:
            yaml.dump(category_config, f, default_flow_style=False)
        
        # 執行 main.py，並帶上必要參數
        try:
            subprocess.run(
                [
                    sys.executable,
                    MAIN_SCRIPT,
                    "--train", "True",
                    "--domain_adaptation", "True",
                    "--detection", "True",
                    "--config", tmp_config_path
                ],
                check=True
            )
            print(f"類別 {category} 處理完成!")
        except subprocess.CalledProcessError as e:
            print(f"處理類別 {category} 時出錯: {e}")
        
        print()
    
    print("所有類別處理完成!")


if __name__ == "__main__":
    main()
