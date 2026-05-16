# rf_predict.py
import numpy as np
import pickle
import os

# *** สำคัญมาก: ต้อง Import คลาสโครงสร้างของโมเดลเพื่อให้ Pickle รู้จักตอนโหลดไฟล์ ***
from rf_fault_train import ScratchRandomForest, ScratchDecisionTree, Node

def implement_model(model_filename):
    """ฟังก์ชันนี้จำลองการรันโปรแกรมใหม่ที่หน้างาน โดยดึงจากไฟล์ที่เซฟไว้"""
    print("\n" + "="*50)
    print(" 💡 การนำไปใช้งานจริง (Implementation / Inference)")
    print("="*50)
    
    # 1. โหลดโมเดลจากไฟล์ (Load)
    try:
        with open(model_filename, 'rb') as f:
            saved_data = pickle.load(f)
            
        trained_model = saved_data['model']
        classes_list = saved_data['classes_list']
        print(f"✅ โหลดโมเดลจาก {model_filename} สำเร็จ พร้อมทำนาย!")
    except FileNotFoundError:
        print(f"❌ ไม่พบไฟล์ {model_filename} กรุณาตรวจสอบชื่อไฟล์ หรือรันไฟล์ rf_train_model.py ก่อน")
        return
    
    # 2. จำลองรับค่าจากเซ็นเซอร์ (ข้อมูลที่ยังไม่เคยเห็น)
    new_sensor_data = np.array([
        [54.41164142, 58.15041275, 57.75001496, 1436.117106, 1890.205411, 1383.781286, 140.109849, 103.7275322, 149.3443468, 2.722447693, -123.0128432, 115.0552212, 50.12618457, 0.927979302, 77.39788761, 132.7245062], # น่าจะเป็น Normal
        [229.673359, 237.0924804, 235.111027, 484.993043, 570.3918817, 496.4823452, 126.5233165, 129.2982547, 118.6321744, -2.12945874, -118.2112116, 116.5714839, 50.00029552, 0.721051051, 62.83533115, 165.3853649], # กระแสสูง แรงดันตก น่าจะ Fault
        [88.96816855, 72.11005077, 239.4878962, 2328.702518, 2050.114986, 507.2192733, 115.4763808, 140.689751, 134.2365586, -3.373830607, -115.8907282, 123.2253724, 50.17991997, 0.617183095, 41.58201847, 144.1158417], # DLG
        [235.0430978, 47.47565507, 72.41195742, 587.2685507, 2009.229044, 1924.190605, 115.1860477, 149.1472461, 113.7068084, -2.97343078, -120.5784589, 115.5684057, 50.19139779, 0.758125047, 41.0145888, 204.5109198], # DLG
        [228.1621549, 235.9328627, 221.391676, 481.1027802, 502.2024519, 574.2566451, 105.0485406, 120.3349986, 125.644227, 1.663970518, -123.967958, 117.8134568, 49.81797309, 0.656248269, 69.70403216, 108.7465244], # DLG
        [221.2366144, 112.6469266, 232.9077978, 431.9484327, 1345.232267, 465.9607487, 140.9901166, 109.7329702, 127.4157061, 2.829355125, -116.7042181, 121.6656609, 49.9198111, 0.69588137, 98.99452911, 146.8874096] # DLG
    ])
    
    print("\nได้รับข้อมูลเซ็นเซอร์ใหม่:")
    print(new_sensor_data)
    
    # 3. ใช้โมเดลพยากรณ์
    predictions_numeric = trained_model.predict(new_sensor_data)
    predictions_text = [classes_list[idx] for idx in predictions_numeric]
    
    print("\nผลการทำนาย (Prediction Results):")
    for i, pred in enumerate(predictions_text):
        print(f"👉 ข้อมูลชุดที่ {i+1} ถูกประเมินว่าเป็นเหตุการณ์: {pred}")
    print("="*50)

if __name__ == "__main__":
    # ระบุชื่อไฟล์โมเดลที่ต้องการโหลด
    try:
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        file_name = os.path.join(curr_dir, 'model_params.pkl')
    except FileNotFoundError:
        print("⚠️ ไม่พบไฟล์")

    model_file_name = file_name
    implement_model(model_file_name)
