
import numpy as np
import pandas as pd
from collections import Counter
from numba import njit
import matplotlib.pyplot as plt
import os
import pickle
import seaborn as sns

# ==========================================
# 🛠 ตั้งค่าระบบพื้นฐาน
# ==========================================
np.random.seed(42)                  # ล็อคค่าการสุ่มให้ผลลัพธ์เหมือนเดิมทุกครั้งที่รัน
plt.rcParams['font.family'] = 'Tahoma'  # ตั้งค่าฟอนต์กราฟให้รองรับภาษาไทย (ถ้ามี)

"""
=============================================================================
โปรแกรม Core Engine สำหรับ Random Forest ที่เขียนขึ้นเองจากศูนย์ 
ประกอบไปด้วย: DataUtils class, Node class, ScratchDecisionTree class, ScratchRandomForest class และฟังก์ชันช่วยเหลืออื่นๆ
โดยที่ method ในแต่ละ class เป็น static method เพื่อให้เรียกใช้งานได้โดยตรงจากชื่อคลาส ไม่ต้องสร้าง instance
หน้าที่: เป็น "ไลบรารีหลัก (Core Engine)" ที่เก็บโครงสร้างสมการคณิตศาสตร์และอัลกอริทึม
        ของ Random Forest ที่เขียนขึ้นเองจากศูนย์ รวมถึงเครื่องมือจัดการข้อมูลต่างๆ
ข้อควรรู้: 
- ไฟล์นี้ไม่ได้มีไว้สำหรับกดรัน (No Execution)
- ไฟล์อื่นๆ (Train / Predict) จะต้องทำการ Import คลาสจากไฟล์นี้ไปใช้งาน
- มีการใช้ไลบรารี Numba (@njit) เพื่อเร่งความเร็วในการคำนวณสมการคณิตศาสตร์
=============================================================================
"""

# ==========================================
# 🧰 Data Utilities (เครื่องมือจัดการข้อมูล)
# ==========================================
class DataUtils:
    @staticmethod
    def custom_stratified_split(X, y, test_size=0.2):
        """
        แบ่งข้อมูล Train/Test โดยรักษาสัดส่วนของแต่ละคลาส (Label) ให้เท่ากัน
        ป้องกันปัญหาคลาสบางประเภทหายไปอยู่ในชุดใดชุดหนึ่งมากเกินไป
        """
        classes = np.unique(y)
        train_indices, test_indices = [], []
        for c in classes:
            c_indices = np.where(y == c)[0]
            np.random.shuffle(c_indices)
            n_test = int(len(c_indices) * test_size)
            test_indices.extend(c_indices[:n_test])
            train_indices.extend(c_indices[n_test:])
        np.random.shuffle(train_indices)
        np.random.shuffle(test_indices)
        return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

    @staticmethod
    def custom_k_fold(X, y, k=5):
        """
        แบ่งข้อมูลเป็น K ส่วน (Folds) สำหรับทำ Cross Validation
        คืนค่าเป็นลิสต์ของ (Train Index, Test Index) ในแต่ละรอบ
        """
        n_samples = len(y)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        fold_sizes = np.full(k, n_samples // k, dtype=int)
        fold_sizes[:n_samples % k] += 1 
        folds = []
        current_idx = 0
        for fold_size in fold_sizes:
            start, stop = current_idx, current_idx + fold_size
            test_idx = indices[start:stop]
            train_idx = np.concatenate([indices[:start], indices[stop:]])
            folds.append((train_idx, test_idx))
            current_idx = stop
        return folds

    @staticmethod
    def custom_confusion_matrix(y_true, y_pred, labels):
        """สร้างตาราง Confusion Matrix เพื่อนับจำนวนการทายถูก/ผิด ในแต่ละคลาส"""
        cm = np.zeros((len(labels), len(labels)), dtype=int)
        label_to_index = {label: i for i, label in enumerate(labels)}
        for true_val, pred_val in zip(y_true, y_pred):
            if true_val in label_to_index and pred_val in label_to_index:
                cm[label_to_index[true_val], label_to_index[pred_val]] += 1
        return cm

# ==========================================
# 🌲 อัลกอริทึม Numba และโครงสร้าง Tree
# ==========================================
# ฟังก์ชันคำนวณ Entropy วัดความไม่บริสุทธิ์ของข้อมูล (เร่งความเร็วด้วย Numba)
@njit
def fast_entropy(y):
    counts = np.bincount(y)
    ps = counts / len(y)
    ent = 0.0
    for p in ps:
        if p > 0:
            ent -= p * np.log2(p)
    return ent

# ฟังก์ชันแยกข้อมูลเป็น 2 กิ่ง ซ้าย-ขวา ตามจุดตัด (Threshold)
@njit
def fast_split(X_column, split_thresh):
    left_idxs = np.argwhere(X_column <= split_thresh).flatten()
    right_idxs = np.argwhere(X_column > split_thresh).flatten()
    return left_idxs, right_idxs

# ฟังก์ชันคำนวณกำไรของข้อมูล (Information Gain) เพื่อหาจุดแบ่งที่ดีที่สุด
@njit
def fast_information_gain(y, X_column, threshold):
    parent_entropy = fast_entropy(y)
    left_idxs, right_idxs = fast_split(X_column, threshold)
    if len(left_idxs) == 0 or len(right_idxs) == 0:
        return 0.0
    n = len(y)
    n_l, n_r = len(left_idxs), len(right_idxs)
    e_l, e_r = fast_entropy(y[left_idxs]), fast_entropy(y[right_idxs])
    child_entropy = (n_l / n) * e_l + (n_r / n) * e_r
    return parent_entropy - child_entropy

class Node:
    """ตัวแทนของจุดเชื่อม (Node) แต่ละจุดในต้นไม้ตัดสินใจ"""
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature       # คอลัมน์ที่ใช้เป็นเงื่อนไข (เช่น V_a)
        self.threshold = threshold   # จุดตัด (เช่น <= 200)
        self.left = left             # กิ่งซ้าย (เงื่อนไขเป็นจริง)
        self.right = right           # กิ่งขวา (เงื่อนไขเป็นเท็จ)
        self.value = value           # ผลลัพธ์สุดท้าย (เฉพาะใบไม้)
        
    def is_leaf_node(self):
        """ตรวจสอบว่าเป็นจุดสิ้นสุดของกิ่ง (ใบไม้) แล้วหรือไม่"""
        return self.value is not None

class ScratchDecisionTree:
    """คลาสสร้างต้นไม้ตัดสินใจ (Decision Tree) 1 ต้น"""
    def __init__(self, max_depth=5, n_feats=None):
        self.max_depth = max_depth
        self.n_feats = n_feats 
        self.root = None

    def fit(self, X, y):
        # จำกัดจำนวนฟีเจอร์ที่ใช้พิจารณาตอนสร้างต้นไม้
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1])
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        
        # กฎการหยุดสร้างต้นไม้: ลึกสุดแล้ว หรือ เหลือคำตอบเดียว หรือ ไม่มีข้อมูลแล้ว
        if depth >= self.max_depth or n_labels == 1 or n_samples == 0:
            return Node(value=self._most_common_label(y))

        # หาจุดแบ่งกิ่งที่ดีที่สุด
        best_feat, best_thresh = self._best_split(X, y, n_features)
        if best_feat is None:
            return Node(value=self._most_common_label(y))

        # แตกกิ่งซ้ายขวา และสร้างต้นไม้ต่อไปเรื่อยๆ แบบ Recursive
        left_idxs, right_idxs = fast_split(X[:, best_feat], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feat, best_thresh, left, right)

    def _best_split(self, X, y, n_features):
        best_gain = -1
        split_idx, split_thresh = None, None
        feat_idxs = np.random.choice(n_features, self.n_feats, replace=False)

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = fast_information_gain(y, X_column, threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold
        return split_idx, split_thresh

    def _most_common_label(self, y):
        return Counter(y).most_common(1)[0][0]

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

class ScratchRandomForest:
    """คลาสรวมต้นไม้หลายๆ ต้นเข้าด้วยกัน (Random Forest)"""
    def __init__(self, n_trees=10, max_depth=5, n_feats=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = ScratchDecisionTree(max_depth=self.max_depth, n_feats=self.n_feats)
            n_samples = X.shape[0]
            # สุ่มข้อมูลแบบแทนที่ (Bootstrap Aggregating / Bagging)
            idxs = np.random.choice(n_samples, n_samples, replace=True)
            tree.fit(X[idxs], y[idxs])
            self.trees.append(tree)

    def predict(self, X):
        # ให้ต้นไม้ทุกต้นโหวตคำตอบ แล้วเลือกคำตอบที่ถูกโหวตมากที่สุด (Majority Vote)
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        return np.array([Counter(preds).most_common(1)[0][0] for preds in tree_preds])


"""
=============================================================================
ส่วนนี้เป็น "สคริปต์หลักสำหรับสร้างโมเดล" (Training Pipeline) ที่จะถูกเรียกใช้เมื่อเราต้องการเทรนโมเดลใหม่
หน้าที่: เป็น "สคริปต์หลักสำหรับสร้างโมเดล" (Training Pipeline)
ประกอบไปด้วย: ModelTrainer class ที่มี static method สำหรับโหลดข้อมูล, ค้นหาพารามิเตอร์ที่ดีที่สุด, และประเมินผลโมเดลตัวสุดท้าย
การทำงาน:
1. โหลดข้อมูลจากไฟล์ CSV (หรือสร้างข้อมูลจำลองถ้าไม่พบไฟล์)
2. ค้นหาพารามิเตอร์ที่ดีที่สุด (Hyperparameter Tuning) ด้วยการทำ Cross Validation
3. สร้างโมเดลตัวสุดท้ายและทดสอบความแม่นยำ
4. บันทึกโมเดล (Export) เป็นไฟล์ .pkl เพื่อส่งต่อให้ระบบอื่นนำไปใช้งาน
เงื่อนไขการรัน: ต้องมีไฟล์ rf_core.py อยู่ในโฟลเดอร์เดียวกัน
=============================================================================
"""

class ModelTrainer:
    @staticmethod
    def load_and_prepare_data(directory=None, feature=None, target=None):
        """ฟังก์ชันสำหรับอ่านไฟล์ ทำความสะอาดข้อมูล และแยกฟีเจอร์กับคำตอบ"""
        try:
            if directory is None:
                curr_dir = os.path.dirname(os.path.abspath(__file__))
                file_name = os.path.join(curr_dir, 'relay_fault_dataset.csv')
                df = pd.read_xlsx(file_name)
                print("✅ โหลดไฟล์ CSV สำเร็จ")
            else:
                file_name = directory
                df = pd.read_xlsx(file_name)
                print("✅ โหลดไฟล์ Excel สำเร็จ")
        except FileNotFoundError:
            # ระบบ Fallback: ถ้าลืมใส่ไฟล์ Excel มันจะสร้างข้อมูลสุ่มขึ้นมาให้รันผ่านไปก่อนได้
            print("⚠️ ไม่พบไฟล์ Excel กำลังสร้างข้อมูลจำลอง (Synthetic Data) เพื่อการทดสอบ...")
            df = pd.DataFrame({
                'I_a': np.random.uniform(10, 100, 500), 'V_a': np.random.uniform(200, 240, 500),
                'I_b': np.random.uniform(10, 100, 500), 'V_b': np.random.uniform(200, 240, 500),
                'I_c': np.random.uniform(10, 100, 500), 'V_c': np.random.uniform(200, 240, 500),
                'Fault_Type': np.random.choice(['Normal', 'SLG', 'DLG', 'LL', 'Three_Phase'], 500)
            })

        df = df.dropna() # ตัดแถวที่มีข้อมูลสูญหาย (NaN) ทิ้ง
        df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

        if target is None:
            target = 'Fault_Type'
        if feature is None:
            feature = [col for col in df.columns if col != target]
            
        selected_features = feature
        
        # ดึงคอลัมน์ข้อมูล (X) และคอลัมน์เป้าหมาย (y)
        X_all = df_shuffled[selected_features].to_numpy()
        y_raw = df_shuffled[target].to_numpy()
        
        # แปลง Label แบบข้อความให้เป็นตัวเลข 0, 1, 2...
        classes_list, y_all = np.unique(y_raw, return_inverse=True) 
        return X_all, y_all, classes_list

    @staticmethod
    def hyperparameter_tuning(X_train, y_train, n_feats, k=5, param_grid=None):
        """
        ฟังก์ชันทดลองเปลี่ยนค่า Setting (Hyperparameters) เพื่อหาคู่ที่ให้ความแม่นยำสูงสุด
        โดยใช้วิธีทดสอบแบบหมุนเวียน (K-Fold Cross Validation)
        """
        print("\n" + "="*50)
        print(" 🔄 กำลังทำการค้นหาพารามิเตอร์ที่ดีที่สุด (Hyperparameter Tuning)")
        print("    ด้วยวิธี 5-Fold Cross Validation บนชุดข้อมูล Training")
        print("="*50)
        
        if param_grid is None:
            param_grid = {
                'n_trees': [5, 10, 15, 20],
                'max_depth': [3, 5, 7]
            }
        
        folds = DataUtils.custom_k_fold(X_train, y_train, k=k)
        best_acc = 0
        best_params = {}
        
        # ลองจับคู่ทุกความเป็นไปได้ระหว่างจำนวนต้นไม้ (n_trees) และความลึก (max_depth)
        for n_t in param_grid['n_trees']:
            for m_d in param_grid['max_depth']:
                fold_accs = []
                for i ,(train_idx, val_idx) in enumerate(folds):
                    X_tr_cv, y_tr_cv = X_train[train_idx], y_train[train_idx]
                    X_val_cv, y_val_cv = X_train[val_idx], y_train[val_idx]
                    
                    rf_cv = ScratchRandomForest(n_trees=n_t, max_depth=m_d, n_feats=n_feats)
                    rf_cv.fit(X_tr_cv, y_tr_cv)
                    preds = rf_cv.predict(X_val_cv)
                    acc = np.mean(preds == y_val_cv)
                    fold_accs.append(acc)
                    print(f"\r   รอบที่ {i+1}, n_trees={n_t}, max_depth={m_d} | CV Accuracy: {acc*100:.2f}%", end="")
                    
                avg_acc = np.mean(fold_accs)
                print(f"-> ทดสอบ n_trees={n_t}, max_depth={m_d} | CV Accuracy: {avg_acc*100:.2f}%")
                
                if avg_acc > best_acc:
                    best_acc = avg_acc
                    best_params = {'n_trees': n_t, 'max_depth': m_d}
                    
        print("-" * 50)
        print(f"🎯 พารามิเตอร์ที่ดีที่สุดคือ: {best_params} (CV Accuracy: {best_acc*100:.2f}%)")
        return best_params

    @staticmethod
    def final_evaluation(X_train, y_train, X_test, y_test, best_params, n_feats, classes_list):
        """นำพารามิเตอร์ที่ดีที่สุด มาเทรนกับข้อมูลทั้งหมด แล้ววัดผลครั้งสุดท้าย พร้อมวาดกราฟ"""
        print("\n" + "="*50)
        print(" 🚀 เริ่มเทรนโมเดลตัวสุดท้าย (Final Model) ด้วยพารามิเตอร์ที่ดีที่สุด")
        print("="*50)
        
        final_rf = ScratchRandomForest(
            n_trees=best_params['n_trees'], 
            max_depth=best_params['max_depth'], 
            n_feats=n_feats
        )
        final_rf.fit(X_train, y_train)
        y_pred = final_rf.predict(X_test)
        
        test_acc = np.mean(y_test == y_pred)
        print(f"✅ ความแม่นยำบนข้อมูล Test Set (Unseen Data): {test_acc * 100:.2f}%")
        
        numeric_labels = np.arange(len(classes_list))
        cm = DataUtils.custom_confusion_matrix(y_test, y_pred, labels=numeric_labels)

    
        
        # วาดกราฟแสดงความแม่นยำ Heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=classes_list, yticklabels=classes_list)
        plt.title(f"Final Confusion Matrix\n(n_trees={best_params['n_trees']}, max_depth={best_params['max_depth']})", fontsize=14)
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('Actual', fontsize=12)
        plt.tight_layout()
        plt.show()
        return final_rf

    @staticmethod
    def save_model_to_file(model, classes_list, filename="trained_model.pkl"):
        """บรรจุโมเดลและชื่อคลาสลงในไฟล์ .pkl เพื่อนำไปใช้ที่อื่นต่อ"""
        print("\n" + "="*50)
        print(f" 💾 กำลังบันทึกโมเดลลงไฟล์: {filename}")
        print("="*50)
 
        export_data = {'model': model, 'classes_list': classes_list}
        
        with open(filename, 'wb') as f:
            pickle.dump(export_data, f)
            
        print("✅ บันทึกโมเดลสำเร็จ! นำไฟล์นี้ไปใช้ในระบบ Production ได้เลย")
    
      
# ==========================================
# 🚦 Main Execution (จุดเริ่มต้นการรันโปรแกรม)
# ==========================================
if __name__ == "__main__":
    # 1. กำหนดที่อยู่ไฟล์
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    file_name = os.path.join(curr_dir, 'relay_fault_dataset.xlsx')

    # 2. กำหนดว่าเราจะใช้คอลัมน์ไหนทำนายอะไร (สำคัญมาก: ลำดับต้องตรงกับตอน Predict)
    SELECTED_FEATURES = ['V_a', 'V_b', 'V_c', 'I_a', 'I_b', 'I_c', 'Z_a', 'Z_b', 'Z_c',
                         'Angle_a', 'Angle_b', 'Angle_c', 'Frequency', 'Power_Factor', 
                         'Load_Level', 'Line_Length']
    TARGET = 'Fault_Type'
    
    # 3. เตรียมข้อมูล
    X_all, y_all, classes_list = ModelTrainer.load_and_prepare_data(directory=file_name, feature=SELECTED_FEATURES, target=TARGET)
    
    # กฎของ Random Forest: คอลัมน์ที่จะใช้สุ่มพิจารณาต่อ 1 จุดตัด = สแควร์รูทของคอลัมน์ทั้งหมด
    n_feats_to_sample = max(1, int(np.sqrt(X_all.shape[1])))
    
    # 4. แบ่งข้อมูล 80% สำหรับสอน (Train) และ 20% สำหรับสอบ (Test)
    X_train, X_test, y_train, y_test = DataUtils.custom_stratified_split(X_all, y_all, test_size=0.2)
    
    # 5. สั่งจูนพารามิเตอร์ (ปรับแต่งตัวเลือกเพื่อหาจุดที่ดีที่สุด)
    
    param_grid = {'n_trees': [5, 10, 15, 20], 'max_depth': [3, 5, 7]}
    # จับคู่จำนวนต้นไม้ (n_trees) กับความลึก (max_depth) แล้วทำ Cross Validation เพื่อหาคู่ที่ให้ความแม่นยำสูงสุด
    best_params = ModelTrainer.hyperparameter_tuning(X_train, y_train, n_feats_to_sample, k=5, param_grid=param_grid)
    
    # 6. ประเมินผลครั้งสุดท้าย และบันทึกไฟล์
    final_model = ModelTrainer.final_evaluation(X_train, y_train, X_test, y_test, best_params, n_feats_to_sample, classes_list)
    model_file_name = os.path.join(curr_dir, 'model_params.pkl')
    ModelTrainer.save_model_to_file(final_model, classes_list, filename=model_file_name)
