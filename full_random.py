import numpy as np
import pandas as pd
from collections import Counter
import os
import matplotlib.pyplot as plt
import seaborn as sns
from numba import njit

# ==========================================
# 🛠 ตั้งค่าระบบ (ตั้งค่าฟอนต์ไทย และล็อคผลการสุ่ม)
# ==========================================
# ล็อค Random Seed ให้ผลลัพธ์เหมือนเดิมทุกครั้งที่รัน
np.random.seed(42)

# ตั้งค่าฟอนต์ภาษาไทยสำหรับ Matplotlib (แก้ปัญหาสี่เหลี่ยม)
plt.rcParams['font.family'] = 'Tahoma' 
# หาก Tahoma ไม่ทำงาน สามารถเปลี่ยนเป็น 'Leelawadee UI' หรือ 'Cordia New' ได้

@njit
def fast_entropy(y):
    # Numba ชอบการทำงานกับ Array พื้นฐาน
    counts = np.bincount(y)
    ps = counts / len(y)
    ent = 0.0
    for p in ps:
        if p > 0:
            ent -= p * np.log2(p)
    return ent

@njit
def fast_split(X_column, split_thresh):
    # คืนค่า Index เป็น 2 ก้อน
    left_idxs = np.argwhere(X_column <= split_thresh).flatten()
    right_idxs = np.argwhere(X_column > split_thresh).flatten()
    return left_idxs, right_idxs

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
# ==========================================
# 🌲 ส่วนที่ 1: คลาส Node สำหรับ Decision Tree
# ==========================================
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

# ==========================================
# 🌲 ส่วนที่ 2: อัลกอริทึม Decision Tree (รองรับ Random Features)
# ==========================================
class ScratchDecisionTree:
    def __init__(self, max_depth=5, n_feats=None):
        self.max_depth = max_depth
        self.n_feats = n_feats 
        self.root = None

    def fit(self, X, y):
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1])
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        if depth >= self.max_depth or n_labels == 1 or n_samples == 0:
            return Node(value=self._most_common_label(y))

        best_feat, best_thresh = self._best_split(X, y, n_features)

        if best_feat is None:
            return Node(value=self._most_common_label(y))

        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
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
                gain = self._information_gain(y, X_column, threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold

        return split_idx, split_thresh
    
    def _information_gain(self, y, X_column, threshold):
        return fast_information_gain(y, X_column, threshold)

    def _split(self, X_column, split_thresh):
        return fast_split(X_column, split_thresh)

    def _entropy(self, y):
        return fast_entropy(y)

    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

# ==========================================
# 🌳 ส่วนที่ 3: อัลกอริทึม Random Forest (Ensemble)
# ==========================================
class ScratchRandomForest:
    def __init__(self, n_trees=10, max_depth=5, n_feats=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        X_np = X.to_numpy() if isinstance(X, pd.DataFrame) else X
        y_np = y.to_numpy() if isinstance(y, pd.Series) else y

        for _ in range(self.n_trees):
            tree = ScratchDecisionTree(max_depth=self.max_depth, n_feats=self.n_feats)
            X_samp, y_samp = self._bootstrap_samples(X_np, y_np)
            tree.fit(X_samp, y_samp)
            self.trees.append(tree)

    def _bootstrap_samples(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def predict(self, X):
        X_np = X.to_numpy() if isinstance(X, pd.DataFrame) else X
        tree_preds = np.array([tree.predict(X_np) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        predictions = np.array([self._most_common_label(preds) for preds in tree_preds])
        return predictions

    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]

# ==========================================
# 🧰 ส่วนที่ 4: เครื่องมือจัดการข้อมูล (Data Utilities)
# ==========================================
class DataUtils:
    @staticmethod
    def custom_train_test_split(X, y, test_size=0.2):
        n_samples = len(y)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        split_idx = int(n_samples * (1 - test_size))
        train_idx, test_idx = indices[:split_idx], indices[split_idx:]
        return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

    @staticmethod
    def custom_minmax_scaler(X_train, X_test):
        X_min = X_train.min(axis=0)
        X_max = X_train.max(axis=0)
        X_train_scaled = (X_train - X_min) / (X_max - X_min)
        X_test_scaled = (X_test - X_min) / (X_max - X_min)
        return X_train_scaled, X_test_scaled
    
    @staticmethod
    def custom_stratified_split(X, y, test_size=0.2):
        """แบ่งข้อมูลแบบรักษาสัดส่วนคลาส"""
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
        """เตรียม Index สำหรับ K-Fold Cross Validation"""
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
        """สร้างตาราง Confusion Matrix แบบแมนนวล"""
        cm = np.zeros((len(labels), len(labels)), dtype=int)
        label_to_index = {label: i for i, label in enumerate(labels)}
        for true_val, pred_val in zip(y_true, y_pred):
            if true_val in label_to_index and pred_val in label_to_index:
                cm[label_to_index[true_val], label_to_index[pred_val]] += 1
        return cm

# ==========================================
# 🚀 ส่วนที่ 5: การเรียกใช้งานจริง (Main Execution)
# ==========================================
if __name__ == "__main__":
    # 1. โหลดและจัดการข้อมูล
    try:
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        file_name = os.path.join(curr_dir, 'relay_fault_dataset.csv')
        df = pd.read_csv(file_name)
        print("✅ โหลดไฟล์ CSV สำเร็จ")
    except FileNotFoundError:
        print("⚠️ ไม่พบไฟล์ CSV กำลังสร้างข้อมูลจำลอง (Synthetic Data) เพื่อการทดสอบ...")
        df = pd.DataFrame({
            'I_a': np.random.uniform(10, 100, 500), 'V_a': np.random.uniform(200, 240, 500),
            'I_b': np.random.uniform(10, 100, 500), 'V_b': np.random.uniform(200, 240, 500),
            'I_c': np.random.uniform(10, 100, 500), 'V_c': np.random.uniform(200, 240, 500),
            'Fault_Type': np.random.choice(['Normal', 'SLG', 'DLG', 'LL', 'Three_Phase'], 500)
        })

    df = df.dropna()
    # สับเปลี่ยนข้อมูลเบื้องต้น
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

    X_all = df_shuffled.drop('Fault_Type', axis=1).to_numpy()
    
    # ดึง y ออกมาเป็นตัวหนังสือ
    y_raw = df_shuffled['Fault_Type'].to_numpy()
    
    # ⭐️ แปลงข้อความเป็นตัวเลข (Label Encoding) เพื่อให้ Numba ดีใจ
    # classes_list จะเก็บชื่อ ['DLG', 'LL', 'Normal', 'SLG', 'Three_Phase']
    # y_all จะกลายเป็นตัวเลข [2, 0, 3, 2, 4, 1, ...] ที่ตรงกับชื่อคลาส
    classes_list, y_all = np.unique(y_raw, return_inverse=True) 
    
    n_features_to_sample = max(1, int(np.sqrt(X_all.shape[1])))


    classes_list = np.unique(y_all)
    
    n_features_to_sample = max(1, int(np.sqrt(X_all.shape[1])))

    print("\n" + "="*50)
    print(" 🌲 เริ่มเทรน Random Forest ด้วย Stratified Split (80/20)")
    print("="*50)
    
    # 2. แบ่งข้อมูลแบบรักษาสัดส่วน
    X_train, X_test, y_train, y_test = DataUtils.custom_stratified_split(X_all, y_all, test_size=0.2)
    
    # 3. เทรนและวัดผล
    rf_clf = ScratchRandomForest(n_trees=30, max_depth=6, n_feats=n_features_to_sample)
    rf_clf.fit(X_train, y_train)
    y_pred = rf_clf.predict(X_test)
    
    acc = np.sum(y_test == y_pred) / len(y_test)
    print(f"🎯 ความแม่นยำ (Accuracy): {acc * 100:.2f}%")

    # 4. วาด Confusion Matrix
    # สร้าง List ของตัวเลข 0 ถึง จำนวนคลาส เพื่อไปใส่เป็น labels ในแกน X, Y
    numeric_labels = np.arange(len(classes_list))
    cm = DataUtils.custom_confusion_matrix(y_test, y_pred, labels=numeric_labels)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=classes_list, yticklabels=classes_list)
    plt.title('Confusion Matrix - Random Forest (100% From Scratch)', fontsize=14)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------
    # โบนัส: ทดสอบสอบไล่แบบ 5-Fold Cross Validation
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print(" 🔄 กำลังทำการทดสอบ 5-Fold Cross Validation ...")
    print("="*50)
    
    folds = DataUtils.custom_k_fold(X_all, y_all, k=5)
    accuracies = []

    for i, (train_idx, test_idx) in enumerate(folds):
        X_tr, y_tr = X_all[train_idx], y_all[train_idx]
        X_te, y_te = X_all[test_idx], y_all[test_idx]
        
        # ปรับจำนวนต้นไม้ลงหน่อยเพื่อให้รันเร็วขึ้นสำหรับ CV
        rf_cv = ScratchRandomForest(n_trees=10, max_depth=5, n_feats=n_features_to_sample)
        rf_cv.fit(X_tr, y_tr)
        
        preds_cv = rf_cv.predict(X_te)
        fold_acc = np.sum(y_te == preds_cv) / len(y_te)
        accuracies.append(fold_acc)
        print(f"  รอบที่ {i+1}: ความแม่นยำ = {fold_acc * 100:.2f}%")

    print(f"\n📊 ความแม่นยำเฉลี่ย (Mean Accuracy) 5 รอบ: {np.mean(accuracies) * 100:.2f}%")
    print("="*50)