# import numpy as np
# import pandas as pd
# import pickle
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.preprocessing import LabelEncoder

# # Đọc dữ liệu CSV
# df = pd.read_csv('data.csv')

# # Kiểm tra xem các cột có tồn tại trong DataFrame không
# print("Columns in the dataset:", df.columns)

# # Mã hóa các cột phân loại (nếu cần thiết)
# label_encoders = {}
# for column in ['PH', 'Turbidity', 'temperature', 'Potability']:
#     le = LabelEncoder()
#     # Kiểm tra nếu cột tồn tại trong dữ liệu
#     if column in df.columns:
#         df[column] = le.fit_transform(df[column])
#         label_encoders[column] = le
#     else:
#         print(f"Column '{column}' not found in the dataset")

# # Tách dữ liệu
# X = df[['PH', 'Turbidity', 'temperature']]  # Đảm bảo không có 'Potability' trong X
# y = df['Potability']  # Cột 'Potability' là biến mục tiêu

# # Tách dữ liệu thành tập huấn luyện và kiểm tra
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=100)

# # Huấn luyện mô hình
# dt_clf = DecisionTreeClassifier(random_state=2)
# dt_clf.fit(X_train, y_train)

# # Kiểm tra độ chính xác của mô hình
# accuracy = dt_clf.score(X_test, y_test)
# print(f"Model accuracy: {accuracy * 100:.2f}%")

# # Lưu mô hình và bộ mã hóa
# pickle.dump(dt_clf, open('model.pkl', 'wb'))
# pickle.dump(label_encoders, open('label_encoders.pkl', 'wb'))

# print("Model and label encoders have been saved.")

import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# Đọc dữ liệu CSV
df = pd.read_csv('data.csv')

# Kiểm tra xem các cột có tồn tại trong DataFrame không
print("Columns in the dataset:", df.columns)

# Mã hóa cột mục tiêu 'Potability'
label_encoders = {}
le = LabelEncoder()
df['Potability'] = le.fit_transform(df['Potability'])
label_encoders['Potability'] = le

# Tách dữ liệu
X = df[['PH', 'Turbidity', 'temperature']]  # Không cần mã hóa các cột này
y = df['Potability']  # Cột 'Potability' là biến mục tiêu

# Tách dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=100)

# Huấn luyện mô hình
dt_clf = DecisionTreeClassifier(random_state=2)
dt_clf.fit(X_train, y_train)

# Kiểm tra độ chính xác của mô hình
accuracy = dt_clf.score(X_test, y_test)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# Lưu mô hình và bộ mã hóa
pickle.dump(dt_clf, open('model.pkl', 'wb'))
pickle.dump(label_encoders, open('label_encoders.pkl', 'wb'))

print("Model and label encoders have been saved.")

# Kiểm tra với một giá trị đầu vào bất kỳ
input_values = {
    'PH': 7.0,  # Ví dụ giá trị PH
    'Turbidity': 10.0,  # Ví dụ giá trị Turbidity
    'temperature': 30.0  # Sửa lại tên cột thành 'temperature'
}

# Chuyển đổi input_values thành DataFrame với tên cột giống như dữ liệu huấn luyện
input_df = pd.DataFrame([input_values], columns=['PH', 'Turbidity', 'temperature']) 

# Dự đoán với mô hình đã huấn luyện
prediction = dt_clf.predict(input_df)

# Hiển thị kết quả dự đoán
predicted_value = le.inverse_transform(prediction)
print(f"Predicted Potability (0 = Not Potable, 1 = Potable): {predicted_value[0]}")

