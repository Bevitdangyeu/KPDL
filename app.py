from flask import Flask, request, render_template
import pickle
import pandas as pd
app = Flask(__name__)

# Giả sử bạn đã tải mô hình và các bộ mã hóa
model = pickle.load(open('model.pkl', 'rb'))
label_encoders = pickle.load(open('label_encoders.pkl', 'rb'))

# Các cột đầu vào cần thiết cho dự đoán
feature_columns = ['ph', 'turbidity', 'temperature']

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/classify', methods=[ 'POST'])
def classify():
    if request.method == 'POST':
        # Lấy giá trị từ form
        ph = request.form.get('ph')  # Lấy giá trị của PH
        turbidity = request.form.get('turbidity')  # Lấy giá trị của Turbidity
        temperature = request.form.get('temperature')  # Lấy giá trị của Temperature

        # Chuyển giá trị từ string sang float
        try:
            ph = float(ph)
            turbidity = float(turbidity)
            temperature = float(temperature)
        except ValueError:
            return render_template('index.html', prediction_text="Invalid input. Please enter valid values.")

        # Tạo danh sách các giá trị đầu vào cho mô hình
        input_data = [ph, turbidity, temperature]
        input_df = pd.DataFrame([input_data], columns=['PH', 'Turbidity', 'temperature'])
        # Dự đoán với mô hình
        prediction = model.predict(input_df)

        # Kết quả dự đoán
        output = "có thể uống được" if prediction[0] == 1 else "không thể uống được"
        
        # Trả kết quả về template
        return render_template('index.html', prediction_text=f"Nước {output}.")

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
