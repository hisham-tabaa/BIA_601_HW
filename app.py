# app.py
from flask import Flask, render_template, request, flash
import pandas as pd
import numpy as np
import os
from werkzeug.utils import secure_filename
from ga_algorithm import GeneticFeatureSelection
from baseline_methods import run_baselines
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('لم يتم اختيار ملف')
        return render_template('index.html')
    file = request.files['file']
    if file.filename == '':
        flash('اسم الملف فارغ')
        return render_template('index.html')
    if file and file.filename.endswith('.csv'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            flash('خطأ في قراءة الملف CSV')
            return render_template('index.html')

        if df.empty or df.shape[1] < 2:
            flash('الملف يجب أن يحتوي على عمود هدف واحد على الأقل وعمود ميزات واحد')
            return render_template('index.html')

        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        # تحويل الهدف النصي إلى رقمي إن لزم
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)

        # تقسيم البيانات
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # تسوية البيانات
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # تشغيل GA
        ga = GeneticFeatureSelection(X_train, y_train, pop_size=15, n_gen=15)
        best_chrom, _ = ga.evolve()
        n_ga_features = int(np.sum(best_chrom))

        # حساب دقة GA على مجموعة الاختبار من أجل مقارنة عادلة
        if n_ga_features == 0:
            ga_acc = 0.0
        else:
            selected_idx = np.where(best_chrom == 1)[0]
            clf = RandomForestClassifier(n_estimators=30, random_state=42)
            clf.fit(X_train[:, selected_idx], y_train)
            y_pred = clf.predict(X_test[:, selected_idx])
            ga_acc = accuracy_score(y_test, y_pred)

        # تشغيل الطرق التقليدية (نستخدم نفس عدد الميزات المختارة من GA للمقارنة)
        k_compare = max(1, n_ga_features)
        baselines = run_baselines(X_train, y_train, X_test, y_test, k=k_compare)

        # إعداد النتائج
        results = {
            'GA': {'accuracy': round(ga_acc, 4), 'n_features': n_ga_features},
            **baselines
        }

        feature_names = df.columns[:-1].tolist()
        selected_features = [feature_names[i] for i in range(len(best_chrom)) if best_chrom[i] == 1]

        return render_template('results.html',
                               results=results,
                               selected_features=selected_features,
                               filename=filename)
    else:
        flash('يرجى رفع ملف بصيغة CSV فقط')
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
