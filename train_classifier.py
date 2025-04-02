import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# تحميل البيانات
data_dict = pickle.load(open('./data.pickle', 'rb'))

# تحويل البيانات إلى مصفوفات Numpy
data_list = [np.asarray(item) for item in data_dict['data']]

# تحديد الأبعاد القصوى
max_length = max(len(item) for item in data_list)

# إعادة تشكيل البيانات لتكون جميع الأبعاد متساوية
padded_data = [np.pad(item, (0, max_length - len(item)), mode='constant') for item in data_list]

# تحويل البيانات المعدلة إلى مصفوفة Numpy
data = np.asarray(padded_data)

labels = np.asarray(data_dict['labels'])

# تقسيم البيانات إلى مجموعات تدريب واختبار
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# إنشاء وتدريب النموذج
model = RandomForestClassifier()
model.fit(x_train, y_train)

# التنبؤ بالنتائج
y_predict = model.predict(x_test)

# حساب دقة النموذج
score = accuracy_score(y_predict, y_test)
print('{}% of samples were classified correctly !'.format(score * 100))
# حفظ النموذج
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
f.close()
