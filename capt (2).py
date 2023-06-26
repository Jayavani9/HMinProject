import algo

from flask import request,Flask,send_from_directory
app=Flask(__name__, static_url_path='')


@app.route('/predict_test', methods=['GET','POST'])

def predict_test():
    #QA,age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,target
    #'QA', 'age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal'
    if request.method == 'POST':
        #QA=request.form['QA']
        age=request.form['age']
        sex=request.form['sex']
        cp=request.form['cp']
        trestbps = request.form['trestbps']
        chol = request.form['chol']
        fbs=request.form['fbs']
        restecg = request.form['restecg']
        thalach = request.form['thalach']
        exang = request.form['exang']
        oldpeak= request.form['oldpeak']
        slope= request.form['slope']
        ca= request.form['ca']
        thal= request.form['thal']
        #target= request.form['target']
        #print(target)

        result=algo.algo(age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal)
        return result
    elif request.method == 'GET':
        print(request.form['age'])
        return request.form['age']

@app.route('/html')
def send_html(file):
    return send_from_directory('html',file)

if __name__ == "__main__":
    app.run()
