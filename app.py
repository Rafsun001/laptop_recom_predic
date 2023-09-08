from flask import Flask, request, render_template, jsonify
import numpy as np
import sklearn
import pickle
import pandas as pd
import joblib
import xgboost as xgb 

app = Flask(__name__)


popular_df=pickle.load(open('popular_df.pkl','rb'))
df=pickle.load(open('df.pkl','rb'))
pt=pickle.load(open('pt.pkl','rb'))
#pt2=pickle.load(open('pt2.pkl','rb'))
similarity_scores=pickle.load(open('similarity_scores.pkl','rb'))
#similarity_scores_2=pickle.load(open('similarity_scores_2.pkl','rb'))
model = joblib.load('xgboost_model.joblib')

@app.route('/')
def index():
    return render_template('index.html',
                           name=list(popular_df['name'].values),
                           ram=list(popular_df['ram'].values),
                           image=list(popular_df['img_link'].values),
                           price=list(popular_df['price(in Rs.)'].values),
                           #no_ratings=list(popular_df['no_of_ratings'].values),
                           rating=list(popular_df['rating'].values),
                           storage=list(popular_df['storage'].values),
                           os=list(popular_df['os'].values),
                           processor=list(popular_df['processor'].values),
                           display_size=list(popular_df['display(in inch)'].values))
    

@app.route('/recommend')
def recommend_ui():
    return render_template('recommend.html')

@app.route('/recommend_laptop',methods=['POST'])
def recommend():
    user_input=request.form.get('laptop')
    
    index = np.where(pt.index==user_input)[0][0]
    similar_items = sorted(list(enumerate(similarity_scores[index])),key=lambda x:x[1],reverse=True)[1:5]
    
    data = []
    for i in similar_items:
        item = []
        temp_df = df[df['name'] == pt.index[i[0]]]
        item.extend(list(temp_df.drop_duplicates('name')['name'].values))
        item.extend(list(temp_df.drop_duplicates('name')['ram'].values))
        item.extend(list(temp_df.drop_duplicates('name')['img_link'].values))
        item.extend(list(temp_df.drop_duplicates('name')['price(in Rs.)'].values))
        item.extend(list(temp_df.drop_duplicates('name')['no_of_ratings'].values))
        item.extend(list(temp_df.drop_duplicates('name')['rating'].values))
        item.extend(list(temp_df.drop_duplicates('name')['storage'].values))
        item.extend(list(temp_df.drop_duplicates('name')['os'].values))
        item.extend(list(temp_df.drop_duplicates('name')['processor'].values))
        item.extend(list(temp_df.drop_duplicates('name')['display(in inch)'].values))
        
        data.append(item)
    return render_template('recommend.html',data=data)
    


@app.route("/price_predc")
def prediction_ui():
    return render_template("price_predc.html")

@app.route("/predict", methods = ["POST"])
def predict():
    if request.method == "POST":
        #Company Name
        company_name = request.form["company_nm"]
        if(company_name=='hp'):
            hp = 1
            acer = 0
            aus = 0
            dell = 0
            lenevo = 0
            others = 0
        elif (company_name=='acer'):
            hp = 0
            acer = 1
            aus = 0
            dell = 0
            lenevo = 0
            others = 0
        elif (company_name=='aus'):
            hp = 0
            acer = 0
            aus = 1
            dell = 0
            lenevo = 0
            others = 0
        elif (company_name=='dell'):
            hp = 0
            acer = 0
            aus = 0
            dell = 1
            lenevo = 0
            others = 0
        elif (company_name=='lenevo'):
            hp = 0
            acer = 0
            aus = 0
            dell = 0
            lenevo = 1
            others = 0
        elif (company_name=='others'):
            hp = 0
            acer = 0
            aus = 0
            dell = 0
            lenevo = 0
            others = 1
        else:
            pass
        
        #Laptop type
        laptop_type = request.form["company_nm"]
        if(laptop_type=='ultrabookp'):
            lp_type = 0
            
        elif(laptop_type=='notebook'):
            lp_type = 1
            
        else:
            lp_type = 2
            
        
        #Ram
        ram = request.form["ram"]
        
        #Weight
        weight = float(request.form["weight"])
        
        #Memory
        memory = request.form["memory"]
        
        #Memory Type
        memory_type = request.form["memory_typ"]
        if(memory_type=='ssd'):
            m_type = 0
            
        else:
            m_type=1
            
        #Cpu model name
        cpu_mdl_name = request.form["cpumdl"]
        if(cpu_mdl_name=='intel_core_i5'):
            intel_core_i5=1
            intel_core_i7=0
            intel_core_i3=0
            intel_celeron_series_process=0
            amd_processor=0
            intel_other_processor=0
        elif(cpu_mdl_name=='intel_core_i7'):
            intel_core_i5=0
            intel_core_i7=1
            intel_core_i3=0
            intel_celeron_series_process=0
            amd_processor=0
            intel_other_processor=0
        elif(cpu_mdl_name=='intel_core_i3'):
            intel_core_i5=0
            intel_core_i7=0
            intel_core_i3=1
            intel_celeron_series_process=0
            amd_processor=0
            intel_other_processor=0
        elif(cpu_mdl_name=='intel_celeron_series_process'):
            intel_core_i5=0
            intel_core_i7=0
            intel_core_i3=0
            intel_celeron_series_process=1
            amd_processor=0
            intel_other_processor=0
        elif(cpu_mdl_name=='amd_processor'):
            intel_core_i5=0
            intel_core_i7=0
            intel_core_i3=0
            intel_celeron_series_process=0
            amd_processor=1
            intel_other_processor=0
        else:
            intel_core_i5=0
            intel_core_i7=0
            intel_core_i3=0
            intel_celeron_series_process=0
            amd_processor=0
            intel_other_processor=1
            
        #Cpu GHz
        cpu_ghz = float(request.form["cpu_ghz"])
        
        #Screen Type
        screen_type = request.form["scrn_typ"]
        if(screen_type=='ips'):
            scr_type=0
        else:
            scr_type=1
        print('===============Pre+22+++++++++++++')        
        #Touch Display
        touch_display = request.form["touch_dsply"]
        if(touch_display=='yes'):
            touc_dis=1
        else:
            touc_dis=0
            
        #Screen Resolution 
        screen_res = request.form["scrn_res"]
        xx=screen_res.split('x')
        
        x_re=int(xx[0])
        y_rez=int(xx[1])
        
        #Screen Size
        screen_size1 = request.form["scrn_size"]
        screen_size = float(screen_size1)
        
        
        ppi=(((x_re**2) + (y_rez**2))**0.5/screen_size)
        
        #Gpu Brand
        gpu_brand = request.form["gpu_brand"]
        if(gpu_brand=='intel'):
            gpuBrnd=0
            
        elif(gpu_brand=='amd'):
            gpuBrnd=1
        else:
            gpuBrnd=2

        #OS
        os = request.form["os"]
        if(gpu_brand=='window'):   
            os_nm=1
        else:
            os_nm=0
        print('===============Pre++++++++++++++')  
        inputdt=[
            [hp,acer,aus,dell,lenevo,others,lp_type,
            ram,weight,memory,m_type,intel_core_i5,intel_core_i7,
            intel_core_i3,intel_celeron_series_process,amd_processor,
            intel_other_processor,cpu_ghz,scr_type,touc_dis,ppi,gpuBrnd,os_nm
            ]
            ] 
        features_array = np.array(inputdt, dtype=float)
        
        prediction = model.predict(features_array)[0]
        print('===============Pre33++++++++++++++') 
        
        return render_template('price_predc.html', prediction_text="Your Laptop price is Rs. {}".format(prediction))
    
    return render_template("price_predc.html")

        
    
        
if __name__ == '__main__':
    app.run()