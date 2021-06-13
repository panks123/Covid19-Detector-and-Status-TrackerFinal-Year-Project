from flask import Flask,render_template,request,redirect
import pickle
import requests
import socket
from bs4 import BeautifulSoup


def getData(url):
    '''for getting data from url'''
    r=requests.get(url)
    # print(r.text)
    return  r.text

file=open('detectorModel.pkl','rb')
clf=pickle.load(file)
file.close()
app=Flask(__name__)


@app.route('/')
def main():
    return render_template('index.html')

@app.route('/detector',methods=["POST","GET"])
def detector():
    # code for inference
    if request.method=="POST":
        formDict=request.form
        if formDict['name']=='' or formDict['name']=='' or formDict['city']=='' or formDict['fever']=='':
            return render_template('detector.html')
        name=formDict['name']
        city=formDict['city']
        cough=int(formDict['cough'])
        fever=int(formDict['fever'])
        pain=int(formDict['pain'])
        age=int(formDict['age'])
        rNose=int(formDict['runnyNose'])
        diffBreath=int(formDict['diffBreath'])
        travelled=int(formDict['travelled'])

        infection=clf.predict_proba([[fever,pain,age,rNose,diffBreath,travelled,cough]])
        infProb=infection[0][1]
        print(name,city,cough)

        # Adding entries to the database
        
        return render_template('result.html',name=name,inf=round(infProb*100))

    return render_template('detector.html')

@app.route('/status',methods=["POST",'GET'])
def status():
    IPaddress = socket.gethostbyname(socket.gethostname())
    if IPaddress == "127.0.0.1":
        print("No internet")
        return "OOps! No internet connection."
    else:
        myHtmlData = getData("https://prsindia.org/covid-19/cases")

        soup = BeautifulSoup(myHtmlData, 'html.parser')

        record=[]
        if len(soup.find_all('tbody'))==0:
            return "Data not available"
        for tr in soup.find_all('tbody')[0].find_all('tr'):
            li=[]
            for td in tr.find_all('td'):
                li.append(td.get_text())
            record.append(li)


        headings=['S. No.','State','Confimed','Active','Discharged','Deaths']
        total_confirmed=0
        total_active=0
        total_discharged=0
        total_death=0

        for i in record:
            total_confirmed=total_confirmed+int(i[2])
            total_active=total_active+int(i[3])
            total_discharged=total_discharged+int(i[4])
            total_death=total_death+int(i[5])

    return render_template('status.html',headings=headings,record=record,total_confirmed=total_confirmed,total_active=total_active,
                            total_discharged=total_discharged,total_death=total_death)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/pankaj_fb')
def pankaj_fb():
    return redirect("https://www.facebook.com/people/Pankaj-Kumar/100004369415341", code=302)

@app.route('/pankaj_linkedin')
def pankaj_linkedin():
    return redirect("https://www.linkedin.com/in/pankaj-kumar-353358120/", code=301)

@app.route('/pankaj_twitter')
def pankaj_twitter():
    return redirect("https://twitter.com/Pankaj53175102", code=302)

@app.route('/pankaj_github')
def pankaj_github():
    return redirect("https://github.com/panks123", code=302)



if __name__=='__main__':
    app.run(debug=True)