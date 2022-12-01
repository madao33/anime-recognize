from django.shortcuts import render, redirect
from django.http import HttpResponseRedirect

from recognize.recognize import save_upload_file

def index(request):
    img_name = request.session.get("img_name", None)
    res = request.session.get("res", None)
    return render(request, "index.html", {"img_name": img_name, "res": res})

def upload_img(request):
    if request.method == 'POST':
        myFile = request.FILES.get("img", None)
        if not myFile:
            return render(request, "index.html", {"message": "无效的文件"})
        
        img_name, res = save_upload_file(myFile)
        request.session['img_name'] = img_name
        request.session['res'] = res
        return redirect('/')

