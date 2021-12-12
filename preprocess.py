import json
from tqdm import tqdm
with open('./dataset/train.json', 'r', encoding='UTF-8') as f:
    train_data = json.load(f)

with open('./dataset/test.json', 'r', encoding='UTF-8') as f:
    test_data = json.load(f)
import json
import random
import requests
import urllib.parse
from hashlib import md5

def translate_api(text):
    appid = '20211208001022842'        # 填写你的appid
    secretKey = 'dUzSUQy3f2_gGYSahKRc'    # 填写你的密钥
    myurl = 'http://api.fanyi.baidu.com/api/trans/vip/translate'
    q = text
    fromLang = 'auto'       # 原文语种
    toLang = 'en'           # 译文语种
    salt = random.randint(32768, 65536)
    sign = appid+q+str(salt)+secretKey
    m1 = md5()
    m1.update(sign.encode("utf-8"))
    sign = m1.hexdigest()

    myurl = myurl+'?appid='+appid+'&q='+urllib.parse.quote(q)+'&from='+fromLang+'&to='+toLang+'&salt='+str(salt)+'&sign='+sign
    return myurl

def generate_new_data(data,is_test = False):
    new_data = []
    myurl = '/api/trans/vip/translate'  # 通用翻译API HTTP地址

    fromLang = 'auto'       # 原文语种
    toLang = 'en'           # 译文语种
    salt = random.randint(32768, 65536)
    # 手动录入翻译内容，q存放
    # 建立会话，返回结果
    for i in tqdm(range(len(data))):
        new_res = {}
        print(len(new_data))
        text = data[i]['content'].replace('\n','')
        myurl = translate_api(text)
        response = requests.get(myurl)
        try:
            dst = json.loads(response.text)['trans_result'][0]['dst']
        except Exception:
            myurl = translate_api(text)
            response = requests.get(myurl)
            try:
                dst = json.loads(response.text)['trans_result'][0]['dst']
                print('success!')
            except Exception:
                print(response.text)
                continue
        new_res['content'] = dst
        if not is_test:
            new_res['label'] =  data[i]['label']
        new_data.append(new_res)
    return new_data

new_train_data = generate_new_data(train_data)
with open('./dataset/new_train_data.json','w') as f:
    json.dump(new_train_data,f)

new_test_data = generate_new_data(test_data,is_test=True)