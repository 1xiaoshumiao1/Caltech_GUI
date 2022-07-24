import requests
import json
import base64

def getByte(path):
    with open(path, 'rb') as f:
        img_byte = base64.b64encode(f.read())
    img_str = img_byte.decode('ascii')
    return img_str


img_str = getByte('set07_V010_I01109.jpg')
vis_threshold = 0.5
datas = {'file': img_str, 'vis_threshold': vis_threshold}
host = "http://192.168.0.195:8000"
req = json.dumps(datas)


r = requests.post(host, data=req)
print(r.text)



