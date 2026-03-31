import urllib.request
import json

# Test 3: PDF Upload
pdf_path = 'data/uploads/Umesh_Resume.pdf'
with open(pdf_path, 'rb') as f:
    pdf_data = f.read()

# Create multipart form data
boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'
body = (
    f'------WebKitFormBoundary7MA4YWxkTrZu0gW\r\n'
    f'Content-Disposition: form-data; name="file"; filename="Umesh_Resume.pdf"\r\n'
    f'Content-Type: application/pdf\r\n\r\n'
).encode() + pdf_data + b'\r\n------WebKitFormBoundary7MA4YWxkTrZu0gW--\r\n'

req = urllib.request.Request(
    'http://localhost:8000/upload',
    data=body,
    headers={'Content-Type': 'multipart/form-data; boundary=----WebKitFormBoundary7MA4YWxkTrZu0gW'},
    method='POST'
)

print('Uploading PDF... (this may take a minute for indexing)')
response = urllib.request.urlopen(req)
print('Upload Test:', response.status, response.read().decode())
