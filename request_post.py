import requests

import requests

response = requests.post('https://eurisko-development-project3.herokuapp.com/')

print("POST sent")
print(response.status_code)
print(response.json())