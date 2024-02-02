import requests

url = "http://127.0.0.1:5050/invocations"
data = {
    "dataframe_split": {
        "columns": ["tamanho", "ano", "garagem"],
        "data": [[159.0, 2003, 1]],
    }
}
headers = {"Content-Type": "application/json"}

response = requests.post(url, json=data, headers=headers)

print(response.text)
