import requests

# define the Flask app URL
url = 'http://localhost:5000/predict'

# define the news title
title = input("Enter the news title: ")

# create the JSON payload
payload = {'title': title}

# make an HTTP POST request to the Flask app
response = requests.post(url, json=payload)

# parse the JSON response and print the prediction
print(response)
prediction = response.json()
print(prediction['prediction'])