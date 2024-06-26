from lxml import html
import requests
import re

url = 'https://www.meteo.lt/'

page = requests.get(url)

tree = html.fromstring(page.content)
days = tree.xpath('//div[@class="day-wrap"]/h4/text()')

date = tree.xpath('//div[@class="date"]/text()')
date_list = []
for i in date:
    date_list.append(i.strip())

print(date_list)

temp = tree.xpath('//div[@class="day-wrap"]//div[@class="temprature"]text()')
temp_list = []
for i in temp:
    value = re.match(r"^\d+", i.strip())
    temp_list.append(int(value.group()))

print(temp_list)


wind = tree.xpath('//div[@class="wind"]/text()')
wind_list = []
for i in wind:
    wind_list.append(i.strip())

wind_speed = []
wind_direction = []
for i in wind_list:
    splitted = i.split()
    if len(splitted) == 3:
        wind_speed.append(splitted[0])
        wind_direction.append(splitted[2])

print(wind_speed)

import pandas as pd

df = pd.DataFrame({
    "Data": date_list,
    "Temperatura (C)": temp_list,
    "Vejas_greitis (m/s)": wind_speed,
    "Vejas_kryptis": wind_direction
})

print(df)
df.to_csv('output.csv')

import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
plt.plot(df["Data"], df["Temperatura (C)"], marker="o", linestyle='-',color='b')
plt.title("Temperatura per sav.")
plt.xlabel("Data")
plt.ylabel("Temp (C)")
plt.grid(True)
plt.show()

def avg_wind_speed(wind_speed):
    splitted = wind_speed.split('-')
    return(int(splitted[0])+int(splitted[1]))/2
