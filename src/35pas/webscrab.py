from lxml import html
import requests
import pandas as pd
import matplotlib.pyplot as plt

def scraped_data():
    url = 'https://fbref.com/en/comps/676/stats/UEFA-Euro-Stats'
    page = requests.get(url)
    tree = html.fromstring(page.content)

    table = tree.xpath('//table[@id="stats_squads_standard_for"]')[0]

    # sukuriam stulpelius ir pasalinam pirmus 6 stulpelius jie tusti
    columns = table.xpath('.//thead/tr/th[@data-stat]/@data-stat')[6:]

    # Pridedame stulpelį komandos pavadinimui
    columns.insert(0, "team")

    # sukuriam eilutes
    rows = table.xpath('.//tbody/tr')
    data = []
    for row in rows:
        # Ištraukiame komandos pavadinimą
        team_name = row.xpath('.//th[@data-stat="team"]/a/text()')
        if team_name:  # Patikriname, ar komandos pavadinimas egzistuoja
            team_name = team_name[0]
            row_data = [team_name]
            for col in columns[1:]:
                cell = row.xpath(f'.//td[@data-stat="{col}"]//text()')
                row_data.append(cell[0] if cell else '')
            # Pridedame row_data jei turi duomenu
            if any(row_data[1:]):
                data.append(row_data)

    df = pd.DataFrame(data, columns=columns)

    return df

df = scraped_data()

# Išsaugome duomenis į CSV failą
df.to_csv('euro_teams_stats.csv', index=False)

# Įkeliam duomenis iš CSV failo
df = pd.read_csv('euro_teams_stats.csv')

# Patikrinkime stulpelių pavadinimus
print(df.columns)

# 1. Komandos pagal įmuštų įvarčių skaičių
plt.figure(figsize=(14, 7))
plt.bar(df['team'], df['goals'], label='Įmušti įvarčiai')
plt.xlabel('Komanda')
plt.ylabel('Įmuštų įvarčių skaičius')
plt.title('Įmuštų įvarčių palyginimas tarp komandų')
plt.xticks(rotation=45)
plt.legend()
plt.savefig('goals_comparison.png')


# 2. Vidurkis ivarciu
df['avg_goals_per_game'] = df['goals'] / df['games']

plt.figure(figsize=(14, 7))
plt.bar(df['team'], df['avg_goals_per_game'], label='Vidutinis įvarčių skaičius per rungtynes')
plt.xlabel('Komanda')
plt.ylabel('Vidutinis įvarčių skaičius per rungtynes')
plt.title('Vidutinis įvarčių skaičius per rungtynes tarp komandų')
plt.xticks(rotation=45)
plt.legend()
plt.savefig('avg_goals_per_game.png')


# 3Kamuolio kontrole
plt.figure(figsize=(10, 10))
plt.pie(df['possession'], labels=df['team'], autopct='%1.1f%%', startangle=140, colors=plt.get_cmap('tab10').colors)
plt.title('Kamuolio kontrolės procentas pagal komandas')
plt.savefig('possession_pie_chart.png')


# 4. Geltonos korteles
plt.figure(figsize=(14, 7))
plt.bar(df['team'], df['cards_yellow'], label='Geltonos kortelės')
plt.xlabel('Komanda')
plt.ylabel('Geltonų kortelių skaičius')
plt.title('Geltonų kortelių palyginimas tarp komandų')
plt.xticks(rotation=45)
plt.legend()
plt.savefig('yellow_cards_comparison.png')


#5. Sukurkime scatter plot grafiką su smūgių į vartus ir atliktų kampinių skaičiumi
plt.figure(figsize=(12, 8))

plt.scatter(df['goals'], df['assists'], c='blue', alpha=0.7, edgecolors='w', s=100)

plt.xlabel('Smūgiai į vartus')
plt.ylabel('Rezultatyvus perdavimai')
plt.title('Smūgių į vartus vs Rezultatyvus perdavimai')
plt.grid(True)
plt.savefig('shots_vs_assists.png')
plt.show()