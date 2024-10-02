import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt

st.title('1 uzd') #1. Sukurkite "Streamlit" programėlę, kurioje būtų rodomas užrašas "Hello, World!" su antrašte ir paantrašte.

st.subheader('Hello World!')

st.title('2 uzd') #2. Sukurkite "Streamlit" programą, kuri įkelia CSV failą ir rodo duomenų lentelę (DataFrame).

uploaded_file = st.file_uploader("Pasirinkite CSV faila", type="csv")

#kai ikeliam faila:
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    #Rodome duomenys
    st.write(df)
    #3Sukurkite programą, kuri įkelia CSV failą ir rodo apibendrintą statistiką (vidurkį, medianą, min, max) kiekvienam skaitiniam stulpeliui.
    fig, ax = plt.subplots()
    ax.plot(df['Atstumas (km)'], df['Kaina ($)'], marker='o')
    ax.set_xlabel('Atstumas (km)')
    ax.set_ylabel('Kaina ($)')
    ax.set_title('Atstumo nuo kainos priklausomybe')

    # Rodome grafiką
    st.pyplot(fig)
    # 4. Sukurkite programą, kuri įkelia CSV failą ir rodo apibendrintą statistiką (vidurkį, medianą, min, max) kiekvienam skaitiniam stulpeliui.
    st.write(df.describe())
    #importuojame altair
    x_axis = st.selectbox('Select Dydis-axis', df.columns)
    y_axis = st.selectbox('Select Kaina-axis', df.columns)

    # Sukuriame linijinį grafiką
    chart = alt.Chart(df).mark_line(point=True).encode(
        x=x_axis,
        y=y_axis
    ).properties(
        title=f'{y_axis} vs {x_axis}'
    )

    # Rodome grafiką
    st.altair_chart(chart, use_container_width=True)


    #5Sukurkite programą, kuri įkelia CSV failą ir rodo pasirinkto skaitinio stulpelio histogramą.
    column = st.selectbox('Select Column', df.columns)

    # Sukuriame histogramą
    chart = alt.Chart(df).mark_bar().encode(
        alt.X(column, bin=True),
        y='count()'
    ).properties(
        title=f'Histogram of {column}'
    )

    # Rodome histogramą
    st.altair_chart(chart, use_container_width=True)


    #6Sukurkite programą, kuri įkelia CSV failą ir rodo skaičių stulpelių koreliacijos šilumos žemėlapį.