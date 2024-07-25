import os
# pip install python-dotenv
from dotenv import load_dotenv
import psycopg2
import pandas as pd

##2 Nuskaityti duomenis iš DB: shops, products, cutomers, orders.
load_dotenv()
def nuskaityt_duomenys(lenteles_pavadinimas):
    connection = None
    try:

        connection = psycopg2.connect(
            dbname=os.getenv('DATABASE_NAME'),
            user=os.getenv('DB_USERNAME'),
            password=os.getenv('PASSWORD'),
            host=os.getenv('HOST'),
            port=os.getenv('PORT')
        )

        cursor = connection.cursor()

        # SQL query to fetch all data from the specified table
        query = f"SELECT * FROM {lenteles_pavadinimas};"

        # Execute the query
        cursor.execute(query)

        rows = cursor.fetchall()

        columns = [desc[0] for desc in cursor.description]

        # konvertuojam duomenys i pandas
        df = pd.DataFrame(rows, columns=columns)

        return df

    except (Exception, psycopg2.Error) as error:
        print("Error while connecting to PostgreSQL", error)
    finally:
        if connection:
            cursor.close()
            connection.close()
            print("PostgreSQL connection is closed")



data_frame = nuskaityt_duomenys('products') #
print(data_frame)  # Print the DataFrame


# 3Pridėti po vieną įrašą kiekvinoje lentelėje naudojant python.

def ivesti_duomenys(lenteles_pavadinimas, columns, values):
    connection = None
    try:

        connection = psycopg2.connect(
            dbname=os.getenv('DATABASE_NAME'),
            user=os.getenv('DB_USERNAME'),
            password=os.getenv('PASSWORD'),
            host=os.getenv('HOST'),
            port=os.getenv('PORT')
        )

        cursor = connection.cursor()

        # Create the SQL insert query dynamically
        columns_list = ", ".join(columns)
        values_list = ", ".join(["%s"] * len(values))
        query = f"INSERT INTO {lenteles_pavadinimas} ({columns_list}) VALUES ({values_list})"

        # Execute the query
        cursor.execute(query, values)

        # Commit the changes to the database
        connection.commit()

        print(f"Record inserted successfully into {lenteles_pavadinimas} table")

    except (Exception, psycopg2.Error) as error:
        print(f"Failed to insert record into {lenteles_pavadinimas} table", error)
    finally:
        if connection:
            cursor.close()
            connection.close()
            print("PostgreSQL connection is closed")

# duomenu ivedimas pvz
#ivesti_duomenys(
#    lenteles_pavadinimas='shops',
#    columns=['shopid', 'shopname', 'location'],
#    values=['Music Brothers', '3rd Floor'])

#ivesti_duomenys(
    #lenteles_pavadinimas='products',
    #columns=['productname', 'price', 'shopid'],
    #values=['Guitar', 19.99, 4])

#ivesti_duomenys(
#    lenteles_pavadinimas='customers',
#    columns=['firstname', 'lastname', 'email'],
#    values=['Jack', 'Booze', 'jack.booze@example.com'])

#ivesti_duomenys(
    #lenteles_pavadinimas='orders',
    #columns=['customerid', 'productid', 'quantity'],
    #values=[4,6,4])


#4 Išsaugoti visus duomenis pandas lentelėse.

def duomenys_i_pandas(lenteles_pavadinimas):
    connection = None
    try:
        connection = psycopg2.connect(
            dbname=os.getenv('DATABASE_NAME'),
            user=os.getenv('DB_USERNAME'),
            password=os.getenv('PASSWORD'),
            host=os.getenv('HOST'),
            port=os.getenv('PORT')
        )

        cursor = connection.cursor()

        query = f"SELECT * FROM {lenteles_pavadinimas};"

        # performatuojam i pandas
        df = pd.read_sql_query(query, connection)

        return df
        print(f"Data from {lenteles_pavadinimas} table:")
        print(df.to_string(), "\n")

    except (Exception, psycopg2.Error) as error:
        print(f"Error while fetching data from PostgreSQL table {lenteles_pavadinimas}", error)
    finally:
        # Close the database connection
        if connection:
            cursor.close()
            connection.close()
            print("PostgreSQL connection is closed")


# kaip naudot? irasome norimas perskaityt lenteles
tables = ['shops', 'products', 'customers', 'orders']
data_frames = {table: duomenys_i_pandas(table) for table in tables}

# paprintinam duomenys is visu lenteliu
for table, df in data_frames.items():
    if df is not None:
        print(f"Data from {table} table:")
        print(df.to_string(), "\n")
    else:
        print(f"No data retrieved for {table} table.")

#savo uzklausa: klientai uzsake prekes is daugiau nei vienos parduotuves
def fetch_customers_with_multiple_shops():
    connection = None
    try:
        # Establish a connection to the PostgreSQL database
        connection = psycopg2.connect(
            dbname=os.getenv('DATABASE_NAME'),
            user=os.getenv('DB_USERNAME'),
            password=os.getenv('PASSWORD'),
            host=os.getenv('HOST'),
            port=os.getenv('PORT')
        )

        # Define the SQL query
        query = """
        SELECT DISTINCT customers.customerid, customers.firstname, customers.lastname
        FROM customers
        JOIN orders ON customers.customerid = orders.customerid
        JOIN products ON orders.productid = products.productid
        GROUP BY customers.customerid
        HAVING COUNT(DISTINCT products.shopid) > 1
        ORDER BY customers.customerid;
        """

        # Perform the query and load data into pandas DataFrame
        df = pd.read_sql_query(query, connection)

        return df

    except (Exception, psycopg2.Error) as error:
        print("Error while fetching data from PostgreSQL", error)
        return None
    finally:
        # Close the database connection
        if connection:
            connection.close()
            print("PostgreSQL connection is closed")


# Fetch data and print results
df = fetch_customers_with_multiple_shops()

if df is not None:
    print("Data from the query:")
    print(df.to_string(), "\n")
else:
    print("No data retrieved.")

