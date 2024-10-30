import pandas as pd # ? Import pandas module (dataframe)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


if __name__ == "__main__":
    # ? Importing data from population.csv
    data: pd.DataFrame = pd.read_csv("population.csv") 
    
    # ? Copy and combine data from Indonesia and United States
    countries = pd.concat([
        data.loc[data["Country Name"] == "Indonesia"], 
        data.loc[data["Country Name"] == "United States"]
    ])

    # ? Remove unnecessary datas
    countries.drop(
        ["Country Name", "Country Code", "Indicator Name", "Indicator Code"], 
        axis=1, 
        inplace=True
    )

    # ? Transpose the dataframe (rotate)
    countries = countries.T 
    
    # ? Removes empty data
    countries.dropna(inplace=True)

    # ? Resets and rename the indexes
    countries = countries.reset_index().rename(
        columns={106:"Indonesia_Population", 251:"USA_Population", "index":"Year"}
    )
    
    # ? Changing number types in the data
    # ? And assigning x and y from the datas
    countries["Indonesia_Population"] = pd.to_numeric(countries["Indonesia_Population"])
    countries["Year"] = pd.to_numeric(countries["Year"])
    x_id = countries["Year"]
    y_id = countries["Indonesia_Population"]

    countries["USA_Population"] = pd.to_numeric(countries["USA_Population"])
    countries["Year"] = pd.to_numeric(countries["Year"])
    x_us = countries["Year"]
    y_us = countries["USA_Population"]
    
    # ? Training the models with 20% test size
    x_train_id, x_test_id, y_train_id, y_test_id = train_test_split(x_id, y_id, test_size=0.2)
    x_train_id = x_train_id.values.reshape(-1, 1)
    y_train_id = y_train_id.values.reshape(-1, 1)
    x_test_id = x_test_id.values.reshape(-1, 1)
    y_test_id = y_test_id.values.reshape(-1, 1)

    x_train_us, x_test_us, y_train_us, y_test_us = train_test_split(x_us, y_us, test_size=0.2)
    x_train_us = x_train_us.values.reshape(-1, 1)
    y_train_us = y_train_us.values.reshape(-1, 1)
    x_test_us = x_test_us.values.reshape(-1, 1)
    y_test_us = y_test_us.values.reshape(-1, 1)
    
    # ? Making predictions
    model_id = LinearRegression()
    model_us = LinearRegression()

    model_id = model_id.fit(x_train_id, y_train_id)
    y_predict_id = model_id.predict(x_test_id)

    model_us = model_us.fit(x_train_us, y_train_us)
    y_predict_us = model_us.predict(x_test_us)
    
    year: int = int(input("Enter a starting year: "))
    year_end: int = int(input("Enter an ending year: "))
    
    print(f"The USA's population in 2020: 329484123")
    print(f"Indonesia's population in 2020: 273523621")

    while year <= year_end:
        print(f"Year: {year}")
        print(f"    Indonesia: {model_id.predict([[year]])[0][0]}")
        print(f"    The USA  : {model_us.predict([[year]])[0][0]}")
        year += 1
