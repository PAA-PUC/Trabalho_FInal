import pandas as pd
import pandasql as pdsql

data = pd.read_csv("../data/olist_products_dataset.csv")
data = data.set_index("product_id")

print(data.query("product_volume_cm3 > 60000"))