import pandas as pd
import pandasql as pdsql
from ortools.linear_solver import pywraplp
import warnings
warnings.filterwarnings('ignore')

# List of available cars in the fleet
def get_fleet():
    fleet = [
        {
            'code': 'Transit',
            'number': 1,
            'max_weight': 3500000,
            'max_volume': 20385200
        },
        {
            'code': 'Master',
            'number': 2,
            'max_weight': 1593000, 
            'max_volume': 7779200
        },            
        {
            'code': 'Doblo',
            'number': 3,
            'max_weight': 620000,
            'max_volume': 5765039
        },
        {
            'code': 'Fiorino',
            'number': 6,
            'max_weight': 650000, 
            'max_volume': 2751568.68
        }
    ]

    return fleet


# create data model for knapsack problem 
# paramter optimize are data to be packing into the available vehicle in totalFleet
def create_data_model(optimize, totalFleet):
    """Create the data for the example."""
    data = {}
    weights = optimize['product_weight_g'].to_list()
    volumes = optimize['product_volume_cm3'].to_list()
    
    data['weights'] = weights
    data['volumes'] = volumes
    
    data['items'] = list(range(len(weights)))
    data['num_items'] = len(weights)
    
    max_volumes = []
    max_weights = []
    truck_types = []
    
    # reserve totalFleet data to be starting from small vehicle first
    totalFleet.reverse()
    # resgister max_weight and max_volume for each available vehicle
    for tL in totalFleet:
        for i in range(tL['number']):
            max_volumes.append(tL['max_volume'])
            max_weights.append(tL['max_weight'])
            truck_types.append(tL['code'])
    
    data['max_volume'] = max_volumes 
    data['max_weight'] = max_weights 
    data['truck_types'] = truck_types
    
    data['trucks'] = list(range(len(data['max_volume'])))
    
    return data


def main():

    # Reads from csv files
    products = pd.read_csv("./data/olist_products_dataset.csv")
    products = products.set_index("product_id")
    customers = pd.read_csv("./data/olist_customers_dataset.csv")
    customers = customers.set_index("customer_id")
    orders = pd.read_csv("./data/olist_orders_dataset.csv")
    orders = orders.set_index("order_id")    
    order_items = pd.read_csv("./data/olist_order_items_dataset.csv")
    order_items = order_items.set_index("order_id")

    # Queries the datasets
    sql_query = "SELECT orders.order_id, customers.customer_id FROM orders INNER JOIN customers ON orders.customer_id = customers.customer_id"
    cust_order = pdsql.sqldf(sql_query, locals())
    print(cust_order)
    
    sql_query = "SELECT order_items.product_id FROM order_items JOIN cust_order ON order_items.order_id = cust_order.order_id"
    cust_order_items = pdsql.sqldf(sql_query, locals())
    
    print(cust_order_items)

    sql_query = "SELECT products.* FROM cust_order_items JOIN products ON products.product_id = cust_order_items.product_id WHERE products.product_weight_g>=20000"
    total_items = pdsql.sqldf(sql_query, locals())
    total_items['product_volume_cm3'] = total_items['product_width_cm'] * total_items['product_height_cm'] * total_items['product_length_cm']
    print(total_items)

    
    # Calculate the total weight and total volume of all items
    total_volume = total_items['product_volume_cm3'].sum()
    total_weight = total_items['product_weight_g'].sum()
    print('Total Volume: ', total_volume, 'cm3')
    print('Total Weight: ', total_weight, ' g')

    data = create_data_model(total_items, get_fleet())

    # Create the mip solver with the SCIP backend.
    solver = pywraplp.Solver.CreateSolver('SCIP')

    # Variables
    # x[i, j] = 1 if item i is packed in bin j.
    x = {}
    for i in data['items']:
        # print(f"{i} in {len(data['items'])}")
        for j in data['trucks']:
            x[(i, j)] = solver.IntVar(0, 1, 'x_%i_%i' % (i, j))
            # print(f"{j} in {len(data['trucks'])}")


    # Constraints
    # Each item can be in at most one bin.
    for i in data['items']:
        # print(f"{i} in {len(data['items'])}")
        solver.Add(sum(x[i, j] for j in data['trucks']) <= 1)

    # The amount packed in each bin cannot exceed its max weight.
    for j in data['trucks']:
        # print(f"{j} in {len(data['trucks'])}")
        solver.Add(
            sum(x[(i, j)] * data['weights'][i]
                for i in data['items']) <= data['max_weight'][j])

    # The amount packed in each bin cannot exceed its max volume.
    for j in data['trucks']:
        # print(f"{j} in {len(data['trucks'])}")
        solver.Add(
            sum(x[(i, j)] * data['volumes'][i]
                for i in data['items']) <= data['max_volume'][j])


    # Add objectives
    objective = solver.Objective()

    for i in data['items']:
        # print(f"{i} in {len(data['items'])}")
        for j in data['trucks']:
            # print(f"{j} in {len(data['trucks'])}")
            objective.SetCoefficient(x[(i, j)], data['volumes'][i])

    print('-----------------------first stage done-----------------------')
    objective.SetMaximization()


    status = solver.Solve()

    total_len = len(total_items)

    _totalLeftVolume = 0
    _totalLeftWeight = 0
    if status == pywraplp.Solver.OPTIMAL:
        assign = []
        total_weight = 0
        total_items = 0
        print()
        print(f'Total Items: {total_len}')
        print()
        for j in data['trucks']:
            bin_weight = 0
            bin_volume = 0
            print('Truck ', j, '[', data['truck_types'][j] ,'] - max_weight:[', "{:,.2f}".format(data['max_weight'][j]), '] - max volume:[', "{:,.2f}".format(data['max_volume'][j]), ']' )
            for i in data['items']:
                if x[i, j].solution_value() > 0:
                    assign.append(i)
                    total_items += 1
                    print('>> Item', i, '- weight:', data['weights'][i],
                        ' volumes:', data['volumes'][i])
                    bin_weight += data['weights'][i]
                    bin_volume += data['volumes'][i]

            print('Packed truck volume:', "{:,.2f}".format(bin_volume))
            print('Packed truck weight:', "{:,.2f}".format(bin_weight))
            print()

            if (bin_volume > 0) & (bin_weight > 0):
                leftVolume = data['max_volume'][j] - bin_volume
                leftWeight = data['max_weight'][j] - bin_weight
            else:
                leftVolume = 0
                leftWeight = 0

            print('Left Volume', "{:,.2f}".format(leftVolume))
            print('Left Weight', "{:,.2f}".format(leftWeight))
            print()
            print()

            total_weight += bin_weight
            _totalLeftVolume += leftVolume
            _totalLeftWeight += leftWeight

        print('Total packed weight:', "{:,.2f}".format(total_weight))
        print('Total packed volume:', "{:,.2f}".format(objective.Value()))
        print('Total item assigned:', "{:,.0f}".format(total_items))
        print()
        print("#" * 100)
        print('Total Left Volume', "{:,.2f}".format(_totalLeftVolume))
        print('Total Left Weight', "{:,.2f}".format(_totalLeftWeight))
        print("#" * 100)

    else:
        print('The problem does not have an optimal solution.')

    print()





if __name__ == "__main__":
    main()