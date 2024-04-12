import csv
import pandas as pd

# finding points on the pareto curve
def find_pareto_points(candidate_solutions):
    num_solutions = len(candidate_solutions)
    pareto_points = []

    for i in range(num_solutions):
        is_pareto_point = True

        for j in range(num_solutions):
            if i != j:
                j_is_better = True
                for key in ['dice', 'Avg Error', 'Avg FA rate', 'Avg MD rate']:
                    if key != 'dice':
                        j_is_better = j_is_better and candidate_solutions[j][key] <= candidate_solutions[i][key]
                    else:
                        j_is_better = j_is_better and candidate_solutions[j][key] >= candidate_solutions[i][key]

                if j_is_better:
                    is_pareto_point = False
                    break

        if is_pareto_point:
            pareto_points.append(candidate_solutions[i])

    return pareto_points


def main(csv_path):
    candidate_solutions = []
    # reading the points from the path and saving to list of dictionaries
    with open(csv_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)  # Read the first row as headers
        for row in reader:
            candidate_solutions.append(
                dict(zip(headers, row)))  # Create a dictionary with headers as keys and row data as values

    # Find the points on the Pareto curve
    pareto_points = find_pareto_points(candidate_solutions)

    # giving each point a position relative to other points for each category
    df = pd.DataFrame(pareto_points)
    for key in ['dice', 'Avg Error', 'Avg FA rate', 'Avg MD rate']:
        df = df.sort_values(by=key, ascending=(key != 'dice'))
        df.insert(len(df.columns), key + ' Position', range(1, 1 + len(df)))

    # calculating the sum of relative positions (lowest is the best)
    df['sum positions'] = 0
    for key in ['dice', 'Avg Error', 'Avg FA rate', 'Avg MD rate']:
        df['sum positions'] = df['sum positions'] + df[key + ' Position']

    # sorting the dataframe and saving to csv
    df = df.sort_values(by='sum positions', ascending=True)
    df.to_csv(path[:path.find('\\')+1]+"pareto_points.csv", index=False)
    print(df)


if __name__ == "__main__":
    path = "2023_06_23_12_22_31\candidates.csv"
    main(path)
