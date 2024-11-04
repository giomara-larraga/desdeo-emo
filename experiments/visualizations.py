import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates


def plot_parallel_coordinates(solutions, reference_point):
    """
    Creates a parallel coordinates plot.

    Parameters:
    solutions (array-like): A 2D list or numpy array where each row is a solution and each column is a dimension.
    reference_point (array-like): A 1D list or numpy array representing the reference point with the same number of dimensions as the columns in solutions.
    """
    # Convert the solutions and reference point to a DataFrame
    df_solutions = pd.DataFrame(
        solutions, columns=[f"Dim{i+1}" for i in range(solutions.shape[1])]
    )
    df_reference = pd.DataFrame([reference_point], columns=df_solutions.columns)

    # Add a column to differentiate solutions and reference point
    df_solutions["Type"] = "Solution"
    df_reference["Type"] = "Reference Point"

    # Combine both dataframes
    df = pd.concat([df_solutions, df_reference], ignore_index=True)

    # Plot parallel coordinates
    plt.figure(figsize=(10, 6))
    parallel_coordinates(df, class_column="Type", color=["blue", "red"])

    # Customize the plot
    plt.title("Parallel Coordinates Plot")
    plt.xlabel("Dimension")
    plt.ylabel("Value")
    plt.grid(True)
    plt.show()
