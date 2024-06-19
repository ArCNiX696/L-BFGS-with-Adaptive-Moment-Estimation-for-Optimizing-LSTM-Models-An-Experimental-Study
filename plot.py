import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def plot_corr_matrix(corr_matrix):
    plt.figure(figsize=(8,8))
    sns.heatmap(corr_matrix,annot=True,cmap='coolwarm',fmt='.4f') #annot=show corr values ,cmap= colors, fmt=decimals 
    plt.title('Correlation Matrix')
    plt.show()

def boxplot(df, target_variable=str):
    # Create a figure and axes for the plot
    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    
    # Create the boxplot for each column with respect to the target variable
    sns.boxplot(data=df.drop(columns=target_variable), ax=ax)
    sns.swarmplot(data=df.drop(columns=target_variable), color=".25", ax=ax)
    
    # Set the title and axes labels
    ax.set_title('Boxplot of distribution with respect to ' + target_variable)
    ax.set_ylabel('Distributon')
    
    # Rotate x-axis labels for better visualization
    plt.xticks(rotation=45)
    
    # Show the plot
    plt.tight_layout()
    plt.show()

# Example usage:
# Assuming df is your DataFrame and 'target' is the target variable
# boxplot_all_columns(df, 'target')

def plot_corr_scatter(df):
    sns.set_theme(style='whitegrid')
    sns.pairplot(df,height=1.6)
    plt.show()


def plot_datetime(df,column=str,column2=str):
    df= df[[column,column2]]

    df[column]= pd.to_datetime(df[column])

    plt.title(f'{column2} with respect to {column}')
    plt.xlabel(column)
    plt.ylabel(column2)
    plt.plot(df[column] , df[column2])


