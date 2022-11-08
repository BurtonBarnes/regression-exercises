# Question 5


def plot_variable_pairs(train):
    # plot the variables together
    sns.pairplot(data=train.drop(columns='fips'), corner=True)
    # show the plot
    plt.show()
    return train