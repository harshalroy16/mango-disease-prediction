import matplotlib.pyplot as plt


def show_training_graphs():

    # Example training values
    # Replace with your actual history values if available

    epochs = range(1, 11)

    accuracy = [
        0.65,
        0.72,
        0.78,
        0.83,
        0.87,
        0.90,
        0.92,
        0.94,
        0.95,
        0.96
    ]

    loss = [
        0.85,
        0.65,
        0.52,
        0.40,
        0.32,
        0.25,
        0.20,
        0.16,
        0.13,
        0.10
    ]


    fig1, ax1 = plt.subplots()

    ax1.plot(
        epochs,
        accuracy
    )

    ax1.set_title(
        "Training Accuracy"
    )

    ax1.set_xlabel(
        "Epochs"
    )

    ax1.set_ylabel(
        "Accuracy"
    )


    fig2, ax2 = plt.subplots()

    ax2.plot(
        epochs,
        loss
    )

    ax2.set_title(
        "Training Loss"
    )

    ax2.set_xlabel(
        "Epochs"
    )

    ax2.set_ylabel(
        "Loss"
    )


    return fig1, fig2