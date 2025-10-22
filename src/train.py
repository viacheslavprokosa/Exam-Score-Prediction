from model import train_model, load_dataset, make_pairplot


def main():
    print("Loading dataset...")
    df = load_dataset()

    print("Training model...")
    df = train_model(df)

    print("Generating pairplot...")
    make_pairplot(df)

    print(df.head())
    print("✅ Done!")


if __name__ == "__main__":
    main()