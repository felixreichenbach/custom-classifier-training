import model.training


def test_parse_args():
    args = [
        "--dataset_file=./test_dataset/test_data.jsonl",
        "--model_dir=test_output/",
        "--num_epochs=5",
        "--labels='OK NOK'",
        "--model_type=single_label",
    ]

    data_json, model_dir, num_epochs, labels, model_type = model.training.parse_args(
        args
    )

    assert (
        data_json == "./test_dataset/test_data.jsonl"
    ), f"Expected dataset_file to be './test_dataset/test_data.jsonl', got {data_json}"
    assert (
        model_dir == "test_output/"
    ), f"Expected model_output_directory to be 'test_output/', got {model_dir}"
    assert num_epochs == 5, f"Expected num_epochs to be 5, got {num_epochs}"
    assert (
        labels == "'OK NOK'"
    ), f"Expected labels to be 'OK NOK', got {labels}"
    assert (
        model_type == "single_label"
    ), f"Expected model_type to be single_label, got {model_type}"
