from .read_in_data import tsv_to_dataframe, csv_to_dataframe


def transform_class_column_to_ints(dataframe, column_name, mapping):
        """
        Takes in a dataframe and changes a class column
        (where the classes are something else than ints) to
        a column with ints based on the mapping dictionary.
        Returns the transformed dataframe.
        """
        dataframe[column_name] = dataframe[column_name].map(mapping)
        return dataframe


def get_X_and_ys(file_path, tsv=True):
    if tsv:
        dataframe = tsv_to_dataframe(file_path)
    else:
        dataframe = csv_to_dataframe
    y_sub_a_name_to_ints = {"NOT": 0, "OFF": 1}
    y_sub_b_name_to_ints = {"UNT": 0, "TIN": 1}
    y_sub_c_name_to_ints = {"IND": 0, "GRP": 1, "OTH": 2}
    dataframe = transform_class_column_to_ints(
        dataframe=dataframe,
        column_name="subtask_a",
        mapping=y_sub_a_name_to_ints,
    )
    dataframe = transform_class_column_to_ints(
        dataframe=dataframe,
        column_name="subtask_b",
        mapping=y_sub_b_name_to_ints,
    )
    dataframe = transform_class_column_to_ints(
        dataframe=dataframe,
        column_name="subtask_c",
        mapping=y_sub_c_name_to_ints,
    )
    X = dataframe['tweet'].values
    y_sub_a = dataframe['subtask_a'].values
    y_sub_b = dataframe['subtask_b'].values
    y_sub_c = dataframe['subtask_c'].values
    return (
        X,
        y_sub_a,
        y_sub_b,
        y_sub_c,
        y_sub_a_name_to_ints,
        y_sub_b_name_to_ints,
        y_sub_c_name_to_ints,
    )
