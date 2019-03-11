from .data_prep_base import DataPrep
from sklearn.model_selection import train_test_split


class DataPrepOffensEval(DataPrep):
    def __init__(self):
        pass

    def create_train_test_val(
      self,
      input_file,
      output_file_train,
      output_file_test,
      output_file_val,
      random_state=10,
      test_size=0.2,
      val_size=0.1
    ):
        df = self.tsv_to_dataframe(input_file)
        train, test = train_test_split(
            df, random_state=random_state, test_size=test_size)
        train, val = train_test_split(
            train, random_state=random_state, test_size=val_size)
        train.to_csv(output_file_train)
        test.to_csv(output_file_test)
        val.to_csv(output_file_val)

    def get_X_and_ys(self, file_path, tsv=True):
        if tsv:
            dataframe = self.tsv_to_dataframe(file_path=file_path)
        else:
            dataframe = self.csv_to_dataframe(file_path=file_path)
        y_sub_a_name_to_ints = {"NOT": 0, "OFF": 1}
        y_sub_b_name_to_ints = {"UNT": 0, "TIN": 1}
        y_sub_c_name_to_ints = {"IND": 0, "GRP": 1, "OTH": 2}
        dataframe = self.transform_class_column_to_ints(
            dataframe=dataframe,
            column_name="subtask_a",
            mapping=y_sub_a_name_to_ints,
        )
        dataframe = self.transform_class_column_to_ints(
            dataframe=dataframe,
            column_name="subtask_b",
            mapping=y_sub_b_name_to_ints,
        )
        dataframe = self.transform_class_column_to_ints(
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
