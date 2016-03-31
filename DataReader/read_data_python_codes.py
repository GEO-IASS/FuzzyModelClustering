import numpy
def db_data_to_matrix(in_data, separated_column):
    """ Separates the filtered, correct data extracted from db into input and output lists
    
    :param in_data: Filtered, correct data
    :type in_data: numpy.ndarray
    :param separated_column: index position at which we have to separate the ``in_data`` variable
    :type separated_column: Integer
    :precondition: ``in_data`` should be of type numpy.ndarray
    :return:
        - ``input_data``    : numpy.ndarray containing the data of all the columns whose index value is less than that of separated_column
        - ``output_data``    :    numpy.ndarray containing the data of all the columns whose index value is greater than that of separated_column
    """
    
    try:
        if len(in_data.shape) > 1:
            (num_rows, num_cols) = in_data.shape
        else:
            num_rows = 1
            num_cols = in_data.shape[0]
        input_cols = range(separated_column)
        output_cols = range(separated_column, num_cols)
        input_data = in_data[:, input_cols]
        output_data = in_data[:, output_cols]
    except IndexError:
        raise Exception(traceback.format_exc(),
                        "No. of cols in the cleaned data matrix is less than that of length of input_param_fields_cleaned",
                        message_dict["NEIGHBOURHOOD_LOCATOR"]["0"])
    return input_data, output_data


