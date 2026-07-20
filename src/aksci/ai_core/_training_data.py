"""
Bundled training examples for AK-SCI's embedded local diagnostic model.

This is intentionally small and fast to train (a couple hundred short
examples). It is NOT meant to rival a large language model -- it is a
lightweight, offline-first classifier that recognizes the *shape* of common
Python / data-science / automation-script errors so the error handler can
respond instantly without needing network access or an API key.

Each tuple is: (error_type + ": " + message_pattern, category_label)

Categories (16 total):
    missing_column        pandas/polars KeyError on a column name
    key_error_generic     dict access, not a dataframe
    type_mismatch         incompatible types in an operation
    shape_mismatch        numpy/sklearn array or matrix shape mismatch
    division_by_zero      dividing by zero (scalar or array)
    missing_module        package not installed / bad import
    index_out_of_range    list/array/DataFrame positional index out of range
    attribute_error       object missing an attribute/method (often None)
    null_values           NaN / None / inf where clean numeric data expected
    value_error_generic   malformed value passed to a function
    file_not_found        missing file/path on disk
    permission_denied     OS-level permission errors on files/sockets
    json_decode_error     malformed JSON parsing
    connection_error      network/API connectivity failures
    memory_error          out-of-memory conditions
    timeout_error         operation exceeded a time limit
"""
from __future__ import annotations

TRAINING_EXAMPLES: list[tuple[str, str]] = [
    # --- missing_column (pandas/polars KeyError on a column name) ---
    ("KeyError: 'price'", "missing_column"),
    ("KeyError: 'customer_id'", "missing_column"),
    ("KeyError: 'Age'", "missing_column"),
    ("KeyError: column not found in dataframe", "missing_column"),
    ("ColumnNotFoundError: unable to find column revenue", "missing_column"),
    ("KeyError: 'total_sales'", "missing_column"),
    ("KeyError: 'order_date'", "missing_column"),
    ("KeyError: \"['region'] not in index\"", "missing_column"),
    ("KeyError: 'lead_score'", "missing_column"),
    ("ColumnNotFoundError: revenue", "missing_column"),
    ("KeyError: 'zipcode'", "missing_column"),
    ("KeyError: \"None of [Index(['status'], dtype='object')] are in the [columns]\"", "missing_column"),
    ("SchemaError: column 'sku' not found in schema", "missing_column"),
    ("KeyError: 'email_address'", "missing_column"),

    # --- key_error_generic (dict access, not a dataframe) ---
    ("KeyError: 'username'", "key_error_generic"),
    ("KeyError: 'api_key'", "key_error_generic"),
    ("KeyError: dictionary key not found", "key_error_generic"),
    ("KeyError: 'config'", "key_error_generic"),
    ("KeyError: 'access_token'", "key_error_generic"),
    ("KeyError: 'user_id'", "key_error_generic"),
    ("KeyError: 'settings'", "key_error_generic"),
    ("KeyError: 0", "key_error_generic"),
    ("KeyError: 'endpoint'", "key_error_generic"),
    ("KeyError: 'DATABASE_URL'", "key_error_generic"),

    # --- type_mismatch ---
    ("TypeError: unsupported operand type(s) for +: 'int' and 'str'", "type_mismatch"),
    ("TypeError: can only concatenate str (not int) to str", "type_mismatch"),
    ("TypeError: 'NoneType' object is not subscriptable", "type_mismatch"),
    ("TypeError: 'int' object is not callable", "type_mismatch"),
    ("TypeError: expected string or bytes-like object", "type_mismatch"),
    ("TypeError: cannot compare str and int", "type_mismatch"),
    ("TypeError: 'NoneType' object is not iterable", "type_mismatch"),
    ("TypeError: unsupported operand type(s) for -: 'str' and 'int'", "type_mismatch"),
    ("TypeError: 'str' object cannot be interpreted as an integer", "type_mismatch"),
    ("TypeError: list indices must be integers or slices, not str", "type_mismatch"),
    ("TypeError: object of type 'NoneType' has no len()", "type_mismatch"),
    ("TypeError: argument of type 'int' is not iterable", "type_mismatch"),

    # --- shape_mismatch (numpy / sklearn) ---
    ("ValueError: shapes (3,4) and (3,) not aligned", "shape_mismatch"),
    ("ValueError: operands could not be broadcast together with shapes (10,2) (5,2)", "shape_mismatch"),
    ("ValueError: Found input variables with inconsistent numbers of samples", "shape_mismatch"),
    ("ValueError: matmul: Input operand has a mismatch in its core dimension", "shape_mismatch"),
    ("ValueError: cannot reshape array of size 12 into shape (5,3)", "shape_mismatch"),
    ("ValueError: X has 5 features, but StandardScaler is expecting 8 features", "shape_mismatch"),
    ("ValueError: Number of labels does not match number of samples", "shape_mismatch"),
    ("ValueError: all the input array dimensions except for the concatenation axis must match exactly", "shape_mismatch"),
    ("ValueError: Length of values does not match length of index", "shape_mismatch"),

    # --- division_by_zero ---
    ("ZeroDivisionError: division by zero", "division_by_zero"),
    ("ZeroDivisionError: float division by zero", "division_by_zero"),
    ("RuntimeWarning: invalid value encountered in true_divide", "division_by_zero"),
    ("ZeroDivisionError: integer division or modulo by zero", "division_by_zero"),
    ("RuntimeWarning: divide by zero encountered in log", "division_by_zero"),
    ("FloatingPointError: divide by zero encountered in double_scalars", "division_by_zero"),
    ("ZeroDivisionError: modulo by zero", "division_by_zero"),
    ("RuntimeWarning: divide by zero encountered in divide", "division_by_zero"),
    ("ZeroDivisionError: 0.0 cannot be used as a denominator", "division_by_zero"),

    # --- missing_module ---
    ("ModuleNotFoundError: No module named 'requests'", "missing_module"),
    ("ModuleNotFoundError: No module named 'polars'", "missing_module"),
    ("ImportError: cannot import name 'X' from 'Y'", "missing_module"),
    ("ModuleNotFoundError: No module named 'sklearn'", "missing_module"),
    ("ModuleNotFoundError: No module named 'flask'", "missing_module"),
    ("ModuleNotFoundError: No module named 'yaml'", "missing_module"),
    ("ImportError: DLL load failed while importing _imaging", "missing_module"),
    ("ModuleNotFoundError: No module named 'dotenv'", "missing_module"),
    ("ImportError: No module named psycopg2", "missing_module"),

    # --- index_out_of_range ---
    ("IndexError: list index out of range", "index_out_of_range"),
    ("IndexError: index 5 is out of bounds for axis 0 with size 3", "index_out_of_range"),
    ("IndexError: single positional indexer is out-of-bounds", "index_out_of_range"),
    ("IndexError: index 10 is out of bounds for axis 1 with size 4", "index_out_of_range"),
    ("IndexError: pop from empty list", "index_out_of_range"),
    ("IndexError: string index out of range", "index_out_of_range"),

    # --- attribute_error ---
    ("AttributeError: 'DataFrame' object has no attribute 'foo'", "attribute_error"),
    ("AttributeError: 'NoneType' object has no attribute 'split'", "attribute_error"),
    ("AttributeError: 'list' object has no attribute 'shape'", "attribute_error"),
    ("AttributeError: module 'numpy' has no attribute 'int'", "attribute_error"),
    ("AttributeError: 'NoneType' object has no attribute 'get'", "attribute_error"),
    ("AttributeError: 'Series' object has no attribute 'append'", "attribute_error"),
    ("AttributeError: 'dict' object has no attribute 'iteritems'", "attribute_error"),
    ("AttributeError: 'str' object has no attribute 'decode'", "attribute_error"),

    # --- null_values (NaN / None handling) ---
    ("ValueError: Input contains NaN", "null_values"),
    ("ValueError: Input X contains infinity or a value too large", "null_values"),
    ("ValueError: cannot convert float NaN to integer", "null_values"),
    ("ValueError: Input contains NaN, infinity or a value too large for dtype('float64')", "null_values"),
    ("ValueError: array must not contain infs or NaNs", "null_values"),
    ("ValueError: Input y contains NaN", "null_values"),
    ("TypeError: float() argument must be a string or a number, not 'NoneType'", "null_values"),
    ("ValueError: Input contains NaN, infinity or a value too large for dtype('float32')", "null_values"),

    # --- value_error_generic ---
    ("ValueError: invalid literal for int() with base 10: 'abc'", "value_error_generic"),
    ("ValueError: could not convert string to float: 'N/A'", "value_error_generic"),
    ("ValueError: too many values to unpack", "value_error_generic"),
    ("ValueError: not enough values to unpack (expected 3, got 2)", "value_error_generic"),
    ("ValueError: time data '2024-13-01' does not match format '%Y-%m-%d'", "value_error_generic"),
    ("ValueError: unknown format is not supported", "value_error_generic"),

    # --- file_not_found ---
    ("FileNotFoundError: [Errno 2] No such file or directory: 'data.csv'", "file_not_found"),
    ("FileNotFoundError: [Errno 2] No such file or directory: 'config.yaml'", "file_not_found"),
    ("FileNotFoundError: [Errno 2] No such file or directory: '/models/model.pkl'", "file_not_found"),
    ("FileNotFoundError: No such file or directory: 'input.json'", "file_not_found"),
    ("IsADirectoryError: [Errno 21] Is a directory: 'exports'", "file_not_found"),
    ("NotADirectoryError: [Errno 20] Not a directory: 'output.txt/log'", "file_not_found"),
    ("FileNotFoundError: [Errno 2] No such file or directory: '.env'", "file_not_found"),
    ("OSError: Cannot save file into a non-existent directory: 'reports/'", "file_not_found"),
    ("FileNotFoundError: [Errno 2] No such file or directory: 'leads.xlsx'", "file_not_found"),

    # --- permission_denied ---
    ("PermissionError: [Errno 13] Permission denied: '/var/log/app.log'", "permission_denied"),
    ("PermissionError: [Errno 13] Permission denied: 'output.csv'", "permission_denied"),
    ("PermissionError: [Errno 13] Permission denied: '/etc/hosts'", "permission_denied"),
    ("PermissionError: [WinError 5] Access is denied", "permission_denied"),
    ("OSError: [Errno 13] Permission denied", "permission_denied"),
    ("PermissionError: [Errno 13] Permission denied: 'C:\\\\Program Files\\\\data'", "permission_denied"),

    # --- json_decode_error ---
    ("JSONDecodeError: Expecting value: line 1 column 1 (char 0)", "json_decode_error"),
    ("JSONDecodeError: Expecting ',' delimiter: line 3 column 5 (char 42)", "json_decode_error"),
    ("json.decoder.JSONDecodeError: Unterminated string starting at", "json_decode_error"),
    ("JSONDecodeError: Extra data: line 2 column 1 (char 15)", "json_decode_error"),
    ("simplejson.errors.JSONDecodeError: Expecting property name enclosed in double quotes", "json_decode_error"),
    ("JSONDecodeError: Expecting value: line 1 column 1 (char 0) while parsing API response", "json_decode_error"),

    # --- connection_error ---
    ("ConnectionError: HTTPSConnectionPool(host='api.example.com', port=443): Max retries exceeded", "connection_error"),
    ("ConnectionRefusedError: [Errno 111] Connection refused", "connection_error"),
    ("requests.exceptions.ConnectionError: Failed to establish a new connection", "connection_error"),
    ("URLError: <urlopen error [Errno -2] Name or service not known>", "connection_error"),
    ("ConnectionResetError: [Errno 104] Connection reset by peer", "connection_error"),
    ("requests.exceptions.SSLError: HTTPSConnectionPool: certificate verify failed", "connection_error"),
    ("socket.gaierror: [Errno -3] Temporary failure in name resolution", "connection_error"),
    ("OperationalError: could not connect to server: Connection refused", "connection_error"),

    # --- memory_error ---
    ("MemoryError: Unable to allocate 4.00 GiB for an array with shape (500000, 1000)", "memory_error"),
    ("MemoryError: Unable to allocate array with shape and data type float64", "memory_error"),
    ("numpy.core._exceptions._ArrayMemoryError: Unable to allocate 8.00 GiB", "memory_error"),
    ("MemoryError: out of memory", "memory_error"),
    ("OSError: [Errno 12] Cannot allocate memory", "memory_error"),
    ("MemoryError: Unable to allocate 12.0 GiB for an array with shape (1000000, 1500)", "memory_error"),
    ("numpy.core._exceptions._ArrayMemoryError: Unable to allocate 2.10 GiB for an array", "memory_error"),

    # --- timeout_error ---
    ("TimeoutError: [Errno 110] Connection timed out", "timeout_error"),
    ("requests.exceptions.Timeout: HTTPSConnectionPool: Read timed out. (read timeout=30)", "timeout_error"),
    ("requests.exceptions.ReadTimeout: Read timed out", "timeout_error"),
    ("concurrent.futures.TimeoutError", "timeout_error"),
    ("TimeoutError: Operation timed out after 30000 milliseconds", "timeout_error"),
    ("asyncio.exceptions.TimeoutError", "timeout_error"),
    ("requests.exceptions.ConnectTimeout: HTTPSConnectionPool: Connection timed out", "timeout_error"),
]
