"""Processors for the data cleaning step of the worklow.

The processors in this step, apply the various cleaning steps identified
during EDA to create the training datasets.
"""
import numpy as np
import pandas as pd
from scripts import binned_income_cat
from sklearn.model_selection import StratifiedShuffleSplit

from ta_lib.core.api import (
    custom_train_test_split,
    load_dataset,
    register_processor,
    save_dataset,
    string_cleaning,
)


@register_processor("data-cleaning", "product")
def clean_product_table(context, params):
    """Clean the ``PRODUCT`` data table.

    The table contains information on the inventory being sold. This
    includes information on inventory id, properties of the item and
    so on.
    """

    input_dataset = "raw/product"
    output_dataset = "cleaned/product"

    # load dataset
    product_df = load_dataset(context, input_dataset)

    product_df_clean = (
        product_df
        # set dtypes : nothing to do here
        .passthrough()
        .transform_columns(
            product_df.columns.to_list(), string_cleaning, elementwise=False
        )
        .replace({"": np.NaN})
        # drop unnecessary cols : nothing to do here
        .coalesce(["color", "Ext_Color"], "color", delete_columns=True)
        # drop unnecessary cols : nothing to do here
        .coalesce(["MemorySize", "Ext_memorySize"], "memory_size", delete_columns=True)
        # ensure that the key column does not have duplicate records
        .remove_duplicate_rows(col_names=["SKU"], keep_first=True)
        # clean column names (comment out this line while cleaning data above)
        .clean_names(case_type="snake")
    )

    # save the dataset
    save_dataset(context, product_df_clean, output_dataset)

    return product_df_clean


@register_processor("data-cleaning", "orders")
def clean_order_table(context, params):
    """Clean the ``ORDER`` data table.

    The table containts the sales data and has information on the invoice,
    the item purchased, the price etc.
    """

    input_dataset = "raw/orders"
    output_dataset = "cleaned/orders"

    # load dataset
    orders_df = load_dataset(context, input_dataset)

    # list of columns that we want string cleaning op to be performed on.
    str_cols = list(
        set(orders_df.select_dtypes("object").columns.to_list())
        - set(["Quantity", "InvoiceNo", "Orderno", "LedgerDate"])
    )
    orders_df_clean = (
        orders_df
        # set dtypes
        .change_type(["Quantity", "InvoiceNo", "Orderno"], np.int64)
        # set dtypes
        .to_datetime("LedgerDate", format="%d/%m/%Y")
        # clean string columns (NOTE: only handling datetime columns)
        .transform_columns(str_cols, string_cleaning, elementwise=False)
        # clean column names
        .clean_names(case_type="snake").rename_columns({"orderno": "order_no"})
    )

    # save dataset
    save_dataset(context, orders_df_clean, output_dataset)
    return orders_df_clean


@register_processor("data-cleaning", "sales")
def clean_sales_table(context, params):
    """Clean the ``SALES`` data table.

    The table is a summary table obtained by doing a ``inner`` join of the
    ``PRODUCT`` and ``ORDERS`` tables.
    """
    input_product_ds = "cleaned/product"
    input_orders_ds = "cleaned/orders"
    output_dataset = "cleaned/sales"

    # load datasets
    product_df = load_dataset(context, input_product_ds)
    orders_df = load_dataset(context, input_orders_ds)

    sales_df_clean = orders_df.merge(product_df, how="inner", on="sku")

    save_dataset(context, sales_df_clean, output_dataset)
    return sales_df_clean


@register_processor("data-cleaning", "housing")
def clean_product_table(context, params):
    """Clean the ``HOUSING`` data table.

    The table contains information on the inventory being sold. This
    includes information on inventory id, properties of the item and
    so on.
    """

    input_dataset = "raw/housing"
    output_dataset = "cleaned/housing"

    # load dataset
    housing_df = load_dataset(context, input_dataset)

    housing_df_clean = (
        housing_df
        # while iterating on testing, it's good to copy the dataset(or a subset)
        # as the following steps will mutate the input dataframe. The copy should be
        # removed in the production code to avoid introducing perf. bottlenecks.
        .copy()
        # set dtypes : nothing to do here
        .passthrough().replace({"": np.NaN})
        # ensure that the key column does not have duplicate records
        .remove_duplicate_rows(
            col_names=[
                "longitude",
                "latitude",
                "housing_median_age",
                "total_rooms",
                "total_bedrooms",
                "population",
                "households",
                "median_income",
                "median_house_value",
                "ocean_proximity",
            ],
            keep_first=True,
        )
        # clean column names (comment out this line while cleaning data above)
        .clean_names(case_type="snake")
    )
    # save the dataset
    save_dataset(context, housing_df_clean, output_dataset)

    return housing_df_clean


@register_processor("data-cleaning", "train-test")
def create_training_datasets(context, params):
    """Split the ``HOUSING`` table into ``train`` and ``test`` datasets."""

    input_dataset = "cleaned/housing"
    output_train_features = "train/housing/features"
    output_train_target = "train/housing/target"
    output_test_features = "test/housing/features"
    output_test_target = "test/housing/target"

    # load dataset
    housing_df_clean = load_dataset(context, input_dataset)

    # split the data
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=params["test_size"], random_state=context.random_seed
    )
    housing_df_train, housing_df_test = custom_train_test_split(
        housing_df_clean, splitter, by=binned_income_cat
    )

    # split train dataset into features and target
    target_col = params["target"]
    train_X, train_y = (
        housing_df_train
        # split the dataset to train and test
        .get_features_targets(target_column_names=target_col)
    )

    # save the train dataset
    save_dataset(context, train_X, output_train_features)
    save_dataset(context, train_y, output_train_target)

    # split test dataset into features and target
    test_X, test_y = (
        housing_df_test
        # split the dataset to train and test
        .get_features_targets(target_column_names=target_col)
    )

    # save the datasets
    save_dataset(context, test_X, output_test_features)
    save_dataset(context, test_y, output_test_target)