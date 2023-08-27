from sklearn.utils.estimator_checks import check_estimator


def test_existance():
    from ta_lib.data_processing.estimators import CombinedAttributesAdder


def test_estimator_check():
    from ta_lib.data_processing.estimators import CombinedAttributesAdder

    check_estimator(CombinedAttributesAdder())
