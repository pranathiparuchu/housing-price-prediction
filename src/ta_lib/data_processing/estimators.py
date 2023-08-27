"""Module to provide a common place for all useful Estimators regardless of library.

This module simply lists a bunch of curated ``Estimator`` classes from various libraries.
This is useful in loading a class using the name of the regressor regarless of the library
it comes from.

.. code:: python

    from ta_lib.data_processing import estimators

    cls = getattr(estimators, 'LinearRegression')


If needed, we can remove this module and instead use fully qualified name to load the classes.
e.g:

.. code:: python

    regressors = {
        'linear_regressor': 'sklearn.linear_model.LinearRegression',
    }

    def load_class(qual_cls_name):
        module_name, cls_name = qual_cls_name.rsplit('.', 1)
        try:
            mod = importlib.import_module(module_name)
        except Exception:
            logger.exception(f'Failed to import module : {module_name}')
        else:
            return getattr(mod, cls_name)
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# List of estimators exposed by the module
__all__ = ["CombinedAttributesAdder"]


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    """sklearn wrapper estimator for creating new features

    Parameters
    ----------
    add_bedrooms_per_room: bool
        argument to specify whether to calculate bedrooms per room and return it as feature


    """

    def __init__(self, add_bedrooms_per_room=True):  # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(
        self,
        X,
        rooms_ix=3,
        bedrooms_ix=4,
        population_ix=5,
        households_ix=6,
        column_names=None,
    ):
        """Create new features

        Parameters
        ----------
        X : pd.DataFrame or np.Array
            Dataframe/2D Array consisting of independant features
        rooms_ix : integer
            argument to specify index of total_rooms
        bedrooms_ix : integer
            argument to specify index of total_bedrooms
        population_ix : integer
            argument to specify index of population
        households_ix : integer
            argument to specify index of households

        Returns
        -------
        np.Array
            Input data with new features : rooms_per_household, population_per_household, bedrooms_per_room
        """
        if X.shape[1] < max(rooms_ix, bedrooms_ix, population_ix, households_ix):
            print("X does not enough features to perform operation")
        else:
            rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
            population_per_household = X[:, population_ix] / X[:, households_ix]
            if self.add_bedrooms_per_room:
                bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
                return np.c_[
                    X, rooms_per_household, population_per_household, bedrooms_per_room
                ]

            else:
                return np.c_[X, rooms_per_household, population_per_household]
