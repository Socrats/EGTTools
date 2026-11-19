"""
The `egttools.numerical.DataStructures` submodule contains helper data structures.
"""
from __future__ import annotations
import collections.abc
import numpy
import numpy.typing
import typing
__all__ = ['DataTable']
class DataTable:
    """
    
                Store tabular data together with column headers and column types.
            
    """
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self, nb_rows: typing.SupportsInt | typing.SupportsIndex, nb_columns: typing.SupportsInt | typing.SupportsIndex, headers: collections.abc.Sequence[str], column_types: collections.abc.Sequence[str]) -> None:
        """
                        Create a data table.
        
                        Parameters
                        ----------
                        nb_rows : int
                            Number of rows in the table.
                        nb_columns : int
                            Number of columns in the table.
                        headers : list[str]
                            Column names.
                        column_types : list[str]
                            Type label for each column.
        
                        Examples
                        --------
                        >>> from egttools.numerical.DataStructures import DataTable
                        >>> table = DataTable(
                        ...     2,
                        ...     2,
                        ...     ["name", "value"],
                        ...     ["str", "float"]
                        ... )
        """
    @property
    def cols(self) -> int:
        """
        Number of columns in the table.
        """
    @property
    def column_types(self) -> list[str]:
        """
                        Column type labels.
        
                        Returns
                        -------
                        list[str]
                            Type label for each column.
        """
    @column_types.setter
    def column_types(self, arg0: collections.abc.Sequence[str]) -> None:
        ...
    @property
    def data(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
        """
                        Table contents.
        
                        Returns
                        -------
                        numpy.ndarray
                            Two-dimensional array storing the table values.
        """
    @data.setter
    def data(self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, n]"]) -> None:
        ...
    @property
    def headers(self) -> list[str]:
        """
                        Column names.
        
                        Returns
                        -------
                        list[str]
                            Header for each column.
        """
    @headers.setter
    def headers(self, arg0: collections.abc.Sequence[str]) -> None:
        ...
    @property
    def rows(self) -> int:
        """
        Number of rows in the table.
        """
__init__: str = 'The `egttools.numerical.DataStructures` submodule contains helpful data structures.'
