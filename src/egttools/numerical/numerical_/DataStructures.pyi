"""
The `egttools.numerical.DataStructures` submodule contains helper data structures.
"""
from __future__ import annotations
import numpy
__all__: list[str] = ['DataTable']
class DataTable:
    """
    
                Store tabular data together with column headers and column types.
            
    """
    def __init__(self, nb_rows: int, nb_columns: int, headers: list[str], column_types: list[str]) -> None:
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
    def column_types(self, arg0: list[str]) -> None:
        ...
    @property
    def data(self) -> numpy.ndarray[numpy.float64[m, n]]:
        """
                        Table contents.
        
                        Returns
                        -------
                        numpy.ndarray
                            Two-dimensional array storing the table values.
        """
    @data.setter
    def data(self, arg0: numpy.ndarray[numpy.float64[m, n]]) -> None:
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
    def headers(self, arg0: list[str]) -> None:
        ...
    @property
    def rows(self) -> int:
        """
        Number of rows in the table.
        """
__init__: str = 'The `egttools.numerical.DataStructures` submodule contains helpful data structures.'
