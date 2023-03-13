"""Custom data structures used to store data from numerical simulations."""

try:
    from ..numerical.numerical_.DataStructures import DataTable
except Exception:
    raise Exception("numerical package not initialized")

__all__ = ['DataTable']
