try:
    import egttools.numerical.numerical as numerical
except Exception:
    raise Exception("numerical package not initialized")
else:
    from egttools.numerical.numerical import PairwiseMoran

__all__ = ['numerical', 'PairwiseMoran']
