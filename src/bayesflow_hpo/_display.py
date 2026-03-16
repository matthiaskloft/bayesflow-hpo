"""DataFrame subclass that hides the row index in display."""

from __future__ import annotations

import pandas as pd


class DisplayDataFrame(pd.DataFrame):
    """A DataFrame that suppresses the index when printed or rendered.

    All normal DataFrame operations (slicing, merging, CSV export, etc.)
    work as usual.  Only ``__repr__`` (terminal) and ``_repr_html_``
    (Jupyter) are overridden to hide the numeric row index.
    """

    @property
    def _constructor(self):
        return DisplayDataFrame

    def __repr__(self) -> str:
        return self.to_string(index=False)

    def _repr_html_(self) -> str:
        return self.style.hide().to_html()
