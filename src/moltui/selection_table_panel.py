from __future__ import annotations

from textual.app import ComposeResult
from textual.message import Message
from textual.widget import Widget
from textual.widgets import DataTable


class SelectionTablePanel(Widget):
    """Shared table-selection behavior for right-side panels."""

    def __init__(self, table_id: str) -> None:
        super().__init__()
        self._table_id = table_id
        self._populating = False

    def compose(self) -> ComposeResult:
        yield DataTable(id=self._table_id, cursor_type="row")

    def on_mount(self) -> None:
        if self._has_rows():
            self._populate_table()

    def _table(self) -> DataTable:
        return self.query_one(f"#{self._table_id}", DataTable)

    def _with_populating(self, fn) -> None:
        self._populating = True
        try:
            fn()
        finally:
            self._populating = False

    def _center_row(self, table: DataTable, row: int) -> None:
        if table.row_count == 0:
            return
        viewport_rows = max(1, table.size.height - 1)
        center_target = max(0, row - viewport_rows // 2)
        table.scroll_to(y=center_target, animate=False, immediate=True)

    def _set_cursor_row(self, row: int, *, center: bool = False) -> None:
        table = self._table()
        if table.row_count == 0:
            return
        self._with_populating(lambda: table.move_cursor(row=row, scroll=not center))
        if center:
            self.call_after_refresh(self._center_row, table, row)

    def select_row_key(self, row_key: int, *, center: bool = False) -> bool:
        table = self._table()
        for row, (rk, _) in enumerate(table.rows.items()):
            if rk.value is not None and int(rk.value) == row_key:
                self._set_cursor_row(row, center=center)
                return True
        return False

    def emit_current_highlight(self, dt: DataTable) -> None:
        if not self.has_class("visible") or dt.row_count == 0:
            return
        rk = list(dt.rows.keys())[dt.cursor_row]
        if rk.value is None:
            return
        self.post_message(self._selection_message(int(rk.value)))

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if self._populating or not self.has_class("visible"):
            return
        if event.row_key is None or event.row_key.value is None:
            return
        self.post_message(self._selection_message(int(event.row_key.value)))

    def _has_rows(self) -> bool:
        raise NotImplementedError

    def _populate_table(self) -> None:
        raise NotImplementedError

    def _selection_message(self, row_key: int) -> Message:
        raise NotImplementedError
