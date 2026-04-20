#!/usr/bin/env python3
"""Tests for SelectionTablePanel event filtering."""

from __future__ import annotations

from types import SimpleNamespace

from textual.message import Message

from moltui.selection_table_panel import SelectionTablePanel


class _DummySelection(SelectionTablePanel):
    def __init__(self) -> None:
        super().__init__(table_id="dummy")

    def _has_rows(self) -> bool:
        return False

    def _populate_table(self) -> None:
        return

    def _selection_message(self, row_key: int) -> Message:
        return Message()


def test_row_highlight_ignored_when_table_unfocused(monkeypatch) -> None:
    panel = _DummySelection()
    panel._populating = False
    posted: list[Message] = []

    monkeypatch.setattr(panel, "has_class", lambda class_name: class_name == "visible")
    monkeypatch.setattr(panel, "post_message", lambda message: posted.append(message))

    event = SimpleNamespace(
        data_table=SimpleNamespace(has_focus=False),
        row_key=SimpleNamespace(value="2"),
    )
    panel.on_data_table_row_highlighted(event)

    assert posted == []


def test_row_highlight_emits_when_table_focused(monkeypatch) -> None:
    panel = _DummySelection()
    panel._populating = False
    posted: list[Message] = []

    monkeypatch.setattr(panel, "has_class", lambda class_name: class_name == "visible")
    monkeypatch.setattr(panel, "post_message", lambda message: posted.append(message))

    event = SimpleNamespace(
        data_table=SimpleNamespace(has_focus=True),
        row_key=SimpleNamespace(value="2"),
    )
    panel.on_data_table_row_highlighted(event)

    assert len(posted) == 1
