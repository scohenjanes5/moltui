import sys
from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.reactive import reactive
from textual.strip import Strip
from textual.widget import Widget
from textual.widgets import Header, Static

from .elements import Molecule
from .parsers import load_molecule
from .renderer import Renderer, rotation_matrix


class MoleculeViewport(Widget):
    DEFAULT_CSS = """
    MoleculeViewport {
        width: 1fr;
        height: 1fr;
    }
    """

    rot_x = reactive(0.5)
    rot_y = reactive(0.0)
    rot_z = reactive(0.0)
    camera_distance = reactive(5.0)
    show_bonds = reactive(True)

    def __init__(self, molecule: Molecule, **kwargs):
        super().__init__(**kwargs)
        self.molecule = molecule
        self._renderer: Renderer | None = None
        self._render_size: tuple[int, int] = (0, 0)
        mol_radius = molecule.radius()
        self.camera_distance = max(4.0, mol_radius * 3.0)

    def _rebuild(self) -> None:
        width = self.size.width
        height = self.size.height
        if width < 2 or height < 2:
            self._renderer = None
            return

        renderer = Renderer(width, height)
        rot = rotation_matrix(self.rot_x, self.rot_y, self.rot_z)

        mol = self.molecule
        if not self.show_bonds:
            mol = Molecule(atoms=mol.atoms, bonds=[])

        renderer.render_molecule(mol, rot, self.camera_distance)
        self._renderer = renderer
        self._render_size = (width, height)

    def render_line(self, y: int) -> Strip:
        w, h = self.size.width, self.size.height
        if self._renderer is None or self._render_size != (w, h):
            self._rebuild()
        if self._renderer is None:
            return Strip.blank(self.size.width)
        return self._renderer.get_strip(y)

    def _invalidate(self) -> None:
        self._renderer = None
        self.refresh()

    def watch_rot_x(self) -> None:
        self._invalidate()

    def watch_rot_y(self) -> None:
        self._invalidate()

    def watch_rot_z(self) -> None:
        self._invalidate()

    def watch_camera_distance(self) -> None:
        self._invalidate()

    def watch_show_bonds(self) -> None:
        self._invalidate()


class StatusBar(Static):
    DEFAULT_CSS = """
    StatusBar {
        dock: bottom;
        height: 1;
        background: $surface;
        color: $text-muted;
        padding: 0 1;
    }
    """


class MolTUI(App):
    CSS = """
    Screen {
        background: #000000;
    }
    """

    TITLE = "moltui"

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("up,k", "rotate_up", "Rot Up", show=False),
        Binding("down,j", "rotate_down", "Rot Down", show=False),
        Binding("left,h", "rotate_left", "Rot Left", show=False),
        Binding("right,l", "rotate_right", "Rot Right", show=False),
        Binding("comma", "rotate_cw", "Rot CW", show=False),
        Binding("full_stop", "rotate_ccw", "Rot CCW", show=False),
        Binding("plus,equal", "zoom_in", "Zoom In", show=False),
        Binding("minus", "zoom_out", "Zoom Out", show=False),
        Binding("r", "reset_view", "Reset"),
        Binding("b", "toggle_bonds", "Bonds"),
    ]

    def __init__(self, molecule: Molecule, filepath: str = "", **kwargs):
        super().__init__(**kwargs)
        self.molecule = molecule
        self.filepath = filepath

    def compose(self) -> ComposeResult:
        yield Header()
        yield MoleculeViewport(self.molecule, id="viewport")
        n = len(self.molecule.atoms)
        b = len(self.molecule.bonds)
        info = f" {Path(self.filepath).name} | {n} atoms, {b} bonds | ←↑↓→ rotate | +/- zoom | b bonds | r reset | q quit"
        yield StatusBar(info)

    @property
    def viewport(self) -> MoleculeViewport:
        return self.query_one("#viewport", MoleculeViewport)

    def action_rotate_up(self) -> None:
        self.viewport.rot_x -= 0.1

    def action_rotate_down(self) -> None:
        self.viewport.rot_x += 0.1

    def action_rotate_left(self) -> None:
        self.viewport.rot_y -= 0.1

    def action_rotate_right(self) -> None:
        self.viewport.rot_y += 0.1

    def action_rotate_cw(self) -> None:
        self.viewport.rot_z += 0.1

    def action_rotate_ccw(self) -> None:
        self.viewport.rot_z -= 0.1

    def action_zoom_in(self) -> None:
        self.viewport.camera_distance = max(1.0, self.viewport.camera_distance - 0.5)

    def action_zoom_out(self) -> None:
        self.viewport.camera_distance += 0.5

    def action_reset_view(self) -> None:
        self.viewport.rot_x = 0.5
        self.viewport.rot_y = 0.0
        self.viewport.rot_z = 0.0
        mol_radius = self.molecule.radius()
        self.viewport.camera_distance = max(4.0, mol_radius * 3.0)

    def action_toggle_bonds(self) -> None:
        self.viewport.show_bonds = not self.viewport.show_bonds


def run():
    if len(sys.argv) < 2:
        print("Usage: moltui <file.xyz|file.cube>")
        sys.exit(1)

    filepath = sys.argv[1]
    molecule = load_molecule(filepath)
    app = MolTUI(molecule=molecule, filepath=filepath)
    app.run()
