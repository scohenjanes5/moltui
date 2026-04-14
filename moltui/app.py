import sys
from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Header, Static
from textual_image.widget import Image as ImageWidget

from .elements import Molecule
from .image_renderer import ImageRenderer, rotation_matrix
from .parsers import load_molecule

PIXELS_PER_CELL_X = 8
PIXELS_PER_CELL_Y = 16


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
    dark_bg = reactive(True)

    def __init__(self, molecule: Molecule, **kwargs):
        super().__init__(**kwargs)
        self.molecule = molecule
        self._image_widget = ImageWidget(None, id="mol-image")
        mol_radius = molecule.radius()
        self.camera_distance = max(4.0, mol_radius * 3.0)

    def compose(self) -> ComposeResult:
        yield self._image_widget

    def on_mount(self) -> None:
        self._rebuild()

    def on_resize(self) -> None:
        self._rebuild()

    def _rebuild(self) -> None:
        width = self.size.width
        height = self.size.height
        if width < 2 or height < 2:
            return

        px_w = width * PIXELS_PER_CELL_X
        px_h = height * PIXELS_PER_CELL_Y

        bg = (0, 0, 0) if self.dark_bg else (255, 255, 255)
        renderer = ImageRenderer(px_w, px_h, bg_color=bg)
        rot = rotation_matrix(self.rot_x, self.rot_y, self.rot_z)

        mol = self.molecule
        if not self.show_bonds:
            mol = Molecule(atoms=mol.atoms, bonds=[])

        renderer.render_molecule(mol, rot, self.camera_distance)
        self._image_widget.image = renderer.to_pil_image()

    def watch_rot_x(self) -> None:
        self._rebuild()

    def watch_rot_y(self) -> None:
        self._rebuild()

    def watch_rot_z(self) -> None:
        self._rebuild()

    def watch_camera_distance(self) -> None:
        self._rebuild()

    def watch_show_bonds(self) -> None:
        self._rebuild()

    def watch_dark_bg(self) -> None:
        self._rebuild()


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
    #mol-image {
        width: 1fr;
        height: 1fr;
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
        Binding("i", "toggle_bg", "Background"),
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
        info = f" {Path(self.filepath).name} | {n} atoms, {b} bonds | ←↑↓→ rotate | +/- zoom | b bonds | i bg | r reset | q quit"
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

    def action_toggle_bg(self) -> None:
        self.viewport.dark_bg = not self.viewport.dark_bg


def run():
    if len(sys.argv) < 2:
        print("Usage: moltui <file.xyz|file.cube>")
        sys.exit(1)

    filepath = sys.argv[1]
    molecule = load_molecule(filepath)
    app = MolTUI(molecule=molecule, filepath=filepath)
    app.run()
