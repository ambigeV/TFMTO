"""Dear PyGui helper utilities for DDMTOLab UI."""

import os
import dearpygui.dearpygui as dpg
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Callable


_tex_registry_tag = None


def get_texture_registry():
    """Get or create the shared texture registry."""
    global _tex_registry_tag
    if _tex_registry_tag is None or not dpg.does_item_exist(_tex_registry_tag):
        _tex_registry_tag = dpg.add_texture_registry()
    return _tex_registry_tag


def load_image_to_texture(path: str) -> Optional[Tuple[int, int, int]]:
    """
    Load image into shared texture registry.

    Returns:
        Tuple of (texture_tag, width, height) or None on failure.
    """
    try:
        from PIL import Image
        img = Image.open(path).convert("RGBA")
        w, h = img.size
        data = np.array(img).astype(np.float32) / 255.0
        flat = data.flatten().tolist()
        reg = get_texture_registry()
        # Use static_texture for DearPyGui 2.x compatibility
        tex_tag = dpg.add_static_texture(
            width=w, height=h, default_value=flat,
            parent=reg,
        )
        return tex_tag, w, h
    except Exception:
        return None


def add_checkbox_group(label: str, items: List[str], tag: str, parent,
                       height: int = 200, default_checked: List[str] = None,
                       callback: Callable = None):
    """Create a scrollable child window with checkboxes for multi-select."""
    if label:
        dpg.add_text(label, parent=parent)
    default_checked = default_checked or []
    with dpg.child_window(tag=tag, parent=parent, height=height, border=True):
        for item in items:
            checked = item in default_checked
            dpg.add_checkbox(label=item, tag=f"{tag}__{item}", default_value=checked,
                             callback=callback)


def get_checkbox_selections(tag: str, items: List[str]) -> List[str]:
    """Return list of checked item names from a checkbox group."""
    selected = []
    for item in items:
        cb_tag = f"{tag}__{item}"
        if dpg.does_item_exist(cb_tag) and dpg.get_value(cb_tag):
            selected.append(item)
    return selected


def update_checkbox_group(tag: str, items: List[str], default_checked: List[str] = None,
                          callback: Callable = None):
    """Rebuild checkboxes when the item list changes."""
    if not dpg.does_item_exist(tag):
        return
    default_checked = default_checked or []
    dpg.delete_item(tag, children_only=True)
    for item in items:
        checked = item in default_checked
        dpg.add_checkbox(label=item, tag=f"{tag}__{item}", default_value=checked, parent=tag,
                         callback=callback)


def set_all_checkboxes(tag: str, items: List[str], value: bool):
    """Set all checkboxes in a group to the same value."""
    for item in items:
        cb_tag = f"{tag}__{item}"
        if dpg.does_item_exist(cb_tag):
            dpg.set_value(cb_tag, value)


def show_error_modal(msg: str, title: str = "Error"):
    """Show a modal error dialog with an OK button."""
    modal_tag = "error_modal"
    if dpg.does_item_exist(modal_tag):
        dpg.delete_item(modal_tag)

    with dpg.window(label=title, modal=True, tag=modal_tag, no_close=True,
                    width=450, height=180, pos=[575, 360]):
        dpg.add_text(msg, wrap=420)
        dpg.add_separator()
        dpg.add_spacer(height=10)
        dpg.add_button(label="OK", width=100,
                       callback=lambda: dpg.delete_item(modal_tag))


def show_info_modal(msg: str, title: str = "Info"):
    """Show a modal info dialog with an OK button."""
    modal_tag = "info_modal"
    if dpg.does_item_exist(modal_tag):
        dpg.delete_item(modal_tag)

    with dpg.window(label=title, modal=True, tag=modal_tag, no_close=True,
                    width=350, autosize=True, pos=[625, 380]):
        dpg.add_text(msg, wrap=320, color=(180, 255, 180))
        dpg.add_separator()
        dpg.add_spacer(height=10)
        dpg.add_button(label="OK", width=100,
                       callback=lambda: dpg.delete_item(modal_tag))


def show_confirm_dialog(msg: str, on_yes: Callable, on_no: Callable = None,
                        title: str = "Confirm", yes_label: str = "Yes",
                        no_label: str = "No", cancel_label: str = "Cancel",
                        show_cancel: bool = True):
    """
    Show a confirmation dialog with Yes/No/Cancel buttons.

    Args:
        msg: Message to display
        on_yes: Callback when Yes is clicked
        on_no: Callback when No is clicked (optional)
        title: Dialog title
        yes_label: Label for Yes button
        no_label: Label for No button
        cancel_label: Label for Cancel button
        show_cancel: Whether to show Cancel button
    """
    modal_tag = "confirm_modal"
    if dpg.does_item_exist(modal_tag):
        dpg.delete_item(modal_tag)

    def _on_yes():
        dpg.delete_item(modal_tag)
        if on_yes:
            on_yes()

    def _on_no():
        dpg.delete_item(modal_tag)
        if on_no:
            on_no()

    def _on_cancel():
        dpg.delete_item(modal_tag)

    with dpg.window(label=title, modal=True, tag=modal_tag, no_close=True,
                    width=420, no_resize=True, autosize=True, pos=[565, 360]):
        dpg.add_text(msg, wrap=400)
        dpg.add_separator()
        dpg.add_spacer(height=5)
        with dpg.group(horizontal=True):
            dpg.add_button(label=yes_label, width=120, callback=_on_yes)
            dpg.add_spacer(width=5)
            dpg.add_button(label=no_label, width=100, callback=_on_no)
            if show_cancel:
                dpg.add_spacer(width=5)
                dpg.add_button(label=cancel_label, width=80, callback=_on_cancel)


def copy_text_to_clipboard(text: str):
    """Copy text to system clipboard (Windows)."""
    import sys
    try:
        if sys.platform == 'win32':
            import subprocess
            process = subprocess.Popen(['clip'], stdin=subprocess.PIPE, creationflags=subprocess.CREATE_NO_WINDOW)
            process.communicate(text.encode('utf-8'))
        else:
            import subprocess
            process = subprocess.Popen(['xclip', '-selection', 'clipboard'], stdin=subprocess.PIPE)
            process.communicate(text.encode('utf-8'))
    except Exception:
        pass


def _on_copy_error_click(sender, app_data, user_data):
    """Callback for error copy button."""
    copy_text_to_clipboard(user_data)


def format_time(seconds: float) -> str:
    """Format seconds to human-readable time string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


# ---- Shared theme creators (used by batch_mode and test_mode) ----

_arrow_btn_theme = None

def get_arrow_btn_theme():
    """Get or create the arrow button theme."""
    global _arrow_btn_theme
    if _arrow_btn_theme is None or not dpg.does_item_exist(_arrow_btn_theme):
        with dpg.theme() as _arrow_btn_theme:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (180, 180, 180))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (150, 150, 150))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (120, 120, 120))
                dpg.add_theme_color(dpg.mvThemeCol_Text, (40, 40, 40))
    return _arrow_btn_theme


_disabled_btn_theme = None

def get_disabled_btn_theme():
    """Get or create the disabled button theme."""
    global _disabled_btn_theme
    if _disabled_btn_theme is None or not dpg.does_item_exist(_disabled_btn_theme):
        with dpg.theme() as _disabled_btn_theme:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (100, 100, 100))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (100, 100, 100))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (100, 100, 100))
                dpg.add_theme_color(dpg.mvThemeCol_Text, (160, 160, 160))
    return _disabled_btn_theme


def parse_param_value(value, param_type: str):
    """Parse parameter value, supporting list input like [100, 200]."""
    if isinstance(value, str):
        value = value.strip()
        # Check if it's a list format
        if value.startswith('[') and value.endswith(']'):
            try:
                import ast
                parsed = ast.literal_eval(value)
                if isinstance(parsed, list):
                    # Convert list elements to appropriate type
                    if param_type == 'int':
                        return [int(x) for x in parsed]
                    elif param_type == 'float':
                        return [float(x) for x in parsed]
                    return parsed
            except:
                pass
        # Single value - convert to appropriate type
        try:
            if param_type == 'int':
                return int(value)
            elif param_type == 'float':
                return float(value)
        except:
            pass
    return value


def open_file_location(sender, app_data, user_data):
    """Open the folder containing the file in Windows Explorer."""
    import subprocess
    file_path = user_data
    folder_path = os.path.dirname(file_path)
    if os.path.exists(folder_path):
        subprocess.Popen(['explorer', '/select,', file_path.replace('/', '\\')])


def copy_image_to_clipboard(sender, app_data, user_data):
    """Copy image to clipboard (Windows only)."""
    import sys
    file_path = user_data

    if not os.path.exists(file_path):
        return

    try:
        from PIL import Image
        import io

        if sys.platform == 'win32':
            import win32clipboard

            # Open image and convert to BMP format for clipboard
            img = Image.open(file_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Save to bytes as BMP
            output = io.BytesIO()
            img.save(output, format='BMP')
            bmp_data = output.getvalue()[14:]  # Remove BMP header
            output.close()

            # Copy to clipboard
            win32clipboard.OpenClipboard()
            win32clipboard.EmptyClipboard()
            win32clipboard.SetClipboardData(win32clipboard.CF_DIB, bmp_data)
            win32clipboard.CloseClipboard()
        else:
            show_info_modal("Copy to clipboard is only supported on Windows.")
    except ImportError:
        show_error_modal("Please install pywin32: pip install pywin32")
    except Exception as e:
        show_error_modal(f"Failed to copy image: {e}")


def get_algo_source_path(algo_name: str, category: str) -> Path:
    """Get the source file path for an algorithm."""
    from utils.algo_scanner import DISPLAY_TO_FILE
    file_name = DISPLAY_TO_FILE.get(algo_name, algo_name)

    ui_dir = Path(__file__).resolve().parent.parent
    project_root = ui_dir.parent
    return project_root / 'src' / 'ddmtolab' / 'Algorithms' / category / f'{file_name}.py'


def open_algo_source(sender, app_data, user_data):
    """Open algorithm source file in PyCharm or default editor.

    user_data should be (algo_name, category).
    """
    import subprocess
    import sys
    import shutil

    algo_name, category = user_data
    source_path = get_algo_source_path(algo_name, category)

    if not source_path.exists():
        show_error_modal(f"Source file not found:\n{source_path}")
        return

    # Try PyCharm first
    pycharm_paths = [
        shutil.which('pycharm'),
        shutil.which('pycharm64'),
        r'C:\Program Files\JetBrains\PyCharm Community Edition 2024.1\bin\pycharm64.exe',
        r'C:\Program Files\JetBrains\PyCharm 2024.1\bin\pycharm64.exe',
        r'C:\Program Files\JetBrains\PyCharm Community Edition 2023.3\bin\pycharm64.exe',
        r'C:\Program Files\JetBrains\PyCharm 2023.3\bin\pycharm64.exe',
    ]

    pycharm_exe = None
    for path in pycharm_paths:
        if path and Path(path).exists():
            pycharm_exe = path
            break

    if not pycharm_exe:
        toolbox_base = Path.home() / 'AppData' / 'Local' / 'JetBrains' / 'Toolbox' / 'apps'
        if toolbox_base.exists():
            for pycharm_dir in toolbox_base.glob('PyCharm*/**/pycharm64.exe'):
                pycharm_exe = str(pycharm_dir)
                break

    if pycharm_exe:
        subprocess.Popen([pycharm_exe, str(source_path)],
                         creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0)
    else:
        if sys.platform == 'win32':
            os.startfile(str(source_path))
        elif sys.platform == 'darwin':
            subprocess.run(['open', str(source_path)])
        else:
            subprocess.run(['xdg-open', str(source_path)])
