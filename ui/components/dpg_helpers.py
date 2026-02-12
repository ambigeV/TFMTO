"""Dear PyGui helper utilities for DDMTOLab UI."""

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


def load_svg_to_texture(path: str, width: int = None, height: int = None) -> Optional[Tuple[int, int, int]]:
    """
    Load SVG image and convert to texture.

    Args:
        path: Path to SVG file
        width: Target width (optional)
        height: Target height (optional)

    Returns:
        Tuple of (texture_tag, width, height) or None on failure.
    """
    try:
        import cairosvg
        from PIL import Image
        import io

        # Convert SVG to PNG in memory
        png_data = cairosvg.svg2png(url=path, output_width=width, output_height=height)
        img = Image.open(io.BytesIO(png_data)).convert("RGBA")
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
    except ImportError:
        # cairosvg not available, try alternative
        return load_image_to_texture(path)
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


def create_collapsing_header(label: str, tag: str, parent, default_open: bool = False):
    """Create a collapsing header section."""
    return dpg.add_collapsing_header(label=label, tag=tag, parent=parent,
                                      default_open=default_open)


def add_labeled_input(label: str, input_type: str, tag: str, parent,
                      default_value=None, width: int = -1, **kwargs):
    """
    Add a labeled input field.

    Args:
        label: Label text
        input_type: 'int', 'float', or 'text'
        tag: Unique tag for the input
        parent: Parent container
        default_value: Default value
        width: Input width
        **kwargs: Additional arguments for the input
    """
    dpg.add_text(label, parent=parent)
    if input_type == 'int':
        dpg.add_input_int(tag=tag, parent=parent, default_value=default_value or 0,
                          width=width, **kwargs)
    elif input_type == 'float':
        dpg.add_input_float(tag=tag, parent=parent, default_value=default_value or 0.0,
                            width=width, **kwargs)
    else:
        dpg.add_input_text(tag=tag, parent=parent, default_value=default_value or "",
                           width=width, **kwargs)


def add_labeled_combo(label: str, items: List[str], tag: str, parent,
                      default_value: str = None, width: int = -1,
                      callback: Callable = None):
    """Add a labeled combo box."""
    dpg.add_text(label, parent=parent)
    default = default_value if default_value else (items[0] if items else "")
    dpg.add_combo(items, tag=tag, parent=parent, default_value=default,
                  width=width, callback=callback)


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
