"""DDMTOLab UI - Main entry point."""

import sys
from pathlib import Path

# Ensure 'from main import ...' resolves to this module (not a duplicate)
# when this file is run as __main__
sys.modules.setdefault('main', sys.modules[__name__])

# Setup paths
_ui_dir = Path(__file__).resolve().parent
_project_root = _ui_dir.parent

if str(_project_root / "src") not in sys.path:
    sys.path.insert(0, str(_project_root / "src"))
if str(_ui_dir) not in sys.path:
    sys.path.insert(0, str(_ui_dir))

import dearpygui.dearpygui as dpg
from components.dpg_helpers import get_texture_registry, load_image_to_texture
from pages import test_mode, batch_mode
from config.constants import WINDOW_WIDTH, WINDOW_HEIGHT, COLOR_TITLE

# Store fonts globally
_fonts = {"default": None, "title": None, "logo": None, "header": None, "section": None, "tab": None, "bold": None, "header_large": None}


def get_fonts():
    """Return the fonts dictionary for use by other modules."""
    return _fonts

def _create_dark_theme():
    """Create dark theme."""
    with dpg.theme() as dark_theme:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (30, 30, 30))
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (40, 40, 40))
            dpg.add_theme_color(dpg.mvThemeCol_PopupBg, (50, 50, 50))
            dpg.add_theme_color(dpg.mvThemeCol_Border, (70, 70, 70))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (80, 80, 80))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (95, 95, 95))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, (110, 110, 110))
            dpg.add_theme_color(dpg.mvThemeCol_TitleBg, (40, 40, 40))
            dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive, (50, 50, 50))
            dpg.add_theme_color(dpg.mvThemeCol_MenuBarBg, (40, 40, 40))
            dpg.add_theme_color(dpg.mvThemeCol_Header, (60, 60, 60))
            dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, (80, 80, 80))
            dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, (100, 100, 100))
            dpg.add_theme_color(dpg.mvThemeCol_Button, (60, 60, 60))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (80, 80, 80))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (100, 100, 100))
            dpg.add_theme_color(dpg.mvThemeCol_Tab, (50, 50, 50))
            dpg.add_theme_color(dpg.mvThemeCol_TabHovered, (70, 70, 70))
            dpg.add_theme_color(dpg.mvThemeCol_TabActive, (90, 90, 90))
            dpg.add_theme_color(dpg.mvThemeCol_Text, (220, 220, 220))
            dpg.add_theme_color(dpg.mvThemeCol_CheckMark, (100, 180, 255))
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, (100, 180, 255))
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive, (130, 200, 255))
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 4)
            dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 6)
            dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 4)
    return dark_theme


def _load_logo(parent):
    """Load and display the logo from SVG."""
    from PIL import Image
    import numpy as np
    import io

    svg_logo = _ui_dir / "assets" / "logo.svg"
    img = None

    # Try loading from SVG first
    if svg_logo.exists():
        try:
            import cairosvg
            png_data = cairosvg.svg2png(url=str(svg_logo), scale=3)
            img = Image.open(io.BytesIO(png_data)).convert("RGBA")
            # Crop to content area
            bbox = img.getbbox()
            if bbox:
                img = img.crop(bbox)
        except Exception:
            img = None

    # Fallback to PNG if SVG fails
    if img is None:
        png_logo = _ui_dir / "assets" / "logo.png"
        if png_logo.exists():
            try:
                img = Image.open(str(png_logo)).convert("RGBA")
            except Exception:
                img = None

    if img is not None:
        try:
            w, h = img.size

            target_height = 60
            if h > target_height:
                scale = target_height / h
                w = int(w * scale)
                h = target_height
                img = img.resize((w, h), Image.Resampling.LANCZOS)

            data = np.array(img).astype(np.float32) / 255.0
            flat = data.flatten().tolist()

            tex_reg = get_texture_registry()
            if not dpg.does_item_exist(tex_reg):
                raise RuntimeError("Texture registry does not exist")

            tex_tag = dpg.add_static_texture(
                width=w, height=h, default_value=flat,
                parent=tex_reg
            )

            if not dpg.does_item_exist(parent):
                raise RuntimeError(f"Parent {parent} does not exist")

            dpg.add_image(tex_tag, width=w, height=h, parent=parent)
            return

        except Exception:
            pass

    logo_text = dpg.add_text("DDMTOLab", color=(50, 120, 200, 255), parent=parent)
    if _fonts["logo"]:
        dpg.bind_item_font(logo_text, _fonts["logo"])


def main():
    """Main application entry point."""
    dpg.create_context()
    get_texture_registry()

    # Set base path for data storage
    base_path = str(_project_root / "tests")

    # Create dark theme
    dark_theme = _create_dark_theme()

    # Load fonts
    with dpg.font_registry():
        font_paths = [
            "C:/Windows/Fonts/msyh.ttc",      # Microsoft YaHei (Chinese support)
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/consola.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
        ]
        for fp in font_paths:
            if Path(fp).exists():
                try:
                    _fonts["default"] = dpg.add_font(fp, 18)
                    _fonts["section"] = dpg.add_font(fp, 20)
                    _fonts["tab"] = dpg.add_font(fp, 20)
                    _fonts["title"] = dpg.add_font(fp, 24)
                    _fonts["header"] = dpg.add_font(fp, 28)
                    _fonts["logo"] = dpg.add_font(fp, 32)
                    _fonts["header_large"] = dpg.add_font(fp, 34)
                    # Try to load bold font variant
                    bold_paths = [
                        "C:/Windows/Fonts/msyhbd.ttc",    # Microsoft YaHei Bold
                        "C:/Windows/Fonts/arialbd.ttf",   # Arial Bold
                        "C:/Windows/Fonts/segoeuib.ttf",  # Segoe UI Bold
                    ]
                    for bp in bold_paths:
                        if Path(bp).exists():
                            try:
                                _fonts["bold"] = dpg.add_font(bp, 18)
                                break
                            except Exception:
                                continue
                    # Try to load a stylish title font
                    title_font_paths = [
                        "C:/Windows/Fonts/GOTHIC.TTF",     # Century Gothic
                        "C:/Windows/Fonts/segoeui.ttf",    # Segoe UI
                        "C:/Windows/Fonts/georgia.ttf",    # Georgia
                        "C:/Windows/Fonts/cambria.ttc",    # Cambria
                        "C:/Windows/Fonts/arial.ttf",      # Arial
                    ]
                    for tp in title_font_paths:
                        if Path(tp).exists():
                            try:
                                _fonts["title_stylish"] = dpg.add_font(tp, 36)
                                break
                            except Exception:
                                continue
                    break
                except Exception:
                    continue

    if _fonts["default"]:
        dpg.bind_font(_fonts["default"])

    # Apply default dark theme
    dpg.bind_theme(dark_theme)

    dpg.create_viewport(title="DDMTOLab - Data-Driven Multitask Optimization Laboratory",
                        width=WINDOW_WIDTH, height=WINDOW_HEIGHT)

    with dpg.window(tag="primary_window"):
        # Header with logo and title
        with dpg.group(horizontal=True, tag="header_group"):
            dpg.add_spacer(width=10)  # Left margin
            _load_logo("header_group")
            dpg.add_spacer(width=20)
            with dpg.group():
                dpg.add_spacer(height=1)
                # Gradient title: blue (50,130,220) -> cyan (80,200,235)
                _title_words = ["Data-Driven", " Multitask", " Optimization", " Laboratory"]
                _c_start = (50, 130, 220)
                _c_end = (80, 200, 235)
                _title_font = _fonts.get("title_stylish") or _fonts.get("header_large") or _fonts.get("header")
                # Zero-spacing theme for tight word packing
                with dpg.theme() as _title_spacing_theme:
                    with dpg.theme_component(dpg.mvAll):
                        dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 0, 0)
                with dpg.group(horizontal=True) as _title_grp:
                    for idx, word in enumerate(_title_words):
                        t = idx / max(len(_title_words) - 1, 1)
                        r = int(_c_start[0] + (_c_end[0] - _c_start[0]) * t)
                        g = int(_c_start[1] + (_c_end[1] - _c_start[1]) * t)
                        b = int(_c_start[2] + (_c_end[2] - _c_start[2]) * t)
                        tw = dpg.add_text(word, color=(r, g, b, 255))
                        if _title_font:
                            dpg.bind_item_font(tw, _title_font)
                dpg.bind_item_theme(_title_grp, _title_spacing_theme)

        # Accent line under the header
        accent_w = 1000
        accent_h = 3
        with dpg.drawlist(width=accent_w, height=accent_h, tag="header_accent"):
            # Gradient accent bar: blue->cyan fading to transparent
            segments = 60
            seg_w = accent_w / segments
            for i in range(segments):
                t = i / segments
                alpha = int(255 * (1 - t))
                cr = int(50 + (80 - 50) * t)
                cg = int(130 + (200 - 130) * t)
                cb = int(220 + (235 - 220) * t)
                dpg.draw_rectangle(
                    pmin=(i * seg_w, 0), pmax=((i + 1) * seg_w, accent_h),
                    color=(cr, cg, cb, alpha), fill=(cr, cg, cb, alpha),
                )
        dpg.add_spacer(height=5)

        # Main tab bar - use section font (20pt) for tab labels only
        _tab_font = _fonts.get("section")
        with dpg.tab_bar(tag="main_tabs"):
            with dpg.tab(label="  Test Mode  ", tag="test_tab") as _tab1:
                test_mode.create(parent="test_tab", base_path=base_path)
            with dpg.tab(label="  Batch Experiment  ", tag="batch_tab") as _tab2:
                batch_mode.create(parent="batch_tab", base_path=base_path)

        # Bind font to tab buttons after content is created with default font
        if _tab_font and _fonts.get("default"):
            dpg.bind_item_font(_tab1, _tab_font)
            dpg.bind_item_font(_tab2, _tab_font)
            # Restore default font on content containers
            for child in dpg.get_item_children(_tab1, 1) or []:
                dpg.bind_item_font(child, _fonts["default"])
            for child in dpg.get_item_children(_tab2, 1) or []:
                dpg.bind_item_font(child, _fonts["default"])

    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("primary_window", True)

    # Main loop
    while dpg.is_dearpygui_running():
        test_mode.update()
        batch_mode.update()
        dpg.render_dearpygui_frame()

    dpg.destroy_context()


if __name__ == "__main__":
    main()
