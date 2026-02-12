"""Test script to verify image loading and analysis work correctly."""

import sys
from pathlib import Path

# Setup paths
ui_dir = Path(__file__).resolve().parent
project_root = ui_dir.parent
if str(project_root / "src") not in sys.path:
    sys.path.insert(0, str(project_root / "src"))

import dearpygui.dearpygui as dpg


def test_image_in_dpg():
    """Test loading and displaying an image in DearPyGui."""
    from PIL import Image
    import numpy as np

    dpg.create_context()

    # Create texture registry
    tex_reg = dpg.add_texture_registry()
    print(f"Created texture registry: {tex_reg}")

    # Create a simple test image (100x100 red square)
    test_img = np.zeros((100, 100, 4), dtype=np.float32)
    test_img[:, :, 0] = 1.0  # Red
    test_img[:, :, 3] = 1.0  # Alpha
    flat = test_img.flatten().tolist()

    # Add texture
    tex_tag = dpg.add_raw_texture(
        width=100, height=100, default_value=flat,
        format=dpg.mvFormat_Float_rgba, parent=tex_reg
    )
    print(f"Created texture: {tex_tag}")

    # Create viewport and window
    dpg.create_viewport(title="Image Test", width=800, height=600)

    with dpg.window(tag="main_window"):
        dpg.add_text("Test Image Loading")
        dpg.add_separator()

        # Try to load test image
        dpg.add_text("Generated red square:")
        dpg.add_image(tex_tag, width=100, height=100)

        dpg.add_separator()

        # Try to load a PNG from Results folder if it exists
        results_path = project_root / "tests" / "Results"
        if results_path.exists():
            png_files = list(results_path.glob("*.png"))
            if png_files:
                dpg.add_text(f"Loading PNG from Results: {png_files[0].name}")
                try:
                    img = Image.open(str(png_files[0])).convert("RGBA")
                    w, h = img.size
                    if w > 600:
                        scale = 600 / w
                        w = 600
                        h = int(h * scale)
                        img = img.resize((w, h), Image.Resampling.LANCZOS)

                    data = np.array(img).astype(np.float32) / 255.0
                    flat2 = data.flatten().tolist()
                    tex_tag2 = dpg.add_raw_texture(
                        width=w, height=h, default_value=flat2,
                        format=dpg.mvFormat_Float_rgba, parent=tex_reg
                    )
                    dpg.add_image(tex_tag2, width=w, height=h)
                    print(f"Loaded PNG: {png_files[0]}")
                except Exception as e:
                    dpg.add_text(f"Error loading PNG: {e}", color=(255, 100, 100))
                    print(f"Error: {e}")
            else:
                dpg.add_text("No PNG files in Results folder")
        else:
            dpg.add_text(f"Results folder not found: {results_path}")

    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("main_window", True)
    dpg.start_dearpygui()
    dpg.destroy_context()


def test_analysis():
    """Test running analysis without UI."""
    print("\n" + "="*60)
    print("Testing TestDataAnalyzer")
    print("="*60)

    data_path = project_root / "tests" / "Data"
    results_path = project_root / "tests" / "Results"

    print(f"Data path: {data_path}")
    print(f"Data path exists: {data_path.exists()}")

    if data_path.exists():
        pkl_files = list(data_path.glob("*.pkl"))
        print(f"PKL files found: {[f.name for f in pkl_files]}")

        if pkl_files:
            try:
                from ddmtolab.Methods.test_data_analysis import TestDataAnalyzer

                analyzer = TestDataAnalyzer(
                    data_path=str(data_path),
                    save_path=str(results_path),
                    figure_format='png',
                    log_scale=True,
                    clear_results=True
                )

                print("Running analysis...")
                result = analyzer.run()
                print("Analysis completed!")

                # Check results
                png_files = list(results_path.glob("*.png"))
                print(f"Generated PNG files: {[f.name for f in png_files]}")

            except Exception as e:
                import traceback
                print(f"Analysis error: {e}")
                traceback.print_exc()
        else:
            print("No PKL files to analyze. Run an optimization first.")
    else:
        print("Data folder does not exist.")

    print("="*60 + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", action="store_true", help="Test analysis only")
    parser.add_argument("--image", action="store_true", help="Test image loading in DPG")
    args = parser.parse_args()

    if args.analysis:
        test_analysis()
    elif args.image:
        test_image_in_dpg()
    else:
        # Run both tests
        test_analysis()
        test_image_in_dpg()
