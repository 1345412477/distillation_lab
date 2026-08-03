"""DistillationLab entry point (PyInstaller) - with startup error capture"""
import sys
import os
import traceback

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from app.main import main
    main()
except SystemExit:
    # PyQt's sys.exit() raises SystemExit - this is normal
    raise
except Exception:
    # Write all startup exceptions to a log file (same directory as exe)
    exe_dir = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(os.path.abspath(__file__))
    error_log = os.path.join(exe_dir, "startup_error.log")
    with open(error_log, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("DistillationLab - Startup Failed\n")
        f.write("=" * 60 + "\n")
        f.write(f"Time: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Working Dir: {os.getcwd()}\n")
        f.write(f"Exe Path: {sys.executable if getattr(sys, 'frozen', False) else __file__}\n")
        f.write(f"Python: {sys.version}\n")
        f.write(f"Platform: {sys.platform}\n")
        f.write("-" * 60 + "\n")
        f.write(traceback.format_exc())
        f.write("=" * 60 + "\n")
        f.write("Please send this log file to the developer for troubleshooting.\n")

    # Try to show a GUI error dialog (if PyQt6 is available)
    try:
        from PyQt6.QtWidgets import QApplication, QMessageBox
        qt_app = QApplication(sys.argv)
        QMessageBox.critical(
            None,
            "DistillationLab - Startup Failed",
            f"Application failed to start. Error log saved to:\n{error_log}\n\n"
            f"Common causes:\n"
            f"1. Missing Microsoft Visual C++ Redistributable\n"
            f"   (https://aka.ms/vs/17/release/vc_redist.x64.exe)\n"
            f"2. No NVIDIA GPU or outdated CUDA drivers\n"
            f"   (Requires driver version >= 525.x for CUDA 12.4)\n\n"
            f"Please check the log file for detailed information."
        )
    except Exception:
        # If GUI dialog also fails, print to console
        print(f"\n{'='*60}", file=sys.stderr)
        print("DistillationLab - Startup Failed!", file=sys.stderr)
        print(f"Error log saved to: {error_log}", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)
        print("Common causes:", file=sys.stderr)
        print("1. Missing Microsoft Visual C++ Redistributable", file=sys.stderr)
        print("   https://aka.ms/vs/17/release/vc_redist.x64.exe", file=sys.stderr)
        print("2. No NVIDIA GPU or outdated CUDA drivers", file=sys.stderr)

    # Wait for user input to prevent window from closing immediately
    try:
        input("\nPress Enter to exit...")
    except:
        pass
    sys.exit(1)
