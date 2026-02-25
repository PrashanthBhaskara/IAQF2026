import os
import sys
import subprocess

def main():
    print("=" * 60)
    print("Starting IAQF 2026 Competition Data & Analysis Pipeline")
    print("=" * 60)

    scripts = [
        "src/01_fetch_data.py",
        "src/02_build_master_data.py",
        "src/03_analysis_and_figures.py",
        "src/04_tex_integrity_check.py"
    ]

    for script in scripts:
        print(f"\n---> Executing {script} <---")
        result = subprocess.run([sys.executable, script], cwd=os.path.dirname(os.path.abspath(__file__)))
        
        if result.returncode != 0:
            print(f"\n[ERROR] execution of {script} failed with return code {result.returncode}. Aborting.")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("Figures are saved in `figures/`")
    print("Tables are saved in `tables/`")
    print("=" * 60)

if __name__ == "__main__":
    main()
