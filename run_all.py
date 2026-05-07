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
        "src/03_analysis_tables.py",
        "src/04_enhanced_tables.py",
        "src/05_column_figures.py",
        "src/06_additional_figures.py",
        "src/07_novel_contributions.py",
        "src/08_tex_integrity_check.py"
    ]

    for script in scripts:
        print(f"\n---> Executing {script} <---")
        result = subprocess.run([sys.executable, script], cwd=os.path.dirname(os.path.abspath(__file__)))
        
        if result.returncode != 0:
            print(f"\n[ERROR] execution of {script} failed with return code {result.returncode}. Aborting.")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("Final paper figures are saved in `figures_col/`")
    print("Final paper tables and numeric provenance are saved in `tables/`")
    print("=" * 60)

if __name__ == "__main__":
    main()
