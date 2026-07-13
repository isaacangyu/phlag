import os
import glob
import shutil

def main():
    # 1. Rename files in caster/results/
    # Pattern: distributions_topologies_*.png -> gaussian_all_*.png
    caster_results_dir = os.path.join("caster", "results")
    if os.path.exists(caster_results_dir):
        files = glob.glob(os.path.join(caster_results_dir, "distributions_topologies_*.png"))
        for f in files:
            dir_name = os.path.dirname(f)
            base_name = os.path.basename(f)
            new_name = base_name.replace("distributions_topologies_", "gaussian_all_")
            new_path = os.path.join(dir_name, new_name)
            print(f"Renaming {f} -> {new_path}")
            shutil.move(f, new_path)

    # 2. Rename files in test/
    # Patterns:
    # - report_ape_*.tsv -> report_gaussian_ape_*.tsv
    # - em_ape_*.png -> em_gaussian_ape_*.png
    # - states_ape_*.png -> states_gaussian_ape_*.png
    test_dir = "test"
    if os.path.exists(test_dir):
        # reports
        reports = glob.glob(os.path.join(test_dir, "report_ape_*.tsv"))
        for f in reports:
            base_name = os.path.basename(f)
            new_name = base_name.replace("report_", "report_gaussian_")
            new_path = os.path.join(test_dir, new_name)
            print(f"Renaming {f} -> {new_path}")
            shutil.move(f, new_path)
            
        # em plots
        em_plots = glob.glob(os.path.join(test_dir, "em_ape_*.png"))
        for f in em_plots:
            base_name = os.path.basename(f)
            new_name = base_name.replace("em_", "em_gaussian_")
            new_path = os.path.join(test_dir, new_name)
            print(f"Renaming {f} -> {new_path}")
            shutil.move(f, new_path)
            
        # states plots
        states_plots = glob.glob(os.path.join(test_dir, "states_ape_*.png"))
        for f in states_plots:
            base_name = os.path.basename(f)
            new_name = base_name.replace("states_", "states_gaussian_")
            new_path = os.path.join(test_dir, new_name)
            print(f"Renaming {f} -> {new_path}")
            shutil.move(f, new_path)

if __name__ == "__main__":
    main()
