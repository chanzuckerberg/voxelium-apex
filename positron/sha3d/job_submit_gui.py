import argparse
import json
import os.path
import shutil
import subprocess
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox


def open_file_dialog(entry):
    filepath = filedialog.askopenfilename()
    entry.delete(0, tk.END)
    entry.insert(0, filepath)


def bake_slurm_file(template_path, job_name, cmd):
    with open(template_path, 'r') as file:
        template = file.read()
    if "{{job_name}}" not in template:
        raise KeyError('Keyword "{{job_name}}" missing from SLURM template file.')
    if "{{cmd}}" not in template:
        raise KeyError('Keyword "{{cmd}}" missing from SLURM template file.')
    return (template.
        replace("{{job_name}}", job_name).
        replace("{{cmd}}", cmd)
    )


def run_command_background(command):
    # Start the subprocess with the command
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return process


def wait_for_process(process):
    # Wait for a background process to complete
    stdout, stderr = process.communicate()
    return process.returncode, str(stdout), str(stderr)


def main(args):
    if args.wd is not None:
        os.chdir(args.wd)

    defaults_file = args.defaults
    logdir_root = args.logdir_root

    # Load defaults if defaults file exists
    if os.path.isfile(defaults_file):
        with open(defaults_file, 'r') as json_file:
            defaults = json.load(json_file)

        default_input_path = defaults["input_path"] if "input_path" in defaults else ""
        default_sum_mask_path = defaults["sum_mask_path"] if "sum_mask_path" in defaults else ""
        default_flags = defaults["flags"] if "flags" in defaults else ""
        default_job_name = defaults["job_name"] if "job_name" in defaults else ""
        default_slurm_temp_path = defaults["slurm_temp_path"] if "slurm_temp_path" in defaults else ""
    else:
        default_input_path = ""
        default_sum_mask_path = ""
        default_flags = "--gpu 0 -p"
        default_job_name = ""
        default_slurm_temp_path = ""

    def save_form(dialog):
        data = {
            "input_path": input_path_entry.get(),
            "sum_mask_path": sum_mask_path_entry.get(),
            "flags": flags_entry.get(),
            "job_name": job_name_entry.get(),
            "slurm_temp_path": slurm_temp_path_entry.get()
        }
        try:
            with open('.positron_defaults.json', 'w') as json_file:
                json.dump(data, json_file)
            if dialog:
                messagebox.showinfo("Saved!", f"Settings saved!")
        except Exception:
            messagebox.showerror("Error!", f"There was an error when saving the settings.")
            raise Exception

    def submit_form():
        save_form(False)

        input_path = input_path_entry.get()
        sum_mask_path = sum_mask_path_entry.get()
        flags = flags_entry.get()
        job_name = job_name_entry.get()
        slurm_temp_path = slurm_temp_path_entry.get()

        os.makedirs(logdir_root, exist_ok=True)

        logdir = os.path.join("positron_logdirs", job_name)
        if os.path.isdir(logdir):
            overwrite = messagebox.askquestion("Overwrite?", "Logdir already exists. Do you wish to overwrite?")
            if overwrite == messagebox.YES:
                shutil.rmtree(logdir)
            else:
                return

        os.makedirs(logdir)

        cmd = f"positron SHA3D {input_path} {logdir} {flags}"
        if len(sum_mask_path) > 1:
            cmd += f" && positron SHA3D_summary {logdir} -m {sum_mask_path}"

        if len(slurm_temp_path) > 1:
            if os.path.isfile(slurm_temp_path):
                slurm_file = bake_slurm_file(slurm_temp_path, job_name, cmd)
                slurm_file_path = os.path.join(logdir, "slurm.bash")
                with open(slurm_file_path, 'w') as file:
                    file.write(slurm_file)
                print(f"Running command:\n   sbatch {slurm_file_path}")
                process = run_command_background(f"sbatch {slurm_file_path}")
                returncode, stdout, stderr = wait_for_process(process)
                if returncode == 0:
                    messagebox.showinfo(
                        "Success!",
                        f"SLURM job was submitted!"
                    )
                else:
                    messagebox.showerror(
                        "Error!",
                        f"There was an error in the submission of the SLURM job: \n {stderr}"
                    )
            else:
                messagebox.showerror(
                    "File not found!",
                    f"SLURM template file was not found!"
                )
                return
        else:
            run_command_background(cmd)
            print(f"Running command:\n   {cmd}")

    root = tk.Tk()
    root.title("Positron SHA3D submission GUI")
    root.geometry("800x320")

    # Apply a modern theme
    style = ttk.Style(root)
    style.theme_use('clam')  # 'clam', 'alt', 'default', and 'classic' are some available themes

    main_frame = ttk.Frame(root, padding="10 10 10 10")
    main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    root.grid_columnconfigure(0, weight=1)
    root.grid_rowconfigure(0, weight=1)
    main_frame.grid_columnconfigure(1, weight=1)

    ttk.Label(main_frame, text="Job name").grid(row=0, column=0, padx=10, pady=10, sticky=tk.W)
    job_name_entry = ttk.Entry(main_frame)
    job_name_entry.insert(0, default_job_name)
    job_name_entry.grid(row=0, column=1, padx=10, pady=10, ipady=5, sticky=(tk.W, tk.E))

    ttk.Label(main_frame, text="Input STAR-file").grid(row=1, column=0, padx=10, pady=10, sticky=tk.W)
    input_path_entry = ttk.Entry(main_frame)
    input_path_entry.insert(0, default_input_path)
    input_path_entry.grid(row=1, column=1, padx=10, pady=10, ipady=5, sticky=(tk.W, tk.E))
    ttk.Button(main_frame, text="Browse", command=lambda: open_file_dialog(input_path_entry)).grid(
        row=1, column=2, padx=10, pady=10)

    ttk.Label(main_frame, text="Summary Mask (optional)").grid(row=2, column=0, padx=10, pady=10, sticky=tk.W)
    sum_mask_path_entry = ttk.Entry(main_frame)
    sum_mask_path_entry.insert(0, default_sum_mask_path)
    sum_mask_path_entry.grid(row=2, column=1, padx=10, pady=10, ipady=5, sticky=(tk.W, tk.E))
    ttk.Button(main_frame, text="Browse", command=lambda: open_file_dialog(sum_mask_path_entry)).grid(
        row=2, column=2, padx=10, pady=10)

    ttk.Label(main_frame, text="Flags (optional)").grid(row=3, column=0, padx=10, pady=10, sticky=tk.W)
    flags_entry = ttk.Entry(main_frame)
    flags_entry.insert(0, default_flags)
    flags_entry.grid(row=3, column=1, padx=10, pady=10, ipady=5, sticky=(tk.W, tk.E))

    ttk.Label(main_frame, text="SLURM Template (optional)").grid(row=4, column=0, padx=10, pady=10, sticky=tk.W)
    slurm_temp_path_entry = ttk.Entry(main_frame)
    slurm_temp_path_entry.insert(0, default_slurm_temp_path)
    slurm_temp_path_entry.grid(row=4, column=1, padx=10, pady=10, ipady=5, sticky=(tk.W, tk.E))
    ttk.Button(main_frame, text="Browse", command=lambda: open_file_dialog(slurm_temp_path_entry)).grid(
        row=4, column=2, padx=10, pady=10)

    ttk.Button(main_frame, text="Save & Submit", command=submit_form).grid(row=5, column=0, pady=20, padx=10)
    ttk.Button(main_frame, text="Save", command=save_form).grid(row=5, column=2, pady=20, padx=10)

    root.mainloop()


def append_args(parser):
    parser.add_argument('--wd', help='working directory', type=str, default=None)
    parser.add_argument('--logdir_root', help='root of the log directories', type=str, default="positron_logdirs")
    parser.add_argument('--defaults', help='default value JSON-file', type=str, default=".positron_defaults.json")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="Simple GUI for submitting spectral heterogeneity analysis (SHA) 3D jobs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    append_args(parser)
    args = parser.parse_args()

    main(args)
