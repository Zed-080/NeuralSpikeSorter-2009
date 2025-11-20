import shutil
import datetime
import subprocess

def save_results(dataset="D1-D6", threshold=0.95, refractory=60, folder="Colab Notebooks"):
    """
    Save outputs folder as a timestamped zip into Google Drive.
    """
    # timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # sanitize threshold (replace '.' with '_')
    threshold_str = str(threshold).replace('.', '_')
    # build filename
    zip_name = f"results_{dataset}_thr{threshold_str}_refr{refractory}_{timestamp}"
    output_path = f"/content/drive/MyDrive/{folder}/{zip_name}"
    # save archive
    shutil.make_archive(output_path, 'zip', 'outputs')
    print(f"Saved archive: {output_path}.zip")


def commit_results(threshold=0.95, refractory=60, script_run="main_train.py + main_infer.py"):
    """
    Commit outputs to GitHub with detailed metadata in the commit message.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    gpu_info = subprocess.getoutput("nvidia-smi --query-gpu=name --format=csv,noheader")
    threshold_str = str(threshold).replace('.', '_')

    commit_message = (
        f"Run at {timestamp} | GPU: {gpu_info} | Scripts: {script_run} | "
        f"Threshold={threshold_str} | Refractory={refractory}"
    )

    # Git commands
    subprocess.run(["git", "add", "outputs/*"], shell=True)
    subprocess.run(["git", "commit", "-m", commit_message], shell=True)
    subprocess.run(["git", "push", "origin", "main"], shell=True)
    print("Committed results to GitHub with message:")
    print(commit_message)
