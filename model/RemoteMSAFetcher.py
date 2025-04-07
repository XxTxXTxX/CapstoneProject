import subprocess
import tempfile
import os 
from dotenv import load_dotenv # Get my database access
import paramiko  
from scp import SCPClient
import ipywidgets as widgets
from IPython.display import display

class RemoteMSAFetcher:
    def __init__(self, ssh_host, ssh_user, ssh_key_path, remote_dir, database_path):
        self.ssh_host = ssh_host
        self.ssh_user = ssh_user
        self.ssh_key_path = ssh_key_path
        self.remote_dir = remote_dir
        self.database_path = database_path

    def get_msa(self, sequence, local_output_path):
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.fasta', delete=False) as f:
            f.write(f">userInput\n{sequence}\n")
            f.flush()
            local_fasta = f.name

        remote_fasta = os.path.join(self.remote_dir, "input.fasta")
        remote_a3m = os.path.join(self.remote_dir, "output.a3m")
        local_a3m = tempfile.NamedTemporaryFile(suffix='.a3m', delete=False).name

        progress = widgets.IntProgress(
            value=0,
            min=0,
            max=100,
            step=1,
            description='Progress:',
            style={'description_width': 'initial'}
        )
        display(progress)

        try:
            print("Uploading FASTA file...")
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(self.ssh_host, username=self.ssh_user, key_filename=self.ssh_key_path)

            with SCPClient(ssh.get_transport()) as scp:
                scp.put(local_fasta, remote_fasta)

            progress.value = 25
            print("FASTA file uploaded.")

            print("Running hhblits...")
            hhblits_cmd = (
                f"/home/jameslikelywood/hh-suite/build/bin/hhblits -i {remote_fasta} -d {self.database_path} "
                f"-oa3m {remote_a3m} -n 3 -cpu 8 -norealign"
            )
            stdin, stdout, stderr = ssh.exec_command(hhblits_cmd)

            for line in stdout.readlines():
                print(f"STDOUT: {line.strip()}")
            for line in stderr.readlines():
                print(f"STDERR: {line.strip()}")

            progress.value = 50
            print("hhblits completed.")

            exit_status = stdout.channel.recv_exit_status()
            if exit_status != 0:
                error_msg = stderr.read().decode()
                raise RuntimeError(f"Remote hhblits failed: {error_msg}")

            print("Downloading output.a3m...")
            subprocess.run([
                "scp", "-i", self.ssh_key_path,
                f"{self.ssh_user}@{self.ssh_host}:{remote_a3m}",
                local_output_path
            ], check=True)

            print(f"MSA file downloaded to {local_output_path}")

            progress.value = 75
            print("Output.a3m downloaded.")

            if os.path.exists(local_fasta):
                os.remove(local_fasta)
            if ssh:
                cleanup_cmd = f"rm -f {remote_fasta} {remote_a3m}"
                ssh.exec_command(cleanup_cmd)
                ssh.close()

            progress.value = 100
            print("Cleaned up and finished.")
            return local_a3m

        except Exception as e:
            print(f"Error: {e}")
            progress.value = 0
            return None

# msa_fetcher = RemoteMSAFetcher(
#     ssh_host="34.44.84.176",
#     ssh_user="jameslikelywood",
#     ssh_key_path="/Users/hahayes/.ssh/gcp_manual_key",
#     remote_dir="/home/jameslikelywood/hh-suite/tmp",
#     database_path="/home/jameslikelywood/hh-suite/database/uniclust30_2018_08/uniclust30_2018_08"
# )

# msa_file = msa_fetcher.get_msa("MASMTGGQQMGRIPGNSPRMVLLESEQFLTELTRLFQKCRSSGSVFITLKKYDGRTKPIPRKSSVEGLEPAENKCLLRATDGKRKISTVVSSKEVNKFQMAYSNLLRANMDGLKKRDKKNKSKKSKPAQGGEQKLISEEDDSAGSPMPQFQTWEEFSRAAEKLYLADPMKVRVVLKYRHVDGNLCIKVTDDLVCLVYRTDQAQDVKKIEKFHSQLMRLMVAKESRNVTMETE")
