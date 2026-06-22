#!/usr/bin/env python
"""Upload project files to server via SFTP

Credentials are read from environment variables:
  SEETACLOUD_HOST  (default: connect.westd.seetacloud.com)
  SEETACLOUD_PORT  (default: 31560)
  SEETACLOUD_USER  (default: root)
  SEETACLOUD_PASSWORD  (required)
"""
import paramiko
import os
import sys

host = os.environ.get("SEETACLOUD_HOST", "connect.westd.seetacloud.com")
port = int(os.environ.get("SEETACLOUD_PORT", "31560"))
user = os.environ.get("SEETACLOUD_USER", "root")
password = os.environ.get("SEETACLOUD_PASSWORD", "")

if not password:
    print("ERROR: SEETACLOUD_PASSWORD environment variable is required.")
    sys.exit(1)

print("Connecting for file upload...")
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(host, port=port, username=user, password=password, timeout=15)
print("Connected!")

sftp = client.open_sftp()

# Create remote directory
try:
    sftp.mkdir("/root/UAV")
except OSError:
    pass

local_root = "h:/Projects/UAV"
remote_root = "/root/UAV"

include_exts = {'.py', '.yaml', '.yml', '.sh', '.txt', '.json', '.jsonl', '.md'}
exclude_dirs = {'__pycache__', '.git', 'outputs', 'checkpoints', 'logs', '.claude', 'data', '.vscode'}

uploaded = 0
for root, dirs, files in os.walk(local_root):
    # Filter out excluded directories
    dirs[:] = [d for d in dirs if d not in exclude_dirs and not d.startswith('.')]

    rel_path = os.path.relpath(root, local_root)
    if rel_path == '.':
        remote_dir = remote_root
    else:
        remote_dir = os.path.join(remote_root, rel_path).replace('\\', '/')

    # Create remote directory
    try:
        sftp.stat(remote_dir)
    except (FileNotFoundError, OSError):
        try:
            sftp.mkdir(remote_dir)
        except OSError:
            pass

    for f in files:
        ext = os.path.splitext(f)[1].lower()
        # Upload .py, .yaml, .sh, .txt, .json, .jsonl, .md files
        if ext in include_exts or f in ['requirements.txt', 'autodl_setup.sh'] or ext == '':
            local_path = os.path.join(root, f)
            remote_path = os.path.join(remote_dir, f).replace('\\', '/')
            try:
                sftp.put(local_path, remote_path)
                uploaded += 1
                if uploaded % 10 == 0:
                    print(f"  Uploaded {uploaded} files...")
            except Exception as e:
                print(f"  FAILED: {local_path} -> {remote_path}: {e}")

sftp.close()
client.close()
print(f"\nDone! Uploaded {uploaded} files to {remote_root}")
