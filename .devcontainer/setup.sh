#!/bin/bash
set -e # Exit on any error

echo "$(date): Starting development container setup..." >/app/host/.devcontainer/setup_log.txt

### PACKAGE INSTALLATION ###
echo "Installing Python packages of pre-commit..."
pip install pre-commit

# We're already in /app/host where .git exists, so no need to cd or initialize git
echo "Current directory: $(pwd)"

# Configure git
echo "Configuring git..."
git config --global core.autocrlf input
git config --global user.email "${GITHUB_EMAIL}"
git config --global user.name "${GITHUB_USERNAME}"

# Since .git already exists, we can directly install pre-commit hooks
echo "Installing pre-commit hooks..."
pre-commit install

# ### PYTORCH GENERATOR FIX ###
# # Verify PyTorch installation before proceeding
# echo "Verifying PyTorch installation..."
# if ! python -c "import torch; print(f'PyTorch {torch.__version__} found at {torch.__file__}')"; then
# 	echo "PyTorch not found. Please ensure it is installed in the container."
# 	exit 1
# fi

# # Create the Python script to fix torch generator with better permission handling
# echo "Setting up torch generator fix..."
# cat >/tmp/fix_generator.py <<'EOL'
# #!/usr/bin/env python3
# import os
# import logging
# import shutil
# import subprocess

# def fix_torch_generator():
#     logging.basicConfig(level=logging.INFO)
#     logger = logging.getLogger(__name__)

#     try:
#         # Find torch installation path dynamically
#         import torch
#         torch_path = os.path.dirname(torch.__file__)
#         logger.info(f"Found PyTorch installation at: {torch_path}")

#         # Only the files and patterns that need modification
#         files_to_modify = {
#             os.path.join(torch_path, 'utils/data/sampler.py'): [
#                 ('torch.randperm(n, generator=generator)', 'torch.randperm(n, generator=generator, device=\'cuda\')')
#             ],
#             os.path.join(torch_path, 'utils/data/dataloader.py'): [
#                 ('g = torch.Generator()', 'g = torch.Generator(device=\'cuda\')')
#             ],
#             os.path.join(torch_path, 'utils/data/_utils/worker.py'): [
#                 ('g = torch.Generator()', 'g = torch.Generator(device=\'cuda\')')
#             ],
#         }

#         for file_path, replacements in files_to_modify.items():
#             if not os.path.exists(file_path):
#                 logger.warning(f"File not found: {file_path}")
#                 continue

#             logger.info(f"\nModifying: {file_path}")

#             # Check if we have write permissions
#             if not os.access(file_path, os.W_OK):
#                 logger.warning(f"No write permission for {file_path}, using sudo")
#                 # Try to use sudo to get permissions
#                 try:
#                     # Create backup with sudo if needed
#                     backup_path = file_path + '.backup.original'
#                     if not os.path.exists(backup_path):
#                         subprocess.run(['sudo', 'cp', file_path, backup_path], check=True)
#                         subprocess.run(['sudo', 'chmod', '644', backup_path], check=True)
#                         logger.info(f"Created backup at: {backup_path} using sudo")

#                     # Read file content
#                     with open(file_path, 'r') as file:
#                         filedata = file.read()

#                     # Track if any changes were made
#                     original_content = filedata

#                     # Make all replacements
#                     for original, new in replacements:
#                         count = filedata.count(original)
#                         logger.info(f"Found {count} occurrences of: {original}")
#                         filedata = filedata.replace(original, new)

#                     # Only write if changes were made
#                     if original_content != filedata:
#                         # Write to a temporary file
#                         temp_file = '/tmp/torch_fix_temp'
#                         with open(temp_file, 'w') as file:
#                             file.write(filedata)

#                         # Use sudo to copy the temp file to the destination
#                         subprocess.run(['sudo', 'cp', temp_file, file_path], check=True)
#                         os.remove(temp_file)
#                         logger.info(f"Modified {file_path} using sudo")
#                     else:
#                         logger.info(f"No changes needed in {file_path}")

#                     continue
#                 except Exception as e:
#                     logger.error(f"Sudo approach failed: {str(e)}")
#                     # Fall back to normal approach

#             # Normal approach (when we have permissions)
#             try:
#                 # Create backup
#                 backup_path = file_path + '.backup.original'
#                 if not os.path.exists(backup_path):
#                     shutil.copy2(file_path, backup_path)
#                     logger.info(f"Created backup at: {backup_path}")

#                 # Read file content
#                 with open(file_path, 'r') as file:
#                     filedata = file.read()

#                 # Track if any changes were made
#                 original_content = filedata

#                 # Make all replacements
#                 for original, new in replacements:
#                     count = filedata.count(original)
#                     logger.info(f"Found {count} occurrences of: {original}")
#                     filedata = filedata.replace(original, new)

#                 # Only write if changes were made
#                 if original_content != filedata:
#                     with open(file_path, 'w') as file:
#                         file.write(filedata)
#                     logger.info(f"Modified {file_path}")
#                 else:
#                     logger.info(f"No changes needed in {file_path}")
#             except Exception as e:
#                 logger.error(f"Error modifying {file_path}: {str(e)}")
#                 return False

#             # Verify the changes
#             with open(file_path, 'r') as file:
#                 content = file.read()
#                 for original, new in replacements:
#                     if original in content:
#                         logger.warning(f"Warning: {original} still exists in {file_path}")
#                     if new in content:
#                         logger.info(f"Success: Found {new} in {file_path}")

#         return True
#     except Exception as e:
#         logger.error(f"An error occurred: {str(e)}")
#         return False

# if __name__ == "__main__":
#     success = fix_torch_generator()
#     if success:
#         print("Successfully modified torch files")
#     else:
#         print("Failed to modify torch files")
#         exit(1)  # Exit with error code if fix failed
# EOL

# # Execute the torch generator fix
# echo "Executing torch generator fix..."
# python /tmp/fix_generator.py

# rm /tmp/fix_generator.py

### VERIFICATION ###
echo "Verification:"
echo "Git user: $(git config --global user.name)"
echo "Git email: $(git config --global user.email)"
echo "Pre-commit: $(pre-commit --version)"
if [ -f .pre-commit-config.yaml ]; then
	echo ".pre-commit-config.yaml found"
else
	echo "Warning: .pre-commit-config.yaml not found in $(pwd)"
fi

# Verify debugpy installation
echo "Verifying debugpy installation..."
if python -c "import debugpy; print(f'debugpy {debugpy.__version__} found at {debugpy.__file__}')" 2>/dev/null; then
	echo "debugpy is installed in the user environment"
else
	echo "Warning: debugpy is not installed in the user environment"
fi

### SSH SETUP ###
# SSH key setup
if grep -q "SSH_KEY=" .env && grep -q "BEGIN.*PRIVATE KEY" .env; then
	echo "SSH key found in .env file. Setting up SSH..."

	# Create .ssh directory if it doesn't exist
	mkdir -p ~/.ssh
	chmod 700 ~/.ssh

	# Extract the SSH key from .env and save it
	# Use sed to extract everything between the first and last quotes after SSH_KEY=
	SSH_KEY=$(grep -A 20 "SSH_KEY=" .env | sed -n 's/SSH_KEY="\(.*\)"/\1/p' | tr -d '\n')
	echo "$SSH_KEY" >~/.ssh/id_ed25519
	chmod 600 ~/.ssh/id_ed25519

	# Generate the public key from the private key
	ssh-keygen -y -f ~/.ssh/id_ed25519 >~/.ssh/id_ed25519.pub 2>/dev/null || {
		echo "Error: The SSH key in your .env file appears to be invalid."
		echo "Please ensure you've copied the entire private key including BEGIN/END lines."
		echo "Generating a new SSH key instead..."
		rm -f ~/.ssh/id_ed25519
		ssh-keygen -t ed25519 -C "github-ssh-key" -f ~/.ssh/id_ed25519 -N ""

		# Display instructions for the new key
		echo ""
		echo "==================================================================="
		echo "IMPORTANT: Two steps required:"
		echo ""
		echo "1. Add the following SSH public key to your GitHub account:"
		echo "   https://github.com/settings/keys"
		echo ""
		cat ~/.ssh/id_ed25519.pub
		echo ""
		echo "2. Add the following private key to your .env file as SSH_KEY:"
		echo "   (Copy the entire content including BEGIN/END lines)"
		echo ""
		echo "SSH_KEY=\"$(cat ~/.ssh/id_ed25519)\""
		echo "==================================================================="
		echo ""

		read -p "Press Enter after adding the SSH key to both GitHub and your .env file..."
	}

	# Test GitHub connection
	echo "Testing GitHub SSH connection..."
	ssh -T -o StrictHostKeyChecking=no git@github.com || true
else
	echo "No valid SSH key found in .env file. Generating a new SSH key..."

	# Generate a new SSH key
	ssh-keygen -t ed25519 -C "github-ssh-key" -f ~/.ssh/id_ed25519 -N ""

	# Display the public key for the user to add to GitHub
	echo ""
	echo "==================================================================="
	echo "IMPORTANT: Two steps required:"
	echo ""
	echo "1. Add the following SSH public key to your GitHub account:"
	echo "   https://github.com/settings/keys"
	echo ""
	cat ~/.ssh/id_ed25519.pub
	echo ""
	echo "2. Add the following private key to your .env file as SSH_KEY:"
	echo "   (Copy the entire content including BEGIN/END lines)"
	echo ""
	echo "SSH_KEY=\"$(cat ~/.ssh/id_ed25519)\""
	echo "==================================================================="
	echo ""

	# Prompt user to confirm they've added the key
	read -p "Press Enter after adding the SSH key to both GitHub and your .env file..."

	# Test GitHub connection
	echo "Testing GitHub SSH connection..."
	ssh -T -o StrictHostKeyChecking=no git@github.com || true
fi

echo "Setup completed successfully!"

echo "$(date): Completed setup.sh" >>/app/host/.devcontainer/setup_log.txt
