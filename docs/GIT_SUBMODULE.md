# Git Submodule Documentation

This repository uses **Aave-Simulator** as a Git submodule. The submodule is located in the `Aave-Simulator/` directory.

## Cloning the Repository with Submodule

When cloning this repository for the first time, you need to initialize and update the submodule:

```bash
# Clone the repository
git clone git@github.com:brains-group/Aave-Action-Recommender.git

# Navigate into the repository
cd Aave-Action-Recommender

# Initialize and update the submodule
git submodule update --init --recursive
```

Alternatively, you can clone with the `--recursive` flag to automatically initialize submodules:

```bash
git clone --recursive git@github.com:brains-group/Aave-Action-Recommender.git
```

## Updating the Submodule

To update the submodule to the latest version from its remote repository:

```bash
# Navigate to the submodule directory
cd Aave-Simulator

# Fetch and checkout the latest changes
git fetch origin
git checkout main  # or the branch you want
git pull origin main

# Return to the parent repository
cd ..

# Commit the submodule update
git add Aave-Simulator
git commit -m "Update Aave-Simulator submodule"
```

Or update from the parent repository:

```bash
git submodule update --remote Aave-Simulator
```

## Working with the Submodule

### Making Changes to the Submodule

If you need to make changes to the Aave-Simulator code:

1. Navigate into the submodule directory:
   ```bash
   cd Aave-Simulator
   ```

2. Make your changes and commit them:
   ```bash
   git add .
   git commit -m "Your commit message"
   git push origin main
   ```

3. Return to the parent repository and update the submodule reference:
   ```bash
   cd ..
   git add Aave-Simulator
   git commit -m "Update Aave-Simulator submodule reference"
   git push origin main
   ```

### Checking Submodule Status

To check the status of the submodule:

```bash
git submodule status
```

This will show you:
- The commit hash the submodule is currently pointing to
- Any uncommitted changes in the submodule
- Whether the submodule is on a different branch than expected

### Switching Submodule Branches

To switch the submodule to a different branch:

```bash
cd Aave-Simulator
git checkout <branch-name>
cd ..
git add Aave-Simulator
git commit -m "Switch Aave-Simulator to <branch-name>"
```

## Submodule Configuration

The submodule configuration is stored in `.gitmodules`:

```
[submodule "Aave-Simulator"]
    path = Aave-Simulator
    url = git@github.com:brains-group/Aave-Simulator.git
```

## Troubleshooting

### Submodule appears empty after clone

If the submodule directory exists but appears empty:

```bash
git submodule update --init --recursive
```

### Submodule is on a detached HEAD

If you see a warning about detached HEAD state:

```bash
cd Aave-Simulator
git checkout main
cd ..
```

### Removing the submodule (if needed)

If you need to remove the submodule:

```bash
# Remove the submodule entry from .git/config
git submodule deinit -f Aave-Simulator

# Remove the submodule directory from .git/modules
rm -rf .git/modules/Aave-Simulator

# Remove the submodule entry from .gitmodules
git rm --cached Aave-Simulator

# Remove the submodule directory
rm -rf Aave-Simulator
```

## Why Use a Submodule?

Using a Git submodule allows us to:
- **Maintain code separation**: The Aave-Simulator and Aave-Action-Recommender remain as separate repositories
- **Version control**: Pin the Action Recommender to a specific version of the Simulator
- **Independent development**: Each repository can be developed and versioned independently
- **Selective updates**: Update the Simulator version only when needed

## Related Repositories

- **Aave-Simulator**: [git@github.com:brains-group/Aave-Simulator.git](git@github.com:brains-group/Aave-Simulator.git)
- **Aave-Action-Recommender**: [git@github.com:brains-group/Aave-Action-Recommender.git](git@github.com:brains-group/Aave-Action-Recommender.git)

