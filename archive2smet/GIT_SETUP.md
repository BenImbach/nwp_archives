# Git Setup Instructions

Follow these commands to initialize git and push to Bitbucket:

## 1. Navigate to the archive2smet directory
```bash
cd /path/to/archive2smet
```

## 2. Configure git for Windows filesystem (if on WSL/Windows mount)
If you're on a Windows-mounted filesystem (/mnt/...), configure git first:
```bash
git config core.filemode false
git config core.autocrlf input
```

## 3. Initialize git repository
```bash
git init
```

## 4. Add all files (respecting .gitignore)
```bash
git add .
```

## 5. Create initial commit
```bash
git commit -m "Initial commit: archive2smet pipeline for converting NWP archives to SMET format"
```

## 6. Create the repository on Bitbucket first
- Go to https://bitbucket.org/sfu-arp
- Click "Create repository"
- Name it `archive2smet` (or your preferred name)
- Do NOT initialize with README, .gitignore, or license (we already have these)
- Copy the repository URL

## 7. Add the remote and push

**If you get permission errors on Windows filesystem**, manually edit `.git/config` and add:
```ini
[remote "origin"]
	url = https://bitbucket.org/sfu-arp/nwp_archives.git
	fetch = +refs/heads/*:refs/remotes/origin/*
```

**Otherwise**, use the command:
```bash
# Add remote (replace 'nwp_archives' with your actual repo name if different)
git remote add origin https://bitbucket.org/sfu-arp/nwp_archives.git
```

**Then push to Bitbucket:**
```bash
# Rename branch to main (if needed)
git branch -M main

# Push to Bitbucket (you'll need authentication - see below)
git push -u origin main
```

## Authentication Required

When you run `git push`, Bitbucket will require authentication. You have two options:

### Option 1: App Password (Recommended)
1. Go to: https://support.atlassian.com/bitbucket-cloud/docs/app-passwords/
2. Create an app password with repository read/write permissions
3. When prompted for password during `git push`, use the app password (not your regular password)

### Option 2: SSH
If you prefer SSH, change the remote URL:
```bash
git remote set-url origin git@bitbucket.org:sfu-arp/nwp_archives.git
```
Then set up SSH keys in your Bitbucket account settings.

## Future updates
After making changes:
```bash
git add .
git commit -m "Description of changes"
git push
```

