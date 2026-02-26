#!/bin/bash

BRANCH=$1

if [ -z "$BRANCH" ]; then
    echo "Usage: gwktr <branch-name>"
    exit 1
fi

# We'll put worktrees one level up in a 'worktrees' folder
TARGET_DIR="../worktrees/$BRANCH"

# Check if branch exists locally or remotely
if git show-ref --verify --quiet "refs/heads/$BRANCH" || git show-ref --verify --quiet "refs/remotes/origin/$BRANCH"; then
    echo "Adding worktree with branch $BRANCH..."
    git worktree add "$TARGET_DIR" "$BRANCH"
else
    echo "Branch '$BRANCH' not found. Creating new branch and worktree..."
    # -b creates a new branch starting from your current HEAD
    git worktree add -b "$BRANCH" "$TARGET_DIR"
fi

# Print the path for your shell function to 'cd' into
echo "cd to $TARGET_DIR"
