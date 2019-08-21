#!/usr/bin/env bash

# run this from within the cosmobot-deep-learning directory to rsync it into deepthought:/home/osmo/<username>

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
USAGE_MESSAGE="Usage: ./sync_to_deepthought.sh <username>"
USER_DIRECTORY="/home/osmo/$1"

if [ $# -lt 1 ]; then
  echo 1>&2 "$0: not enough arguments. $USAGE_MESSAGE"
  exit 2
elif [ $# -gt 1 ]; then
  echo 1>&2 "$0: too many arguments. $USAGE_MESSAGE"
  exit 2
fi

# to avoid creating a new directory in case of a typo, only sync if the username directory already exists
if ! ssh osmo@deepthought '[ -d $USER_DIRECTORY ]'; then
  echo 1>&2 "$0: username directory $USER_DIRECTORY does not exist on deepthought."
  exit 2
fi

# assumes you have key set up, and deepthought hostname in hosts file
rsync -avz --exclude ".git" --exclude "*.swp" --exclude "sync_to_deepthought.sh" --exclude-from $DIR/.gitignore $DIR osmo@deepthought:$USER_DIRECTORY

# clean up pyc files on deepthought just in case
ssh osmo@deepthought 'find $USER_DIRECTORY -name "*.pyc" -delete'
