#!/bin/bash - 
#===============================================================================
#
#          FILE: rsync.sh
# 
#         USAGE: ./rsync.sh 
# 
#   DESCRIPTION: 
# 
#       OPTIONS: ---
#  REQUIREMENTS: ---
#          BUGS: ---
#         NOTES: ---
#        AUTHOR: YOUR NAME (), 
#  ORGANIZATION: 
#       CREATED: 30/04/2025 14:31
#      REVISION:  ---
#===============================================================================
set -o nounset                              # Treat unset variables as an error

PROJECT_ROOT="$MDIR_LINUX_DATA"/vc/papers_codes
PROJECT=2025-TC
echo "project root $PROJECT_ROOT"

if [ -z "$1" ]; then
    echo "Usage: $0 <SERVER> [SUBDIR]"
    exit 1
fi

SERVER=$1
SUBDIR=${2:-}  # Optional second argument containing the subpath
INVERT=${3:-}  # Optionally invert src and dst

SRC_PATH="$PROJECT_ROOT/$PROJECT"
DST_PATH="$SERVER":vc/
if [ -n "$SUBDIR" ]; then
    SRC_PATH="$SRC_PATH/$SUBDIR"
    DST_PATH="$DST_PATH/$SUBDIR"
fi

if [ -n "$INVERT" ]; then
    TMP=$SRC_PATH
    SRC_PATH=$DST_PATH
    DST_PATH=$TMP
fi

echo "syncing $SRC_PATH to $SERVER:vc/"

rsync -avz --info=progress2 \
      --filter="merge $PROJECT_ROOT/$PROJECT/rsync_filter.txt" \
      "$SRC_PATH" "$DST_PATH"


