#!/bin/bash

# Check if input is provided
if [ -z "$1" ]; then
    echo "Usage: ./make_cqp.sh \"Your sentence here\""
    exit 1
fi

# 1. Use sed to put a space around common punctuation marks
# 2. Use xargs to trim extra whitespace
# 3. Use awk to wrap every word in [word="... " %c]
echo "$1" | \
    sed -E 's/([[:punct:]])/ \1 /g' | \
    xargs | \
    awk '{
        for (i=1; i<=NF; i++) {
            # Escape backslashes for CQP (especially for the period)
            gsub(/\\/, "\\\\", $i);
            printf "[word=\"%s\" %%c] ", $i
        }
        print ""
    }'
