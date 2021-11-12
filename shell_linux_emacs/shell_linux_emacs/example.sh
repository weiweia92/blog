#!/bin/bash

echo "Starting program at $(date)" #Date will be substituted

echo "Running program $0 with $# arguments with pid $$"
#$# is the number of arguments that we are giving to the command
#$$ is the process ID of the command that is running

for file in "$@"; do
    #$@:all arguments
    grep foobar "$file" > /dev/null 2> /dev/null
    #When pattern is not found, grep has exit status
    #We redirect STDOUT and STDERR to a null register about them
    if [[ "$?" -ne 0 ]]; then
	echo "File $file does not have any foobar, adding one"
	echo "# foobar" >> "$file"
    fi
done
