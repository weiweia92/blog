#!/bin/bash

#create a mcd function, $1:first argument
mcd() {
    mkdir -p "$1"
    cd "$1"
}
# foobar
