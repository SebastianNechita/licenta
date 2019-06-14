#!/bin/bash

cat $1 | awk -f ./awkScript2.awk | sed 's/\(\.[^ 0]*\)0\+ /\1 /g' | sed 's/\. / /g'
