#!/bin/bash

cd /path/to/zip/file/folder
ls *.rar >> ls.log
data=$(cat ls.log)
for i in $data
  do
    unrar x $i & > /dev/null
done
