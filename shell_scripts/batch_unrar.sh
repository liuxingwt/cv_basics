#!/bin/bash

cd /mnt/data/ly_zhijia/data/zip_file/data3_zip
ls *.rar >> ls.log
data=$(cat ls.log)
for i in $data
  do
    unrar x $i & > /dev/null
done
