#!/bin/bash

# Get the weights zip
wget --no-check-certificate "https://onedrive.live.com/download?cid=78E59E7A2C353C95&resid=78E59E7A2C353C95%218858&authkey=ADujCI0PG_Z0hTc"
filename=$(ls | grep -v ".sh")
unzip $filename
mv models/* .
rm -r models
rm $filename


