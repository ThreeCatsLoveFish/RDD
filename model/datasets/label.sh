# DFDC labeling script
cat test/labels.csv | sed -e 's/\(.*mp4\).*\([:digit:]\).*/\2 \1/g' | awk '{print (1-$1)" "$2}' > test/labels.txt
