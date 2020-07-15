# Sorts a geometry XML file to allow for easier comparison
# $1 -- input XML file name. $2 -- output file name.

sed -e '1 d' -e '2 d' -e '/TRACKER/,$ d' $1 > /tmp/temp$$.xml
./sortCompositeMaterials.py /tmp/temp$$.xml /tmp/sortTemp$$.xml >& /dev/null
sed -e 's/ns0://g' -e 's:" /:"/:g' /tmp/sortTemp$$.xml > $2
rm /tmp/temp$$.xml /tmp/sortTemp$$.xml
