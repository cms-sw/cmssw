#! /usr/bin/env python
import hashlib
hashes = set()

f = open('classes.txt')

for line in f :
	fields = line.split("'")
	if fields[0] == 'class ' :
		hash = hashlib.sha1(fields[1])
		hashes.add(hash.hexdigest())
	
f.close()
print "static const std::vector<std::string> dataClassNameHashes = { \"00000000\""
for h in sorted(hashes) : print ", \""+h[:8]+"\""
print "};"

