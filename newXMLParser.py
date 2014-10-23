#!/usr/bin/env python 
import sys

line_replacer = "xa=\"0.0000000\" xb=\"0.0000000\" xc=\"0.0000000\" ya=\"0.0000000\" yb=\"0.0000000\" yc=\"0.0000000\" za=\"0.0000000\" zb=\"0.0000000\" zc=\"0.0000000\" aa=\"0.0000000\" ab=\"0.0000000\" ac=\"0.0000000\" bb=\"0.0000000\" bc=\"0.0000000\" cc=\"0.0000000\" />\n"
#line_replacer = "    <setposition relativeto=\"ideal\" x=\"10.0000000\" y=\"10.0000000\" z=\"10.0000000\" a=\"10.0000000\" b=\"10.0000000\" c=\"10.0000000\" />\n"

input_file = open(str(sys.argv[1]),'r')
output_file = open(str(sys.argv[2]),'w')

for line in input_file:
    if line.find('setape') != -1:
        #print "FOUND ", line
        line_r = line[:-3] + line_replacer
        output_file.write(line_r)        
    else: 
        output_file.write(line)   

output_file.close()
