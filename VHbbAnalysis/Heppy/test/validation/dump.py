import os
import math

json = open('CutBasedID_TightWP.json', 'w')
inp = open('CutBasedID_TightWP.txt', 'r')

ptbins = []

json.write('{\n')
json.write('    \"CutBasedID_TightWP\" : {\n')
json.write('        \"eta_pt_ratio\" : {\n')
for line in inp.readlines():
     split = line.split()
     #print split1
     ptbin = [split[0], split[1]]
     if ptbin not in ptbins:
         ptbins.append(ptbin)
print ptbins

inp.close()

for ptbin in ptbins:    
    json.write('            "eta:[' + ptbin[0] + ','+ ptbin[1] + ']\": {\n')
    inp = open('CutBasedID_TightWP.txt', 'r')

    first = True
    for line in inp.readlines():   
        split = line.split()
        #print split
        bin1 = [split[0], split[1]]
        if bin1 != ptbin :
            continue
        #print bin1
        #if not first:
        #     json.write(',\n')
        #     first = False
        sf = str( float(split[4])/float(split[6]) )
        err = str( math.sqrt( float(split[5])/float(split[4])*float(split[5])/float(split[4]) + 
                         float(split[7])/float(split[6])*float(split[7])/float(split[6]) ) )
        bin2 = [split[2], split[3], sf, err]
        json.write('                \"pt:[' + bin2[0] + ','+bin2[1] + ']\": {\n')
        json.write('                    \"value\": '+ bin2[2]+",\n")   
        json.write('                    \"error\": '+ bin2[3]+"\n")           
        json.write('                },\n')
    #json.write('\n')

    inp.close()
    json.write('            },\n')

json.write('        }\n')
json.write('    }\n')
json.write('}\n')

json.close()
inp.close()
