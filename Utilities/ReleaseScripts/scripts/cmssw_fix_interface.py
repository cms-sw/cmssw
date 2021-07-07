#!/usr/bin/env python3

import os
import sys
import argparse


parser = argparse.ArgumentParser(description='Find includes only used in one non-interface directory.')
parser.add_argument('packageName',
                   help='name of package to check interface usage')
parser.add_argument('--fix', dest='shouldFix', action='store_true',
                    help='move file and fix includes if only used in 1 directory')
parser.add_argument('--remove', dest='removeUnused', action='store_true',
                    help='remove interface files that are not included anywhere')

args = parser.parse_args()

packageName = args.packageName
shouldFix = args.shouldFix
removeUnused = args.removeUnused

interfaceDir = packageName+"/interface"
from os.path import isfile, join
onlyfiles = [join(interfaceDir,f) for f in os.listdir(interfaceDir) if isfile(join(interfaceDir, f))]

for f in onlyfiles:
    print("checking {filename}".format(filename=f))
    result = os.popen('git grep \'#include [",<]{filename}[",>]\' | awk -F\':\' \'{{print $1}}\' | sort -u'.format(filename=f))

    filesUsing = [l[:-1] for l in result]

    if 0 == len(filesUsing):
        print("  "+f+" is unused")
        if removeUnused:
            os.system('git rm {filename}'.format(filename=f))
            print("   "+f+" was removed")
        continue
    
    #directories using
    dirs = set( ( "/".join(name.split("/")[0:3]) for name in filesUsing) )
    if 1 == len(dirs):
        onlyDir = dirs.pop()
        if onlyDir.split("/")[2] != "interface":
            print("  "+f+" is only used in "+onlyDir)
            if shouldFix:
                newFileName = onlyDir+"/"+f.split("/")[3]
                mvCommand = "git mv {oldName} {newName}".format(oldName=f, newName=newFileName)
                #print(mvCommand)
                os.system(mvCommand)
                sedCommand ="sed --in-place 's/{oldName}/{newName}/' {filesToChange}".format(oldName="\/".join(f.split("/")),newName="\/".join(newFileName.split("/")), filesToChange=" ".join( (n for n in filesUsing)) )
                #print(sedCommand)
                os.system(sedCommand)

