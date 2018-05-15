__author__="Aurelija"
__date__ ="$2010-08-13 14.28.08$"

import sys
import glob
import pickle
from os.path import join, split

rulesNames = []

if sys.platform[:3] == 'win':
    slash = "\\"
else:
    slash = "/"

def readPicFiles(directory, toSplit = False):

    ruleResult = {}
    rulesResults = {}
    picFiles = sorted(glob.glob(join(directory, "cmsCodeRule*.dat")))

    for file in picFiles:
        head, fileName = split(file)
        ruleName = fileName[11:-4]
        rulesNames.append(ruleName)

        file = open(file)
        ruleResult = pickle.load(file)
        if toSplit:
            ruleResult = splitToPackages(ruleResult)

        rulesResults[ruleName] = ruleResult

    return rulesResults


def splitToPackages(ruleResult):

    packageResult = []
    info = []

    if not ruleResult: return info

    ruleResult = sorted(ruleResult.items())
    file, lines = ruleResult.pop(0)
    pathList = pathToList(file)
    package = slash.join(pathList[:2])
    packageResult.append((slash.join(pathList[2:]), lines))

    for file, lines in ruleResult:
        pathList = pathToList(file)
        head = slash.join(pathList[:2])
        tail = slash.join(pathList[2:])
        if package == head:
            packageResult.append((tail, lines))
        else:
            info.append((package, packageResult))
            packageResult = []
            package = head
            packageResult.append((tail, lines))
    info.append((package, packageResult))
    return info #list of (package, packageResult)

def pathToList(path):
    list = []
    head, tail = split(path)
    if tail != '':
        list.insert(0, tail)
    while head != '':
        head, tail = split(head)
        if tail != '':
            list.insert(0, tail)
        else:
            break
    return list

