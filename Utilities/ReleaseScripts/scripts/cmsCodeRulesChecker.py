#!/usr/bin/env python

__author__="Aurelija"
__date__ ="$2010-07-14 16.48.55$"

import os
import os.path
import sys
from Utilities.ReleaseScripts.cmsCodeRules.keyFinder import finds
from Utilities.ReleaseScripts.cmsCodeRules.filesFinder import getFilesPathes
from Utilities.ReleaseScripts.cmsCodeRules.pickleFileCreater import createPickleFile
import Utilities.ReleaseScripts.cmsCodeRules.config 
from Utilities.ReleaseScripts.cmsCodeRules.config import Configuration, rulesNames, rulesDescription, helpMsg
from Utilities.ReleaseScripts.commentSkipper.commentSkipper import filter
from Utilities.ReleaseScripts.cmsCodeRules.showPage import run

configuration = Configuration
RULES = rulesNames
checkPath = [Utilities.ReleaseScripts.cmsCodeRules.config.checkPath]
picklePath = Utilities.ReleaseScripts.cmsCodeRules.config.picklePath
txtPath = Utilities.ReleaseScripts.cmsCodeRules.config.txtPath

def splitPathes(listRule, pathHead):
    try:
        for i in range(len(listRule)):
            path, linesNumbers = listRule[i]
            listRule[i] = (path.replace(pathHead, '', 1), linesNumbers)
    except TypeError:
        print "Error: given wrong type of parameter in function splitPathes."
    return listRule

def runRules(ruleNumberList, directoryList):

    result = []
    osWalk = []
    
    for rule in ruleNumberList:
        if str(rule) not in RULES:
            print 'Error: wrong rule parameter. There is no such rule: "'+rule+'"\n\n' + rulesDescription
            print '\nWrite -h for help'
            return -1

    for directory in directoryList:
        osWalk.extend(os.walk(directory))

    for ruleNr in ruleNumberList:
        files = []
        rule = str(ruleNr)
        rule = configuration[ruleNr]


        filesToMatch = rule['filesToMatch']
        for fileType in filesToMatch:
            #fileList = filesFinder.getFilesPathesFromDirectories(directoryList, [fileType])
            fileList = getFilesPathes(osWalk, [fileType])#
            if rule['skipComments'] == True:
                fileList = filter(fileList)
            files.extend(fileList)
        listRule = finds(files, rule['filter'])
        result.append((ruleNr, splitPathes(listRule, checkPath[0])))
    return result

def printOut(listOfResult, filePath = None):
    file = None
    try:
        for rule, result in listOfResult:
            if filePath:
                file = open('%s/cmsCodeRule%s.txt'%(filePath, rule), 'w')
                for path, lineNumbers in result:
                    file.write('%s\n'%path)
                    file.write('%s\n'%str(lineNumbers))
                file.close()
            else:
                if not result or result == -1:
                    print 'No results for rule %s'%rule
                else:
                    print 'Rule %s:' %rule
                    for path, lineNumbers in result:
                        print path
                        print lineNumbers
    except TypeError:
        print "Error: wrong type of parameter in function printOut"

if __name__ == "__main__":
    dict = {}

    cwd = os.getcwd()
    
    createPickle = False
    createTxt = False
    html = False
    help = False
    rules = False
    
    goodParameters = True
    argvLen = len(sys.argv)
    printResult = False
    ruleNumbers = RULES

    i = 1
    while (i != argvLen):
        arg = sys.argv[i]

        if   (arg == '-h'):
            help = True
        elif (arg == '-r'):
            rules = True
            i+=1
            if i < argvLen:
                ruleNumbers = sys.argv[i].split(',')
            else:
                goodParameters = False
                print 'Error: missing rule parameters. Write -h for help'
                break
        elif (arg == '-d'):
            i+=1
            if i < argvLen:
                checkPath = [sys.argv[i]]
                if not os.path.isdir(checkPath[0]):
                    goodParameters = False
                    print 'Error: wrong directory "%s"' %checkPath
                    break
            else:
                goodParameters = False
                print 'Error: missing rule parameters. Write -h for help'
                break
        elif (arg == '-S'):
            createPickle = True
            if i+1 < argvLen and sys.argv[i+1][0] != '-':
                i+=1
                picklePath = sys.argv[i]
                if not os.path.isdir(picklePath):
                    goodParameters = False
                    print 'Error: wrong directory "%s"' %picklePath
                    break                
        elif (arg == '-s'):
            createTxt = True
            if i+1 < argvLen and sys.argv[i+1][0] != '-':
                i+=1
                txtPath = sys.argv[i]
                if not os.path.isdir(txtPath):
                    goodParameters = False
                    print 'Error: wrong directory "%s"' %txtPath
                    break
        elif (arg == '-html'):
            html = True
            createPickle = True
        elif (arg == '-p'):
            printResult = True
        else:
            goodParameters = False
            print 'Error: there is no parameter like "%s". Write -h for help' %arg
            break
        i+=1

    if goodParameters == True:

        if argvLen == 2 and help == True:
            print helpMsg
        else:
            result = runRules(ruleNumbers, checkPath)
                    
            if result != -1:
                if len(sys.argv) == 1 or printResult or (createPickle == False and createTxt == False):
                    printOut(result)
                else:
                    if createPickle:
                        for rule, ruleResult in result:
                            createPickleFile('cmsCodeRule%s.dat'%rule, ruleResult, picklePath)
                    if createTxt:
                        printOut(result, txtPath)
            if html:
                run(picklePath, picklePath, picklePath)
            if help:
                print helpMsg
