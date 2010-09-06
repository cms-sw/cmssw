#!/usr/bin/env python

__author__="Aurelija"
__date__ ="$2010-07-14 16.48.55$"

from os.path import join, isdir
import os
import re
import sys
from Utilities.ReleaseScripts.cmsCodeRules.keyFinder import finds
from Utilities.ReleaseScripts.cmsCodeRules.filesFinder import getFilePathsFromWalk
from Utilities.ReleaseScripts.cmsCodeRules.pickleFileCreater import createPickleFile
from Utilities.ReleaseScripts.cmsCodeRules.config import Configuration, rulesNames, rulesDescription, helpMsg, checkPath, picklePath, txtPath, exceptPaths
from Utilities.ReleaseScripts.cmsCodeRules.pathToRegEx import pathToRegEx
from Utilities.ReleaseScripts.commentSkipper.commentSkipper import filter
from Utilities.ReleaseScripts.cmsCodeRules.showPage import run

configuration = Configuration
RULES = rulesNames
checkPath = checkPath
picklePath = picklePath
txtPath = txtPath

def splitPaths(listRule, pathHead):
    try:
        for i in range(len(listRule)):
            path, linesNumbers = listRule[i]
            listRule[i] = (path.replace(pathHead, '', 1), linesNumbers)
    except TypeError:
        print "Error: given wrong type of parameter in function splitPaths."
    return listRule

def runRules(ruleNumberList, directory):

    result = []
    osWalk = []
    
    for rule in ruleNumberList:
        if str(rule) not in RULES:
            print 'Error: wrong rule parameter. There is no such rule: "'+rule+'"\n\n' + rulesDescription
            print '\nWrite -h for help'
            return -1

    osWalk.extend(os.walk(directory))

    exceptPathsForEachRule = []
    for path in exceptPaths:
        exceptPathsForEachRule.append(join(checkPath, path))

    for ruleNr in ruleNumberList:
        files = []
        rule = str(ruleNr)
        rule = configuration[ruleNr]

        filesToMatch = rule['filesToMatch']
        exceptRuleLines = []
        exceptRulePaths = []
        for path in rule['exceptPaths']:
            try:
                file, line = path.split(":")
                exceptRuleLines.append((pathToRegEx(file), line))
            except ValueError:
                exceptRulePaths.append(pathToRegEx(path))
        for fileType in filesToMatch:
            fileList = getFilePathsFromWalk(osWalk, fileType, exceptPathsForEachRule)
# ------------------------------------------------------------------------------
            for path in exceptRulePaths:
                FileList = []
                for file in fileList:
                    File = file.replace(join(checkPath, ""), "")
                    if not re.match(path, File):
                        FileList.append(file)
                fileList = FileList
# ------------------------------------------------------------------------------
            filesLinesList = []
            if rule['skipComments'] == True:
                filesLinesList = filter(fileList)
            elif not exceptRuleLines:
                filesLinesList = fileList
# ------------------------------------------------------------------------------
            for Nr, fileLine in enumerate(exceptRuleLines):
                regEx, line = fileLine
                for index, file in enumerate(fileList):
                    File = file.replace(join(checkPath, ""), "")
                    if re.match(regEx, File):
                        if rule['skipComments'] == True or Nr > 0:
                            filesLinesList[index] = (filesLinesList[index][0], omitLine(filesLinesList[index][1], line))
                        else:
                            filesLinesList.append([file, omitLine(file, line)])
                    elif rule['skipComments'] == False:
                        filesLinesList.append((file, open(file).readlines()))
            files.extend(filesLinesList)
# ------------------------------------------------------------------------------
        listRule = finds(files, rule['filter'], rule['exceptFilter'])
        result.append((ruleNr, splitPaths(listRule, checkPath)))
    return result

def omitLine(file, line):
    try:
        if type(file).__name__ != 'list':
            fileLines = open(file).readlines()
        else: fileLines = file
        fileLines[int(line)-1] = ''
    except IndexError:
        print 'File = "' + file +'" has only ' + str(len(fileLines)) + ' lines. Wrong given line number: ' + str(line)
    return fileLines

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
                checkPath = sys.argv[i]
                if not isdir(checkPath):
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
                if not isdir(picklePath):
                    goodParameters = False
                    print 'Error: wrong directory "%s"' %picklePath
                    break
        elif (arg == '-s'):
            createTxt = True
            if i+1 < argvLen and sys.argv[i+1][0] != '-':
                i+=1
                txtPath = sys.argv[i]
                if not isdir(txtPath):
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
                if argvLen == 1 or printResult or (createPickle == False and createTxt == False):
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
