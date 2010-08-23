import re
__author__="Aurelija"
__date__ ="$2010-07-15 12.27.32$"

import os
from os.path import join

# files - list of files which want to search. For example: *.h, buildFile.* etc

def getFilePathes(directory, file):

    listOfFiles = []

    if not os.path.exists(directory):
        print 'Wrong directory: "%s"' %directory
        return -1

    file = "\A%s$" %file
    file = file.replace(".", "\.")
    file = file.replace("*",".*")

    for root, dirs, files in os.walk(directory):
            for name in files:
                if re.match(file, name):
                    listOfFiles.append(join(root,name))

    return listOfFiles

def getFilePathesFromWalk(osWalkResult, file):

    listOfFiles = []

    file = "\A%s$" %file
    file = file.replace(".", "\.")
    file = file.replace("*",".*")

    for root, dirs, files in osWalkResult:
            for name in files:
                if re.match(file, name):
                    listOfFiles.append(join(root,name))

    return listOfFiles

#    os.chdir(directory)#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!to do
#
#    if sys.platform[:3] == 'win':
#        findcmd = 'dir "%s" /B /S' %file
#        listOfFiles = list(os.popen(findcmd).readlines())
#        for i in range(len(listOfFiles)):
#            listOfFiles[i] = listOfFiles[i][:-1]  # the last letter is \n. We don't need it
#    else:
#        findcmd = 'find . -iname "%s"' %file
#        listOfFiles = list(os.popen(findcmd).readlines())
#        for i in range(len(listOfFiles)):
#            listOfFiles[i] = directory + listOfFiles[i][1:-1] # the first letter in file is . and the last is \n. We don't need them
#    return listOfFiles

def getFilesPathes(directory, files):
    fileList = []

    for file in files:
        if (type(directory) == str):
            filePathes = getFilePathes(directory, file)
        else:
            filePathes = getFilePathesFromWalk(directory, file)
        if filePathes == -1: return fileList
        fileList.extend(filePathes)

    return fileList

def getFilesPathesFromDirectories(directories, files):
    fileList = []

    for directory in directories:
        fileList.extend(getFilesPathes(directory, files))
        
    return fileList
