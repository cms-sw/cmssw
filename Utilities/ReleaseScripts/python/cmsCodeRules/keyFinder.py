__author__="Aurelija"
__date__ ="$Jul 12, 2010 10:08:20 AM$"

import re

#fileList is list of files pathes or could be a tuple of [path, [filesLines]]
def finds(fileList, regularExpression):
    info = []
    lines = []

    for i in range(len(fileList)):
        if type(fileList[0]).__name__ != 'tuple':
            file = fileList[i]
            lines = find(fileList[i], regularExpression)
        else:
            file = fileList[i][0]
            lines = find(fileList[i][1], regularExpression)
            
        if lines:
            info.append((file, lines))
            
    return info

#file is the file path or the list of file lines
def find(file, regularExpression):
    lines = []

    if type(file).__name__ != 'list':
        fileLines = open(file).readlines()
    else: fileLines = file
    for i in range(len(fileLines)):
        if re.search(regularExpression, fileLines[i]) != None:
            lines.append(i+1)
    return lines

