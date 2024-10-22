from builtins import range
__author__="Aurelija"
__date__ ="$Jul 12, 2010 10:08:20 AM$"

import re

#fileList is list of files paths or could be a tuple of (path, [fileLines])
def finds(fileList, regularExpression, exceptRegEx = []):
    info = []
    lines = []

    for i in range(len(fileList)):
        if type(fileList[0]).__name__ != 'tuple':
            file = fileList[i]
            lines = find(fileList[i], regularExpression, exceptRegEx)
        else:
            file = fileList[i][0]
            lines = find(fileList[i][1], regularExpression, exceptRegEx)
            
        if lines:
            info.append((file, lines))
            
    return info

#file is the file path or the list of file lines
#exceptRegEx is the list of regular expressions that says to skip line if one of these regEx matches
def find(file, regularExpression, exceptRegEx = []):
    lines = []

    if type(file).__name__ != 'list':
        fileLines = open(file).readlines()
    else: fileLines = file
    for i in range(len(fileLines)):
        matchException = False
        if re.search(regularExpression, fileLines[i]) != None:
            for regEx in exceptRegEx:
                if re.search(regEx, fileLines[i]) != None:
                    matchException = True
                    break
            if not matchException:
                lines.append(i+1)
    return lines

