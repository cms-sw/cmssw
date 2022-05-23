from __future__ import print_function
from builtins import range
__author__="Aurelija"
__date__ ="$2010-09-23 15.00.20$"

import re

declarator = '(\*|&)?(\w|<|,|>|$|::)+'
cv_decl = '\s*(const|volatile|noexcept)\s*'
exception = 'throw\(((::)|\w|\s|,|<|>)*\)'
decl_param = '\s((\(%s\))|(%s))\s*\((\w|\s|\*|&|\.|=|\'|\"|-|<|>|,|(::))*\)'%(declarator, declarator)
operator = '(%s|)operator\s*(\(\)|\[\]|\s+(new|delete)(\s*\[\]|)|\-\>[*]{0,1}|[+\-*/%%^&|~!=<>,]{1,2}(=|))'%(declarator)
dm_init = '(:[^{]*)'
functStart_re = re.compile('(\s|~|^)((\(%s\))|(%s)|(%s))\s*\((%s|\w|\s|\*|&|\.|=|\'|\"|-|<|>|,|::)*\)(%s)?(%s)?\s*(%s)?\s*{'%(declarator, declarator, operator, decl_param, cv_decl, exception,dm_init), re.MULTILINE)

def filterFiles(fileList):
    files = []

    for i in range(len(fileList)):
        if type(fileList[0]).__name__ != 'tuple':
            file = fileList[i]
            fileLines = filterFile(fileList[i])
        else:
            file = fileList[i][0]
            fileLines = filterFile(fileList[i][1])

        files.append((file, fileLines))
    return files

def filterFile(file):

    lines = ""

    if type(file).__name__ != 'list':
        lines = open(file, errors='replace').read()
    else:
        for line in file:
            lines += line
    fileLines = lines[:]
    prevEndPosition = 0
    endPosition = 0
    while(True):
        m = functStart_re.search(lines)
        if m != None:
            openBracket = 1
            closeBracket = 0
            startPosition = m.start()
            #print "MATCH: " + lines[m.start():m.end()]
            for i, character in enumerate(lines[m.end():]):
                if character == "{":
                    openBracket += 1
                elif character == "}":
                    closeBracket += 1
                    if openBracket == closeBracket :
                        prevEndPosition += endPosition
                        endPosition = m.end() + i + 1
                        break
            if openBracket != closeBracket:#if there is error in file
                print("Error in file. To much open brackets. Run commentSkipper before you run functionSkipper.")
                break
            else:
                #print "LINES: \n" + lines[startPosition:endPosition] 
                #print "#############################################";
                lines = delLines(lines, startPosition, endPosition)
                fileLines = fileLines[:prevEndPosition] + lines
                lines = lines[endPosition:]
        else:
            break

    listOfLines = []
    startLine = 0
    for index, character in enumerate(fileLines):
        if character == "\n":
            listOfLines.append(fileLines[startLine:index+1])
            startLine = index + 1
    listOfLines.append(fileLines[startLine:])
    return listOfLines

def delLines(string, startPosition, endPosition):
    i = startPosition - 1
    end = startPosition
    while(True):
        if i != -1 and string[i] != '\n':
            i -= 1
        else:
            string = string[:i+1] + (end - i - 1)*' ' + string[end:]
            break

    i = startPosition
    start = startPosition
    while(i != endPosition):
        if string[i] != '\n':
            i += 1
        else:
            string = string[:start] + (i-start)*str(" ") + string[i:]
            i = i+1
            start = i
    string = string[:start] + (endPosition-start)*str(" ") + string[endPosition:]

    return string
