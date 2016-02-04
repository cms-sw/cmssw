__author__="Aurelija"
__date__ ="$2010-07-20 11.52.22$"

import sys

def filterFiles(fileList):
    files = []

    for file in fileList:
        files.append((file, filterFile(file)))
    return files


def filterFile(file):
    lines = open(file).readlines()
    lines = filterMultilineComment(lines, '"""', '"""')
    lines = filterMultilineComment(lines, "'''", "'''")
    lines = filterMultilineComment(lines, '<!--', '-->')
    lines = filterOneLineComment(lines, "#")
    return lines


def filterOneLineComment(lines, commentStart):
    for i in range(len(lines)):
        index = lines[i].find(commentStart)
        if index != -1:
            lines[i] = lines[i].replace(lines[i][index:], '\n')
    return lines


def filterMultilineComment(lines, commentStart, commentEnd):
    i = 0
    tlines = len(lines)
    while (i < tlines):
        startIndex = lines[i].find(commentStart)
        startLine = i
        while((startIndex != -1) and (i < tlines)):
            endIndex = lines[i].find(commentEnd)
            if endIndex != -1:
                if startLine == i:
                    lines[i] = lines[i].replace(lines[i][startIndex:endIndex+3], '', 1)
                else:
                    lines[i] = lines[i].replace(lines[i][:endIndex+3], '')
                startIndex = lines[i].find(commentStart)
                startLine = i
            else:
                if startLine == i:
                    lines[i] = lines[i].replace(lines[i][startIndex:], '\n', 1)
                else:
                    lines[i] = lines[i].replace(lines[i][:], '\n')
                i += 1
        i += 1
    return lines
