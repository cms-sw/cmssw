from builtins import range
__author__="Aurelija"
__date__ ="$2010-07-13 12.17.20$"

import string

def filterFiles(fileList):
    files = []

    for file in fileList:
        files.append((file, filterFile(file)))
    return files

def filterFile(file): #ifstream& input)

    lines = open(file).readlines()
    commentStage = False

    for i in range(len(lines)):
        #for j in range(len(lines[i])):
        j = 0
        
        while lines[i][j] != '\n':
            
            char = lines[i][j]
            #char /
            if char == '/':
                #comment /*
                if lines[i][j+1] == '*' and not commentStage: #why !commentStage? because it could be /*...../*.....*/
                    commentStage = True
                    commentStartLine, commentStartColumn = i, j
                    j += 1
                #comment //  why !commentStage? because, can be a variant of this example: /*....//.....*/
                elif not commentStage and (lines[i][j+1] == '/'):
                    lines[i] = string.replace(lines[i], lines[i][j:],'\n', 1)
                    break
            #char "
            elif char == '"':
                if not commentStage:
                    next = string.find(lines[i][j+1:], '"') #next "
                    lines[i] = string.replace(lines[i], lines[i][j:j+next+2], '', 1) # clear string in ""
            #char '
            elif char == '\'':
                if not commentStage:
                    next = string.find(lines[i][j+1:], '\'') #next '
                    lines[i] = string.replace(lines[i], lines[i][j:j+next+2], '', 1) # clear string in ''
            #char *
            elif char == '*':
                if (commentStage and (lines[i][j+1] == '/')):
                    commentStage = False;
                    if commentStartLine != i:
                        lines[i] = string.replace(lines[i], lines[i][:j+2],'', 1) # comment */ [..code]
                        j = -1 #because of j+=1 at the end
                    else:
                        lines[i] = string.replace(lines[i], lines[i][commentStartColumn:j+2], '', 1) # [code..] /*comment*/ [.. code]
                        j = commentStartColumn - 1 #because of j+=1 at the ends
            if j != len(lines[i]) - 1:
                j += 1
            else:
                j = 0
                break
        if commentStage:
            if i == commentStartLine: lines[i] = lines[i].replace(lines[i][commentStartColumn:],'\n', 1)
            else: lines[i] = lines[i].replace(lines[i][:], '\n')
    return lines