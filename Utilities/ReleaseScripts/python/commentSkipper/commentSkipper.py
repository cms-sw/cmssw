__author__="Aurelija"
__date__ ="$2010-08-09 14.23.54$"

import os.path
import buildFileCommentSkipper
import cppCommentSkipper

cppCommentFiles = ['.h', '.c', '.cc', '.cxx']
buildfilesCommentFiles = ['buildfile', 'buildfile.xml']

def filter(fileList):

    if not fileList: return fileList
    head, tail = os.path.split(fileList[0])
    root, ext = os.path.splitext(tail)
    
    if (tail.lower() in buildfilesCommentFiles):
        fileList = buildFileCommentSkipper.filterFiles(fileList)
    elif (ext.lower() in cppCommentFiles):
        fileList = cppCommentSkipper.filterFiles(fileList)
    return fileList