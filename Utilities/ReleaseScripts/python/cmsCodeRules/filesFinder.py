import config
__author__="Aurelija"
__date__ ="$2010-07-15 12.27.32$"

import re
from Utilities.ReleaseScripts.cmsCodeRules.config import exceptPathes
from os.path import join
from Utilities.ReleaseScripts.cmsCodeRules.pathToRegEx import pathesToRegEx, pathToRegEx

def getFilePathesFromWalk(osWalkResult, file, checkPath):

    listOfFiles = []

    file = pathToRegEx(file)

    for root, dirs, files in osWalkResult:
        for name in files:
            excepted = False
            fullPath = join(root,name)
            dir = fullPath.replace(join(checkPath, ""), '')
            for path in pathesToRegEx(exceptPathes):
                if re.match(path, dir):
                    excepted = True
                    break
            if not excepted and re.match(file, name):
                listOfFiles.append(fullPath)
    return listOfFiles
