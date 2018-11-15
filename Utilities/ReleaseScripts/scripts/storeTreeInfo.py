#!/usr/bin/env python

from __future__ import print_function
import os, sys, stat
from operator import itemgetter

class TreeAnalyzer(object):

    def __init__(self, outFileName):
        self.dirSizes  = {}
        self.fileSizes = {}
        self.outFileName = outFileName
        print("going to write to:",self.outFileName)

    def analyzePath(self, dirIn) :

        for (path, dirs, files) in os.walk(dirIn):

            if 'CVS' in path: continue
            if '.glimpse_' in path: continue
            if 'Configuration/PyReleaseValidation/data/run/' in path: continue

            for file in files:
                if '.glimpse_index' in file: continue
                fileName = os.path.join(path, file)
                fileSize = os.path.getsize(fileName)
                if path in self.dirSizes.keys() :
                    self.dirSizes[path] += fileSize
                else:
                    self.dirSizes[path] =  fileSize
                if os.path.isfile(fileName):
                    self.fileSizes[fileName] = fileSize

        try:
            import json
            jsonFileName = self.outFileName
            jsonFile = open(jsonFileName, 'w')
            json.dump([os.path.abspath(dirIn), self.dirSizes, self.fileSizes], jsonFile)
            jsonFile.close()
            print('treeInfo info  written to ', jsonFileName)
        except Exception as e:
            print("error writing json file:", str(e))

        try:
            import pickle
            pklFileName = self.outFileName.replace('.json','.pkl')
            pickle.dump([os.path.abspath(dirIn), self.dirSizes, self.fileSizes], open(pklFileName, 'w') )
            print('treeInfo info  written to ', pklFileName)
        except Exception as e:
            print("error writing pkl file:", str(e))

    def show(self):

        # for p,s in self.dirSizes.items():
        #     print p, s

        topDirs  = sorted(self.dirSizes.items() , key=itemgetter(1), reverse=True)
        topFiles = sorted(self.fileSizes.items(), key=itemgetter(1), reverse=True)

        emptyFiles = []
        for pair in topFiles:
            p, s = pair
            if s == 0:
                emptyFiles.append(p)
        print("found ",len(emptyFiles),"empty files. ")

        print("found ", len(self.dirSizes), 'directories, top 10 are:')
        for i in range(10):
            print(topDirs[i])

        print("found ", len(self.fileSizes), 'files, top 10 are:')
        for i in range(10):
            print(topFiles[i])


def main():

    import getopt

    try:
        opts, args = getopt.getopt(sys.argv[1:], "c:o:", ['checkDir=', 'outFile='])

        checkDir = '.'
        outFile  = None
        for opt, arg in opts :

            if opt in ('-c', "--checkDir", ):
                checkDir = arg

            if opt in ('-o', "--outFile", ):
                outFile = arg

        ta = TreeAnalyzer(outFile)
        ta.analyzePath(checkDir)
        ta.show()

    except getopt.GetoptError as e:
        print("unknown option", str(e))
        sys.exit(2)

if __name__ == '__main__':
    main()

