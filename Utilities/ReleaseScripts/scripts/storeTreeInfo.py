#!/usr/bin/env python

import os, sys, stat
from operator import itemgetter

class TreeAnalyzer(object):

    def __init__(self):
        self.dirSizes  = {}
        self.fileSizes = {}
    
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
            jsonFileName = '/tmp/treeInfo-IBsrc.json'
            jsonFile = open(jsonFileName, 'w')
            json.dump([os.path.abspath(dirIn), self.dirSizes, self.fileSizes], jsonFile)
            jsonFile.close()
            print 'treeInfo info  written to ', jsonFileName
        except Exception, e:
            print "error writing json file:", str(e)
        
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
        print "found ",len(emptyFiles),"empty files. "

        print "found ", len(self.dirSizes.keys()), 'directories, top 10 are:'
        for i in range(10):
            print topDirs[i]

        print "found ", len(self.fileSizes.keys()), 'files, top 10 are:'
        for i in range(10):
            print topFiles[i]


def main():

    ta = TreeAnalyzer()
    ta.analyzePath('.')
    ta.show()

if __name__ == '__main__':
    main()

    
