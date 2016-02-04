#!/usr/bin/env python

from optparse import OptionParser
import sys,os, re, pprint

# this set of function should be put in a separate module.

def allCastorFiles( castorDir, regexp ):

    try:
        pattern = re.compile( regexp )
    except:
        print 'please enter a valid regular expression '
        sys.exit(1)

    allFiles = os.popen("rfdir %s | awk '{print $9}'" % (castorDir))

    matchingFiles = []
    for file in allFiles.readlines():
        file = file.rstrip()
        
        m = pattern.match( file )
        if m:
            fullCastorFile = 'rfio:%s/%s' % (castorDir, file)
            matchingFiles.append( fullCastorFile )

    allFiles.close()

    return matchingFiles


def cleanFiles( castorDir, regexp, tolerance):

    try:
        pattern = re.compile( regexp )
    except:
        print 'please enter a valid regular expression '
        sys.exit(1)

    allFiles = os.popen("rfdir %s | awk '{print $9}'" % (castorDir))
    sizes = os.popen("rfdir %s | awk '{print $5}'" % (castorDir))

    averageSize = 0
    count = 0.

    matchingFiles = []
    print 'Matching files: '
    for file,size in zip( allFiles.readlines(), sizes.readlines()):
        file = file.rstrip()
        size = float(size.rstrip())

        m = pattern.match( file )
        if m:
            print file
            fullCastorFile = '%s/%s' % (castorDir, file)
            matchingFiles.append( (fullCastorFile, size) )
            averageSize += size
            count += 1

    averageSize /= count
    print 'average file size = ',averageSize

    cleanFiles = []
    dirtyFiles = []

    for file, size in matchingFiles:
        relDiff = (averageSize - size) / averageSize
        if relDiff < tolerance:
            # ok
            # print file, size, relDiff
            cleanFiles.append( file )
        else:
            print 'skipping', file, ': size too small: ', size, relDiff
            dirtyFiles.append( file )

    return (cleanFiles, dirtyFiles)



# returns an integer
def fileIndex( regexp, file ):

    try:
        numPattern = re.compile( regexp )
    except:
        print 'fileIndex: please enter a valid regular expression '
        sys.exit(1)

    m = numPattern.search( file )
    if m:
        return int(m.group(1))
    else:
        print file, ': cannot find number.'
        return -1


def extractNumberAndSort( regexp, files ):
    numAndFile = []
    for file in files:
        num = fileIndex( regexp, file )
        if num>-1:
            numAndFile.append( (num, file) )

    numAndFile.sort()

    return numAndFile 


def sync( files1, files2):

    regexp = '_(\d+)\.root'

    numAndFile1 = extractNumberAndSort( regexp, files1 )
    numAndFile2 = extractNumberAndSort( regexp, files2 )

#    pprint.pprint(numAndFile1) 
#    pprint.pprint(numAndFile2) 
   
    i1 = 0
    i2 = 0

    single = [] 
    while i1<len(numAndFile1) and i2<len(numAndFile2):

        (n1, f1) = numAndFile1[i1]
        (n2, f2) = numAndFile2[i2]

        #        print f1
        #        print f2
        # print 'nums: ', n1, n2, 'index: ', i1, i2
        
        if n1<n2:
            print 'single: ', f1
            single.append(f1)
            i1 += 1
        elif n2<n1:
                print 'single: ', f2
                single.append(f2)
                i2 += 1
        else:
            i1 += 1
            i2 += 1
    return single

def createTrash( trash ):
    absName = '%s/%s' % (castorDir, trash)
    out = os.system( 'rfdir %s' % absName )
    print out
    if out!=0:
        # dir does not exist
        os.system( 'rfmkdir %s' % absName )
    return absName

def remove( absTrash, files ):

    for file in files:
        baseName = os.path.basename(file)
        rfrename = 'rfrename %s %s/%s' % (file, absTrash, baseName)
        #print rfrename
        os.system( rfrename )
        

# main
    
parser = OptionParser()
parser.usage = "%prog <castor dir> <regexp pattern 1> <regexp pattern 2>\nPuts in sync the root files corresponding to different outputs of the same process. By default, the single files are detected, but are not removed"
parser.add_option("-d", "--remove-dirty", action="store_true",
                  dest="removeDirty",
                  help="Remove dirty files",
                  default=False)
parser.add_option("-s", "--remove-single", action="store_true",
                  dest="removeSingle",
                  help="Remove single files",
                  default=False)
parser.add_option("-t", "--tolerance",  
                  dest="cleanTolerance",
                  help="relative tolerance on the file size for considering the file. For a tolerance of 0.5, files with a size smaller than 50% of the average size of all files won't be considered.",
                  default="0.05")



(options,args) = parser.parse_args()

if len(args)!=3:
    parser.print_help()
    sys.exit(1)

castorDir = args[0]
regexp1 = args[1]
regexp2 = args[2]

(clean1, dirty1) = cleanFiles( castorDir, regexp1, options.cleanTolerance)
(clean2, dirty2) = cleanFiles( castorDir, regexp2, options.cleanTolerance)

print 'dirty files, 1: '
pprint.pprint( dirty1 )

print 'dirty files, 2: '
pprint.pprint( dirty2 )

if options.removeDirty:
    trash = 'Dirty'
    absTrash = createTrash( trash )
    remove( absTrash, dirty1 )
    remove( absTrash, dirty2 )
elif len(dirty1) or len(dirty2):
    print 'to remove dirty files in both collections, run again with option -d'

single = sync( clean1, clean2 )

if options.removeSingle:
    trash = 'Single'
    absTrash = createTrash( trash )
    remove( absTrash, single )
elif len(single):
    print 'to remove single files in both collections, run again with option -s'

