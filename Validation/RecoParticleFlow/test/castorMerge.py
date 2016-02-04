#!/usr/bin/env python

from optparse import OptionParser
import sys,os, re, pprint
import FWCore.ParameterSet.Config as cms

chunkNumber = 0


def createDir( dir ):
    absName = '%s/%s' % (castorDir, dir)
    out = os.system( 'rfdir %s' % absName )
    print out
    if out!=0:
        # dir does not exist
        os.system( 'rfmkdir %s' % absName )
    return absName


def processFiles( files ):
    
    global chunkNumber

    print 'Processing files:'
    pprint.pprint( files )
    
    process = cms.Process("COPY")

    process.source = cms.Source(
        "PoolSource",
        fileNames = cms.untracked.vstring( files ),
        noEventSort = cms.untracked.bool(True),
        duplicateCheckMode = cms.untracked.string('noDuplicateCheck')
        )
    
    # build output file name
    
    
    tmpRootFile = '/tmp/aod_QCDForPF_Full_chunk%d.root' % chunkNumber

    print '  destination: ', tmpRootFile
    process.aod = cms.OutputModule(
        "PoolOutputModule",
        fileName = cms.untracked.string( tmpRootFile ),
        outputCommands = cms.untracked.vstring( 'keep *' )
        )
    

    process.outpath = cms.EndPath(process.aod)

    outFile = open("tmpConfig.py","w")
    outFile.write("import FWCore.ParameterSet.Config as cms\n")
    outFile.write(process.dumpPython())
    outFile.close()

    chunkNumber = chunkNumber+1

    if options.negate == True:
        return

    chunkDir = createDir( 'Chunks' )
    
    os.system("cmsRun tmpConfig.py")
    print 'done. transferring file to: ', chunkDir
    os.system("rfcp %s %s" % (tmpRootFile, chunkDir) )
    print 'done'
    os.system("rm %s" % tmpRootFile)
    print 'temporary files removed.'

    
parser = OptionParser()
parser.usage = "%prog <castor dir> <regexp pattern> <chunk size>: merge a set of CMSSW root files on castor."
parser.add_option("-n", "--negate", action="store_true",
                  dest="negate",
                  help="do not produce the merged files",
                  default=False)



(options,args) = parser.parse_args()

if len(args)!=3:
    parser.print_help()
    sys.exit(1)

castorDir = args[0]
regexp = args[1]
chunkSize = int(args[2])

print 'Merging files in: ', castorDir

try:
    pattern = re.compile( regexp )
except:
    print 'please enter a valid regular expression '
    sys.exit(1)

allFiles = os.popen("rfdir %s | awk '{print $9}'" % (castorDir))

matchingFiles = []


print 'matching files:'
for file in allFiles.readlines():
    file = file.rstrip()

    m = pattern.match( file )
    if m:
        print file
        fullCastorFile = 'rfio:%s/%s' % (castorDir, file)
        matchingFiles.append( fullCastorFile )



# grouping files
count = 0
chunk = []
for file in matchingFiles:
    count += 1
    chunk.append( file )
    if count == chunkSize:
        count = 0
        processFiles( chunk )
        chunk = []
        
# remaining files:
if len(chunk)>0:
    processFiles( chunk )
        



