#!/usr/bin/env python

# Script to submit Tau Validation jobs to lxbatch
#  Author: Evan Friis evan.klose.friis@cern.ch
import time
import random
from Validation.RecoTau.ValidationOptions_cff import *

options.parseArguments()
checkOptionsForBadInput()

# Make sure we dont' clobber another directory!
if not CMSSWEnvironmentIsCurrent():
   print "CMSSW_BASE points to a different directory, please rerun cmsenv!"
   sys.exit()

if options.nJobs == 0:
   print "Must specify nJobs > 0. Run 'python LXBatchValidation.py help' for options"
   sys.exit()

if options.maxEvents == -1 and options.nJobs > 1:
   print "Please use maxEvents to specify the number of events to process per job."
   sys.exit()

# Setup path to CASTOR home if desired
if options.writeEDMFile != "":
   if options.copyToCastorDir == "<home>":
      options.copyToCastorDir = "/castor/cern.ch/user/%s/%s/" % (os.environ["LOGNAME"][0], os.environ["LOGNAME"])
      print "Setting castor directory to your home @ %s" % options.copyToCastorDir

if options.copyToCastorDir != "":
   checkCastor = os.system("nsls %s" % options.copyToCastorDir)
   if checkCastor:
      print "Error: castor reports an error when checking the supplied castor location: ", options.copyToCastorDir
      sys.exit()

#print "Converting all relative paths to absolutes..."
#if options.sourceFile != "<none>":
#  options.sourceFile = os.path.abspath(options.sourceFile)
#  if not os.path.exists(opitons.sourceFile):
#     print "Can't stat sourceFile %s after converting to absolute path." % options.sourceFile
#     sys.exit()

#bsoluteModifications = []
#or aModification in options.myModifications:
#  if aModification != "<none>":
#     absoluteModPath = os.path.abspath(aModification)
#     if not os.path.exists(absoluteModPath):
#        print "Can't stat modification file %s after converting to absolute path" % absoluteModPath
#        absoluteModifications.append(absoluteModPath)
#        sys.exit()

#ptions.myModifications = absoluteModifications

print "" 
print "I'm going to submit %i %s jobs, each with %i events, for a total of %i events" % (options.nJobs, options.eventType, options.maxEvents, options.nJobs*options.maxEvents)

if options.writeEDMFile != "":
   print "EDM files with the prefix %s will be produced and stored in %s" % (options.writeEDMFile, castorLocation)

print "Hit Ctrl-c in the next 3 seconds to cancel..."
try:
   time.sleep(2)
except KeyboardInterrupt:
   print "Canceled, exiting."
   sys.exit()

# Setup the environment variables
setupCommands = "cd $PWD; scramv1 runtime -sh > tempEnvs_%i; source tempEnvs_%i; rm tempEnvs_%i; cd -;"
cmsRunCommands = "cmsRun $PWD/RunValidation_cfg.py %s;" % returnOptionsString()
cleanupCommands = ""
if options.writeEDMFile != "":
   cleanupCommand += " rfcp *.root %s;" % options.copyToCastorDir

for iJob in xrange(0, options.nJobs):
   options.batchNumber = iJob #increment batch number
   #setupCommands = "cd $PWD; scramv1 runtime -sh > tempEnvs_%s_%i; source tempEnvs_%s_%i; rm tempEnvs_%s_%i; export PYTHONPATH=$PWD:$PYTHONPATH; cd -;" % (options.eventType, iJob, options.eventType, iJob, options.eventType, iJob)
   #cmsRunCommand = "cmsRun $PWD/RunValidation_cfg.py %s;" % returnOptionsString()
   #setupCommands = "cd $PWD; scramv1 runtime -sh > tempEnvs_%s_%i; source tempEnvs_%s_%i; rm tempEnvs_%s_%i; export PYTHONPATH=$PWD:$PYTHONPATH; cd -;" % (options.eventType, iJob, options.eventType, iJob, options.eventType, iJob)
   setupCommands = "export edmOutputDir=\$PWD; cd $PWD; scramv1 runtime -sh > tempEnvs_%s_%i; source tempEnvs_%s_%i; rm tempEnvs_%s_%i;" % (options.eventType, iJob, options.eventType, iJob, options.eventType, iJob)
   cmsRunCommand = "cmsRun RunValidation_cfg.py %s;" % returnOptionsString()
   cleanupCommand = ""
   if options.writeEDMFile != "":
      cleanupCommand += "cd -; rfcp *.root %s;" % options.copyToCastorDir

   totalCommand = setupCommands + cmsRunCommand + cleanupCommand
   bsubCommand  = "bsub -J %s_%i -q %s \"%s\"" % (options.eventType, iJob, options.lxbatchQueue, totalCommand)
   #bsubCommand  = "echo -J %s_%i -q %s \"%s\"" % (options.eventType, iJob, options.lxbatchQueue, totalCommand)
   os.system(bsubCommand)

