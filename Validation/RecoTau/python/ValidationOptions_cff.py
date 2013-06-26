import os
import sys
try:
   ReleaseBase = os.path.join(os.environ['CMSSW_BASE'], "src")
   ReleaseVersion = os.environ['CMSSW_VERSION']
except KeyError:
   print "CMSSW enviroment not set, please run cmsenv!"
   sys.exit()

import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

#options = VarParsing.VarParsing ('standard')
options = VarParsing.VarParsing ()

allowedOptions = {}

options.register( 'maxEvents',
                   -1,
                   VarParsing.VarParsing.multiplicity.singleton,
                   VarParsing.VarParsing.varType.int,
                   "Specify events to run."
                )

options.register( 'eventType',
                  "ZTT",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  #"If true, generate and validate Z-TauTau (hadronic only) events. Otherwise, generate QCD FlatPt 15-3000 events."
                  "Specify the type of events to validate. (and generate, if using fast or fullsim)\
                        \n\t\tOptions:\
                        \n\t\t\tZTT\
                        \n\t\t\tQCD\
                        \n\t\t\tZEE\
                        \n\t\t\tZMM\
                        \n\t\t\tZTTFastSim\
                        \n\t\t\tZEEFastSim\
                        \n\t\t\tRealData          (Jets faking taus)\
                        \n\t\t\tRealMuonsData     (Iso Muons faking taus)\
                        \n\t\t\tRealElectronsData (Iso Electrons faking taus)\n"
                 )

allowedOptions['eventType'] = [ 'ZTT', 'QCD', 'ZEE', 'ZMM', 'RealData', 'RealMuonsData', 'RealElectronsData','ZTTFastSim','ZEEFastSim']

options.register( 'label',
                  "none",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "Validation output files will be stored in folder Validation_<label>.  The root file will also use this label.\n"
                 )

options.register( 'dataSource',
                  'recoFiles',
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "Specify where the data should come from. \n\t\tOptions: \
                        \n\t\t\trecoFiles:\t\t\tGet data from [sourceFile] (must have RECO)\
                        \n\t\t\trecoFiles+PFTau:\t\tGet reco data as above, and rerun PFTau with current tags \
                        \n\t\t\trecoFiles+PFTau+CaloTau:\t\tRun CaloTau too \
                        \n\t\t\tdigiFiles:\t\t\tGet data from [sourceFile] (must have DIGI) and rerun RECO \
                        \n\t\t\tfastsim:\t\t\tRun FastSim \
                        \n\t\t\tfullsim:\t\t\tGen-Sim-Digi-Reco-Validate!\n"
                  )

allowedOptions['dataSource'] = ['recoFiles', 'recoFiles+PFTau', 'recoFiles+PFTau+CaloTau', 'recoFiles+CaloTau', 'recoFiles+CaloTau+PFTau', 'fastsim', 'digiFiles', 'fullsim']

options.register( 'sourceFile',
                  'none',
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "Specify a file containing events to read if using the recoFiles or digiFiles options. If left as none files will be taken from\
                        \n\t\t\trecoFiles:\t\t\tGet data from [sourceFile]EventSource_<eventType>_RECO_cff.py \
                        \n\t\t\tdigiFiles:\t\t\tGet data from EventSource_<eventType>_DIGI_cff.py \n"
                  )

options.register( 'conditions',
                  'whatever',
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "Specify global tag for conditions.  If left as default ('whatever'), whatever is defined in Configuration.FrontierConditions_GlobalTag_cff will be taken\n"
                )

options.register( 'myModifications',
                  'none',
                  VarParsing.VarParsing.multiplicity.list,
                  VarParsing.VarParsing.varType.string,
                  "Specify one (or more) files to be loaded in the process. You can use this to modifiy cfi parameters from their defaults. See ExampleModification_cfi.py"
                  )

options.register( 'batchNumber',
                  -1,
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.int,
                  "Run in batch mode: add batch number to output file, dont' compute efficiency, and change random seed.  If running from files, skipEvents will automatically be \
                   set."
               )

options.register( 'writeEDMFile',
                  "",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "Specify an (optional) output file.  Batch number and .root suffix will be appended. Event content will PFTau RECO content along with genParticles"
               )

options.register( 'edmFileContents',
                  "AODSim",
                  VarParsing.VarParsing.multiplicity.list,
                  VarParsing.VarParsing.varType.string,
                  "Specify event content. Not yet implemented"
                )

options.register( 'gridJob',
                  False,
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.bool,
                  "Set to true makes some modifications to enable to run on grid. On the analysis has the same effect of running in batch, but has also other features."
                )

################################
#
#        Batchmode options
#
################################

# add options for number of batchJobs
options.register( 'nJobs',
                  0,
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.int,
                  "Number of jobs to submit to LXBatch. [Batchmode only]"
                )

options.register( 'copyToCastorDir',
                  "",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "If writeEDMFile is specified, copy the edmOutputFile to castor.  Specify <home> to use your home dir. [Batchmode only]"
                )

options.register( 'lxbatchQueue',
                  '8nh',
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "LXBatch queue (8nm, 1nh, 8nh, 1nd, 1nw) [Batchmode only]"
                )

allowedOptions['lxbatchQueue'] = ['8nm', '1nh', '8nh', '1nd', '1nw']


def checkOptionsForBadInput():
   # Sanity check
   for optionName, allowedValues in allowedOptions.iteritems():
      if not getattr(options, optionName) in allowedValues:
         print "Bad input to option: %s" % optionName
         sys.exit()

def calledBycmsRun():
   ''' Returns true of this python file is being called via cmsRun '''
   if sys.argv[0].find('cmsRun') == -1:
      return False
   else:
      return True

def CMSSWEnvironmentIsCurrent():
   ''' Make sure that our CMSSW environment doesn't point ot another release!'''
   if ReleaseBase != os.path.commonprefix([ReleaseBase, os.getcwd()]):
      return False
   else:
      return True

def returnOptionsString():
   ''' format the options to be passed on the command line.  Used when submitting batch jobs'''
   outputString = ""
   for optionsName, optionValue in options.__dict__['_singletons'].iteritems():
      outputString += " %s=%s" % (optionsName, optionValue)

   for optionsName, optionValues in options.__dict__['_lists'].iteritems():
      for anOption in optionValues:
         outputString += " %s=%s" % (optionsName, anOption) 
   return outputString
