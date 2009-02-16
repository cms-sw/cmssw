import FWCore.ParameterSet.Config as cms

#-------------------------------------------------
# test cfg file for mva training for jet parton 
# association
#-------------------------------------------------
process = cms.Process("COMPUTER")

## add message logger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = 'WARNINGS'

#-------------------------------------------------
# process configuration
#-------------------------------------------------

## define your trainingfile(s)
process.source = cms.Source("PoolSource",
     fileNames = cms.untracked.vstring(
     "file:/tmp/mrenz/CMSSW_2_2_4/src/job_0_TQAFLayer1_Output.root",
     "file:/tmp/mrenz/CMSSW_2_2_4/src/job_1_TQAFLayer1_Output.root",
     ),
     skipEvents = cms.untracked.uint32(0)
)

## define maximal number of events to loop over
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

## configure process options
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(False)
)

## configure geometry
process.load("Configuration.StandardSequences.Geometry_cff")

## configure conditions
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = cms.string('IDEAL_V9::All')

## load magnetic field
process.load("Configuration.StandardSequences.MagneticField_cff")

## configure mva computer
process.load("TopQuarkAnalysis.TopEventSelection.TtSemiLepSignalSelMVAComputer_cff")

process.out = cms.OutputModule(
  "PoolOutputModule",
  SelectEvents = cms.untracked.PSet( SelectEvents = cms.vstring('p')),
  outputCommands = cms.untracked.vstring('drop *',
                                         'keep double_*_DiscSel_*'
                                        ),
  verbose = cms.untracked.bool(True),
  fileName = cms.untracked.string('MVAComputer_Output.root')
)

#-------------------------------------------------
# process paths;
#-------------------------------------------------
process.p = cms.Path(process.findTtSemiLepSignalSelMVA)

## output
process.outpath = cms.EndPath(process.out)
                      
