#

import FWCore.ParameterSet.Config as cms

process = cms.Process("MSTEST")

#Adding SimpleMemoryCheck service:
process.SimpleMemoryCheck=cms.Service("SimpleMemoryCheck",
                                   ignoreTotal=cms.untracked.int32(1),
                                   oncePerEventMode=cms.untracked.bool(True)
)

process.Timing = cms.Service("Timing"
    ,excessiveTimeThreshold=cms.untracked.double(600)
    ,summaryOnly = cms.untracked.bool(True)
)

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)



process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

# Example configuration for the magnetic field

# Uncomment ONE of the following:

### Uniform field
#process.load("Configuration.StandardSequences.MagneticField_0T_cff")
#process.localUniform.ZFieldInTesla = 3.8


### Full field map, static configuration for each field value
#process.load("Configuration.StandardSequences.MagneticField_20T_cff")
#process.load("Configuration.StandardSequences.MagneticField_30T_cff")
#process.load("Configuration.StandardSequences.MagneticField_35T_cff")
#process.load("Configuration.StandardSequences.MagneticField_38T_cff")
#process.load("Configuration.StandardSequences.MagneticField_40T_cff")

### Configuration to select map based on recorded current in the DB
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
#process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')
#process.GlobalTag = GlobalTag(process.GlobalTag,'auto:phase1_2017_design', '')
process.GlobalTag = GlobalTag(process.GlobalTag,'auto:phase1_2017_realistic', '')
#process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic', '')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load("Configuration.StandardSequences.MagneticField_cff")
process.VolumeBasedMagneticFieldESProducer.valueOverride = 18000

from RecoTracker.TkNavigation.TkMSParameterizationBuilder_cfi import *
process.load('RecoTracker.TkNavigation.TkMSParameterizationBuilder_cfi')

process.myTest  = cms.EDAnalyzer("TkMSParameterizationTest",
)
process.p1 = cms.Path(process.myTest)

