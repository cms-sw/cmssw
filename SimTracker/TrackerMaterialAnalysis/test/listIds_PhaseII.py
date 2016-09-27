#! /usr/bin/env cmsRun

import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras

process = cms.Process('RECO',eras.Phase2C1)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.Geometry.GeometryExtended2023D1Reco_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

# Additional output definition

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(1),
            firstRun = cms.untracked.uint32(1)
            )

process.listIds = cms.EDAnalyzer("ListIds")
process.path = cms.Path(process.listIds)

# customisation of the process.

# Automatic addition of the customisation function from SLHCUpgradeSimulations.Configuration.combinedCustoms
from SLHCUpgradeSimulations.Configuration.combinedCustoms import cust_2023tilted 

#call to customisation function cust_2023tilted imported from SLHCUpgradeSimulations.Configuration.combinedCustoms
process = cust_2023tilted(process)

# End of customisation functions

# Customisation from command line
