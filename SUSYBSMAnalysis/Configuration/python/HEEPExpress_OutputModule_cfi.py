import FWCore.ParameterSet.Config as cms

#
#
# Output module for HEEP Express Stream Skim
#
#
from SUSYBSMAnalysis.Configuration.HEEPExpress_EventContent_cff import *
hEEPExpressOutputModule = cms.OutputModule("PoolOutputModule",
    hEEPExpressEventSelection,
    AODSIMHEEPExpressEventContent,
    datasets = cms.untracked.PSet(
        filterName = cms.untracked.string('hEEPExpress'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('hEEPExpress.root')
)


