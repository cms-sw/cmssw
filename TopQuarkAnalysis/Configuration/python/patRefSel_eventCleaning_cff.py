import FWCore.ParameterSet.Config as cms

from CommonTools.RecoAlgos.HBHENoiseFilter_cfi import *

from DPGAnalysis.Skims.goodvertexSkim_cff import noscraping

eventCleaning = cms.Sequence(
  HBHENoiseFilter
* noscraping
)
