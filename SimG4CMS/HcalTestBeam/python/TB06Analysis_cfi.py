import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import *
from SimG4CMS.HcalTestBeam.TBDirectionParameters_cfi import *

testbeam = cms.EDAnalyzer("HcalTB06Analysis",
                          common_beam_direction_parameters,
                          ECAL = cms.bool(True),
                          TestBeamAnalysis = cms.PSet(
        Verbose = cms.untracked.bool(False),
        EHCalMax   = cms.untracked.double(400.0),
        ETtotMax   = cms.untracked.double(400.0),
        beamEnergy = cms.untracked.double(50.),
        TimeLimit  = cms.double(180.0),
        EcalWidth  = cms.double(0.362),
        HcalWidth  = cms.double(0.640),
        EcalFactor = cms.double(1.0),
        HcalFactor = cms.double(100.0),
        MIP        = cms.double(0.8),
        MakeTree   = cms.untracked.bool(False)
        )
                          )
