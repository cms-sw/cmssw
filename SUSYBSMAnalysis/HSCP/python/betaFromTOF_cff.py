import FWCore.ParameterSet.Config as cms

from SUSYBSMAnalysis.HSCP.betaFromTOF_cfi import *
from RecoMuon.Configuration.RecoMuon_cff import *
from RecoLocalMuon.DTSegment.dt4DSegments_MTPatternReco4D_ParamDrift_cfi import *
from CalibMuon.Configuration.DT_FakeConditions_cff import *
muonTOF = cms.Sequence(dt4DSegments+MuonSeed+standAloneMuons+betaFromTOF)
dt4DSegments.Reco4DAlgoConfig.Reco2DAlgoConfig.debug = False
dt4DSegments.Reco4DAlgoConfig.debug = False
#standAloneMuons.STATrajBuilderParameters.BWFilterParameters.MuonTrajectoryUpdatorParameters.Granularity = 0
standAloneMuons.STATrajBuilderParameters.BWFilterParameters.MaxChi2 = 1000
standAloneMuons.STATrajBuilderParameters.BWFilterParameters.MuonTrajectoryUpdatorParameters.MaxChi2 = 1000
MuonSeed.EnableCSCMeasurement = False
#standAloneMuons.STATrajBuilderParameters.RefitterParameters.EnableRPCMeasurement = False
#standAloneMuons.STATrajBuilderParameters.BWFilterParameters.EnableRPCMeasurement = False
#standAloneMuons.STATrajBuilderParameters.RefitterParameters.EnableCSCMeasurement = False
#standAloneMuons.STATrajBuilderParameters.BWFilterParameters.EnableCSCMeasurement = False


