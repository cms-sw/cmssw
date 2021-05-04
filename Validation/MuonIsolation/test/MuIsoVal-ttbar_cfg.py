# The following comments couldn't be translated into the new config version:

#  Obsolete service from 1_x_x
#	service = DaqMonitorROOTBackEnd{}
#  Replaced by

#	muIsoDepositCalByAssociatorTowers,

import FWCore.ParameterSet.Config as cms

process = cms.Process("A")
process.load("RecoMuon.MuonIsolationProducers.muIsoDeposits_cff")

process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

process.load("Configuration.StandardSequences.MagneticField_cff")

process.load("Geometry.CaloEventSetup.CaloGeometry_cfi")

process.load("Geometry.CommonTopologies.globalTrackingGeometry_cfi")

process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAny_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('XXXX', 
        'XXXX', 
        'XXXX')
)

process.DQMStore = cms.Service("DQMStore")

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
process.analyzer_incMuon = DQMEDAnalyzer('MuIsoValidation',
    Global_Muon_Label = cms.untracked.InputTag("muons"),
    requireCombinedMuon = cms.untracked.bool(False),
    rootfilename = cms.untracked.string('ttbar-validation.root'),
)

process.analyzer_combinedMuon = DQMEDAnalyzer('MuIsoValidation',
    Global_Muon_Label = cms.untracked.InputTag("muons"),
    requireCombinedMuon = cms.untracked.bool(True),
    rootfilename = cms.untracked.string('ttbar-validation.root'),
)

process.p = cms.Path(process.analyzer_incMuon+process.analyzer_combinedMuon)
process.PoolSource.fileNames = ['/store/relval/CMSSW_3_1_0_pre4/RelValTTbar/GEN-SIM-RECO/IDEAL_30X_v1/0003/00E48100-3A16-DE11-A693-001617DBCF6A.root',
                                '/store/relval/CMSSW_3_1_0_pre4/RelValTTbar/GEN-SIM-RECO/IDEAL_30X_v1/0003/12C01897-4616-DE11-8AA7-000423D98B5C.root',
                                '/store/relval/CMSSW_3_1_0_pre4/RelValTTbar/GEN-SIM-RECO/IDEAL_30X_v1/0003/289FC85A-4216-DE11-ACEE-000423D98844.root',
                                '/store/relval/CMSSW_3_1_0_pre4/RelValTTbar/GEN-SIM-RECO/IDEAL_30X_v1/0003/6ED9476F-4C16-DE11-8BFC-001617C3B76A.root',
                                '/store/relval/CMSSW_3_1_0_pre4/RelValTTbar/GEN-SIM-RECO/IDEAL_30X_v1/0003/76E8D7B2-5216-DE11-8A7A-000423D174FE.root',
                                '/store/relval/CMSSW_3_1_0_pre4/RelValTTbar/GEN-SIM-RECO/IDEAL_30X_v1/0003/B0D94AFE-3616-DE11-BFD5-000423D9880C.root',
                                '/store/relval/CMSSW_3_1_0_pre4/RelValTTbar/GEN-SIM-RECO/IDEAL_30X_v1/0003/BCE77A07-AC16-DE11-80B9-000423D986A8.root']

