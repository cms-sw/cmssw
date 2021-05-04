# The following comments couldn't be translated into the new config version:

#	muIsoDepositCalByAssociatorTowers,

import FWCore.ParameterSet.Config as cms

process = cms.Process("A")
process.load("RecoMuon.MuonIsolationProducers.muIsoDeposits_cff")

process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

process.load("MagneticField.Engine.volumeBasedMagneticField_cfi")

process.load("Geometry.CaloEventSetup.CaloGeometry_cfi")

process.load("Geometry.CommonTopologies.globalTrackingGeometry_cfi")

process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAny_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(2000)
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
    ecalIsoDeposit_Label = cms.untracked.InputTag("muIsoDepositCalByAssociatorTowers","ecal"),
    rootfilename = cms.untracked.string('Z-MM-validation.root'),
    hcalIsoDeposit_Label = cms.untracked.InputTag("muIsoDepositCalByAssociatorTowers","hcal"),
    tkIsoDeposit_Label = cms.untracked.InputTag("muIsoDepositTk"),
    hoIsoDeposit_Label = cms.untracked.InputTag("muIsoDepositCalByAssociatorTowers","ho")
)

process.analyzer_combinedMuon = DQMEDAnalyzer('MuIsoValidation',
    Global_Muon_Label = cms.untracked.InputTag("muons"),
    requireCombinedMuon = cms.untracked.bool(True),
    ecalIsoDeposit_Label = cms.untracked.InputTag("muIsoDepositCalByAssociatorTowers","ecal"),
    rootfilename = cms.untracked.string('Z-MM-validation.root'),
    hcalIsoDeposit_Label = cms.untracked.InputTag("muIsoDepositCalByAssociatorTowers","hcal"),
    tkIsoDeposit_Label = cms.untracked.InputTag("muIsoDepositTk"),
    hoIsoDeposit_Label = cms.untracked.InputTag("muIsoDepositCalByAssociatorTowers","ho")
)

process.p = cms.Path(process.analyzer_incMuon+process.analyzer_combinedMuon)
process.PoolSource.fileNames = ['/store/relval/2008/3/19/RelVal-RelValZMUMU-1205894772-HLT/0000/0A5C7078-18F6-DC11-B01A-001617E30CD4.root', '/store/relval/2008/3/19/RelVal-RelValZMUMU-1205894772-HLT/0000/30575B3F-18F6-DC11-8565-001617C3B69C.root', '/store/relval/2008/3/19/RelVal-RelValZMUMU-1205894772-HLT/0000/32E5B292-19F6-DC11-A8D3-000423D6C8EE.root', '/store/relval/2008/3/19/RelVal-RelValZMUMU-1205894772-HLT/0000/3CCD7325-1AF6-DC11-838C-000423D9870C.root', '/store/relval/2008/3/19/RelVal-RelValZMUMU-1205894772-HLT/0000/6627B336-18F6-DC11-BDEB-001617E30F4C.root', 
    '/store/relval/2008/3/19/RelVal-RelValZMUMU-1205894772-HLT/0000/6ABE2EA8-17F6-DC11-9EF9-001617C3B6CE.root', '/store/relval/2008/3/19/RelVal-RelValZMUMU-1205894772-HLT/0000/7ACE00EB-19F6-DC11-8D71-000423D6AF24.root', '/store/relval/2008/3/19/RelVal-RelValZMUMU-1205894772-HLT/0000/FE66C439-18F6-DC11-BC7A-001617C3B782.root']

