import FWCore.ParameterSet.Config as cms

from RecoMuon.MuonIsolationProducers.muIsoDeposits_cff import *
from Geometry.CMSCommonData.cmsIdealGeometryXML_cfi import *
from Configuration.StandardSequences.MagneticField_cff import *
from Geometry.CaloEventSetup.CaloGeometry_cfi import *
from Geometry.CommonDetUnit.globalTrackingGeometry_cfi import *
from TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAny_cfi import *

maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

DQMStore = cms.Service("DQMStore")

MuIsoValidation_inc = cms.EDFilter("MuIsoValidation",
    Global_Muon_Label = cms.untracked.InputTag("muons"),
    requireCombinedMuon = cms.untracked.bool(False),
    ecalIsoDeposit_Label = cms.untracked.InputTag("muIsoDepositCalByAssociatorTowers","ecal"),
#    rootfilename = cms.untracked.string('ttbar-validation.root'),
    hcalIsoDeposit_Label = cms.untracked.InputTag("muIsoDepositCalByAssociatorTowers","hcal"),
    tkIsoDeposit_Label = cms.untracked.InputTag("muIsoDepositTk"),
    hoIsoDeposit_Label = cms.untracked.InputTag("muIsoDepositCalByAssociatorTowers","ho")
)

MuIsoValidation_global = cms.EDFilter("MuIsoValidation",
    Global_Muon_Label = cms.untracked.InputTag("muons"),
    requireCombinedMuon = cms.untracked.bool(True),
    ecalIsoDeposit_Label = cms.untracked.InputTag("muIsoDepositCalByAssociatorTowers","ecal"),
#    rootfilename = cms.untracked.string('ttbar-validation.root'),
    hcalIsoDeposit_Label = cms.untracked.InputTag("muIsoDepositCalByAssociatorTowers","hcal"),
    tkIsoDeposit_Label = cms.untracked.InputTag("muIsoDepositTk"),
    hoIsoDeposit_Label = cms.untracked.InputTag("muIsoDepositCalByAssociatorTowers","ho")
)

muIsoVal_seq = cms.Sequence(MuIsoValidation_inc+MuIsoValidation_global)
