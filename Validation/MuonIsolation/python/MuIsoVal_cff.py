import FWCore.ParameterSet.Config as cms

from RecoMuon.MuonIsolationProducers.muIsoDeposits_cff import *
from TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAny_cfi import *


DQMStore = cms.Service("DQMStore")

MuIsoValidation_inc = DQMStep1Module('MuIsoValidation',
    Global_Muon_Label = cms.untracked.InputTag("muons"),
    requireCombinedMuon = cms.untracked.bool(False),
    ecalIsoDeposit_Label = cms.untracked.InputTag("muIsoDepositCalByAssociatorTowers","ecal"),
#    rootfilename = cms.untracked.string('ttbar-validation.root'),
    hcalIsoDeposit_Label = cms.untracked.InputTag("muIsoDepositCalByAssociatorTowers","hcal"),
    tkIsoDeposit_Label = cms.untracked.InputTag("muIsoDepositTk"),
    hoIsoDeposit_Label = cms.untracked.InputTag("muIsoDepositCalByAssociatorTowers","ho"),
    directory = cms.untracked.string("Muons/MuonIsolationV_inc")                             
)

MuIsoValidation_global = DQMStep1Module('MuIsoValidation',
    Global_Muon_Label = cms.untracked.InputTag("muons"),
    requireCombinedMuon = cms.untracked.bool(True),
    ecalIsoDeposit_Label = cms.untracked.InputTag("muIsoDepositCalByAssociatorTowers","ecal"),
#    rootfilename = cms.untracked.string('ttbar-validation.root'),
    hcalIsoDeposit_Label = cms.untracked.InputTag("muIsoDepositCalByAssociatorTowers","hcal"),
    tkIsoDeposit_Label = cms.untracked.InputTag("muIsoDepositTk"),
    hoIsoDeposit_Label = cms.untracked.InputTag("muIsoDepositCalByAssociatorTowers","ho"),
    directory = cms.untracked.string("Muons/MuonIsolationV_global")
)

muIsoVal_seq = cms.Sequence(MuIsoValidation_inc+MuIsoValidation_global)
