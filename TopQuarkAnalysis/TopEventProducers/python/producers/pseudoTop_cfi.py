import FWCore.ParameterSet.Config as cms

pseudoTop = cms.EDProducer("PseudoTopProducer",
    genParticles = cms.InputTag("prunedGenParticles"),
    finalStates = cms.InputTag("packedGenParticles"),
    minLeptonPt = cms.double(15),
    maxLeptonEta = cms.double(2.5),
    minJetPt = cms.double(30),
    maxJetEta = cms.double(2.4),
    leptonConeSize = cms.double(0.1),
    jetConeSize = cms.double(0.4),
    wMass = cms.double(80.4),
    tMass = cms.double(172.5),

    minLeptonPtDilepton = cms.double(20),
    maxLeptonEtaDilepton = cms.double(2.4),
    minDileptonMassDilepton = cms.double(20),
    minLeptonPtSemilepton = cms.double(20),
    maxLeptonEtaSemilepton = cms.double(2.4),
    minVetoLeptonPtSemilepton = cms.double(15),
    maxVetoLeptonEtaSemilepton = cms.double(2.5),
    minMETSemiLepton = cms.double(30),
    minMtWSemiLepton = cms.double(30),
)
