
import FWCore.ParameterSet.Config as cms

from SLHCUpgradeSimulations.Configuration.muonCustomsPreMixing import customise_csc_PostLS1
import postLS1Customs
from Configuration.StandardSequences.Eras import eras

# restore a few settings that customisePostLS1 is not supposed to over-write for fastsim
# (temporary measure)
def fastSimFix(process):    
    process.l1extraParticles.centralJetSource = cms.InputTag("simCaloStage1LegacyFormatDigis","cenJets")
    process.l1extraParticles.etHadSource = cms.InputTag("simCaloStage1LegacyFormatDigis")
    process.l1extraParticles.etMissSource = cms.InputTag("simCaloStage1LegacyFormatDigis")
    process.l1extraParticles.etTotalSource = cms.InputTag("simCaloStage1LegacyFormatDigis")
    process.l1extraParticles.forwardJetSource = cms.InputTag("simCaloStage1LegacyFormatDigis","forJets")
    process.l1extraParticles.hfRingBitCountsSource = cms.InputTag("simCaloStage1LegacyFormatDigis")
    process.l1extraParticles.hfRingEtSumsSource = cms.InputTag("simCaloStage1LegacyFormatDigis")
    process.l1extraParticles.htMissSource = cms.InputTag("simCaloStage1LegacyFormatDigis")
    process.l1extraParticles.isoTauJetSource = cms.InputTag("simCaloStage1LegacyFormatDigis","isoTauJets")
    process.l1extraParticles.isolatedEmSource = cms.InputTag("simCaloStage1LegacyFormatDigis","isoEm")
    process.l1extraParticles.nonIsolatedEmSource = cms.InputTag("simCaloStage1LegacyFormatDigis","nonIsoEm")
    process.l1extraParticles.tauJetSource = cms.InputTag("simCaloStage1LegacyFormatDigis","tauJets")
    return process

def customisePostLS1(process):

    # apply the general 25 ns post-LS1 customisation
    process = postLS1Customs.customisePostLS1(process)
    # restore a few settings that customisePostLS1 is not supposed to over-write for fastsim
    if eras.fastSim.isChosen():
        process = fastSimFix(process)
    # deal with premixing-specific CSC changes separately
    process = customise_csc_PostLS1(process)

    return process


def customisePostLS1_50ns(process):

    # apply the general 25 ns post-LS1 customisation
    process = postLS1Customs.customisePostLS1_50ns(process)
    # restore a few settings that customisePostLS1 is not supposed to over-write for fastsim
    if eras.fastSim.isChosen():
        process = fastSimFix(process)
    # deal with premixing-specific CSC changes separately
    process = customise_csc_PostLS1(process)

    return process


def customisePostLS1_HI(process):

    # apply the general 25 ns post-LS1 customisation
    process = postLS1Customs.customisePostLS1_HI(process)
    # restore a few settings that customisePostLS1 is not supposed to over-write for fastsim
    if eras.fastSim.isChosen():
        process = fastSimFix(process)
    # deal with premixing-specific CSC changes separately
    process = customise_csc_PostLS1(process)

    return process

