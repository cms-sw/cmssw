import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

## add message logger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.ParticleListDrawer=dict()
process.MessageLogger.cerr.TtDecayChannelSelector = cms.untracked.PSet(
    limit = cms.untracked.int32(-1)
)

## define input
from TopQuarkAnalysis.TopEventProducers.tqafInputFiles_cff import relValTTbar
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(relValTTbar),
    skipEvents = cms.untracked.uint32(0)
)

## define maximal number of events to loop over
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)

## configure process options
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

## load decay-channel filters
process.load("TopQuarkAnalysis.TopSkimming.ttDecayChannelFilters_cff")

## add one of the following two lines in order to choose either muon+jets or electron+jets events only
#process.ttSemiLeptonicFilter.allowedTopDecays.decayBranchA.electron = False
#process.ttSemiLeptonicFilter.allowedTopDecays.decayBranchA.muon     = False

## this is how one would add those lepton+jets events in which a tau decays into an electron or muon
#process.ttSemiLeptonicFilter.allowedTopDecays.decayBranchA.tau = True
#process.ttSemiLeptonicFilter.restrictTauDecays.electron = cms.bool(True)
#process.ttSemiLeptonicFilter.restrictTauDecays.muon     = cms.bool(True)

## and this is how one would add those dileptonic events in which a tau decays into an electron or muon
#process.ttFullLeptonicFilter.allowedTopDecays.decayBranchA.tau = True
#process.ttFullLeptonicFilter.allowedTopDecays.decayBranchB.tau = True
#process.ttFullLeptonicFilter.restrictTauDecays.electron = cms.bool(True)
#process.ttFullLeptonicFilter.restrictTauDecays.muon     = cms.bool(True)

## produce printout of particle listings (for debugging)
#process.load("TopQuarkAnalysis.TopEventProducers.sequences.printGenParticles_cff")
#process.printDecaySubset.maxEventsToPrint = 1

## paths
process.p1 = cms.Path(process.ttFullHadronicFilter)
process.p2 = cms.Path(process.ttSemiLeptonicFilter)
process.p3 = cms.Path(process.ttFullLeptonicFilter)

## the following lines illustrate how one would use the TtGenEvent as input instead of the std collection of GenParticles
#process.load("TopQuarkAnalysis.TopEventProducers.sequences.ttGenEvent_cff")
#process.ttFullHadronicFilter.src = "genEvt"
#process.p1.replace(process.ttFullHadronicFilter,
#                   process.makeGenEvt*process.ttFullHadronicFilter)

