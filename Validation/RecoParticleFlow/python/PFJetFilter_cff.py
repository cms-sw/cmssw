import FWCore.ParameterSet.Config as cms  

from Validation.RecoParticleFlow.PFJetFilter_cfi import *
from RecoJets.Configuration.GenJetParticles_cff import *
from RecoJets.Configuration.RecoGenJets_cff import *

# should do a cloning
pfGenParticlesForJets = genParticlesForJets.clone()
pfGenParticlesForJets.ignoreParticleIDs.append(14)
pfGenParticlesForJets.ignoreParticleIDs.append(12)
pfGenParticlesForJets.ignoreParticleIDs.append(16)
pfGenParticlesForJets.ignoreParticleIDs.append(1)
pfGenParticlesForJets.ignoreParticleIDs.append(2)
pfGenParticlesForJets.ignoreParticleIDs.append(3)
pfGenParticlesForJets.ignoreParticleIDs.append(4)
pfGenParticlesForJets.ignoreParticleIDs.append(5)
pfGenParticlesForJets.ignoreParticleIDs.append(21)
pfGenParticlesForJets.excludeResonances = False

pfIterativeCone5GenJets = iterativeCone5GenJets.clone()
pfIterativeCone5GenJets.src = 'pfGenParticlesForJets'

pfJetFilter.InputTruthLabel = 'pfIterativeCone5GenJets'

pfFilter =cms.Sequence(
    pfGenParticlesForJets*
    pfIterativeCone5GenJets*
    pfJetFilter
)
