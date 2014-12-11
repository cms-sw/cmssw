import FWCore.ParameterSet.Config as cms  

from Validation.RecoParticleFlow.PFJetFilter_cfi import *
from RecoJets.Configuration.GenJetParticles_cff import *
from RecoJets.Configuration.RecoGenJets_cff import *

# should do a cloning
pfGenParticlesForJets = genParticlesForJetsNoNu.clone()
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

pfAk4GenJetsNoNu = ak4GenJetsNoNu.clone()
pfAk4GenJetsNoNu.src = 'pfGenParticlesForJets'

pfJetFilter.InputTruthLabel = 'pfAk4GenJetsNoNu'
#pfJetFilter.verbose = True

pfFilter =cms.Sequence(
    pfGenParticlesForJets*
    pfAk4GenJetsNoNu*
    pfJetFilter
)
