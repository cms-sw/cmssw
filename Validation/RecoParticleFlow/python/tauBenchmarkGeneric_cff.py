import FWCore.ParameterSet.Config as cms

from Validation.RecoParticleFlow.goodGenJets_cfi import *

from Validation.RecoParticleFlow.pfTauBenchmarkGeneric_cfi import pfTauBenchmarkGeneric
from Validation.RecoParticleFlow.caloTauBenchmarkGeneric_cfi import caloTauBenchmarkGeneric
from PhysicsTools.JetMCAlgos.TauGenJets_cfi import tauGenJets
from PhysicsTools.JetMCAlgos.TauGenJetsDecayModeSelectorAllHadrons_cfi import tauGenJetsSelectorAllHadrons
from Validation.RecoParticleFlow.GenJetClosestMatchSelector_cfi import genJetClosestMatchSelector

# setting the sources

def changeSource( benchmark, rec, gen ):
    #benchmark.BenchmarkLabel = rec
    benchmark.InputRecoLabel = rec
    benchmark.InputTruthLabel = gen
    

fromTaus = True

# taking the jets, reconstructed from the large jet cone
# and as a reference the closest genjet (reconstructed with the same cone)
# from the tau genjets. In this way, the contribution of the 
# underlying event cancels
pfsource = 'iterativeCone5PFJets'
calosource = 'iterativeCone5CaloJets'
gensource = 'genJetClosestMatchSelector'
trueTaus = cms.Sequence(
    goodGenJets +
    tauGenJets + 
    tauGenJetsSelectorAllHadrons + 
    genJetClosestMatchSelector  
    )

if( fromTaus==True):
    # taking the taus, reconstructed from the signal cone elements
    # and as a reference the tau genjets
    pfsource = 'pfRecoTauProducerHighEfficiency'
    gensource = 'tauGenJetsSelectorAllHadrons'
    trueTaus = cms.Sequence(
        tauGenJets + 
        tauGenJetsSelectorAllHadrons
    )
    

changeSource(pfTauBenchmarkGeneric, 
             pfsource, 
             gensource)
changeSource(caloTauBenchmarkGeneric, 
             calosource, 
             gensource)



# define barrel and endcap benchmarks for the PF case: 

pfBarrel = pfTauBenchmarkGeneric.clone()
pfBarrel.minEta = -1
pfBarrel.maxEta = 1.4
pfBarrel.BenchmarkLabel = cms.string('%s_barrel' % pfBarrel.BenchmarkLabel.value() )

pfEndcap = pfTauBenchmarkGeneric.clone()
pfEndcap.minEta = 1.6
pfEndcap.maxEta = 2.8
pfEndcap.BenchmarkLabel = cms.string('%s_endcap' % pfEndcap.BenchmarkLabel.value() )

# define barrel and endcap benchmarks for the calo case: 

caloBarrel = caloTauBenchmarkGeneric.clone()
caloBarrel.minEta = -1
caloBarrel.maxEta = 1.4
caloBarrel.BenchmarkLabel = cms.string('%s_barrel' % caloBarrel.BenchmarkLabel.value() )


caloEndcap = caloTauBenchmarkGeneric.clone()
caloEndcap.minEta = 1.6
caloEndcap.maxEta = 2.8
caloEndcap.BenchmarkLabel = cms.string('%s_endcap' % caloEndcap.BenchmarkLabel.value() )




tauBenchmarkGeneric = cms.Sequence(
    trueTaus + 
    pfTauBenchmarkGeneric +
    pfBarrel +
    pfEndcap +
    caloTauBenchmarkGeneric +
    caloBarrel +
    caloEndcap 
    )
