import FWCore.ParameterSet.Config as cms

from Validation.RecoParticleFlow.pfTauBenchmarkGeneric_cfi import pfTauBenchmarkGeneric

pfBarrel = pfTauBenchmarkGeneric.clone()
pfBarrel.minEta = -1
pfBarrel.maxEta = 1.4
pfBarrel.BenchmarkLabel = cms.string('%s_barrel' % pfBarrel.BenchmarkLabel.value() )


pfEndcap = pfTauBenchmarkGeneric.clone()
pfEndcap.minEta = 1.6
pfEndcap.maxEta = 2.8
pfEndcap.BenchmarkLabel = cms.string('%s_endcap' % pfEndcap.BenchmarkLabel.value() )

from Validation.RecoParticleFlow.caloTauBenchmarkGeneric_cfi import caloTauBenchmarkGeneric
from PhysicsTools.JetMCAlgos.TauGenJets_cfi import tauGenJets

caloBarrel = caloTauBenchmarkGeneric.clone()
caloBarrel.minEta = -1
caloBarrel.maxEta = 1.4
caloBarrel.BenchmarkLabel = cms.string('%s_barrel' % caloBarrel.BenchmarkLabel.value() )


caloEndcap = caloTauBenchmarkGeneric.clone()
caloEndcap.minEta = 1.6
caloEndcap.maxEta = 2.8
caloEndcap.BenchmarkLabel = cms.string('%s_endcap' % caloEndcap.BenchmarkLabel.value() )


tauBenchmarkGeneric = cms.Sequence(
    tauGenJets + 
    pfTauBenchmarkGeneric +
    pfBarrel +
    pfEndcap +
    caloTauBenchmarkGeneric +
    caloBarrel +
    caloEndcap 
    )
