import FWCore.ParameterSet.Config as cms

from DQMOffline.PFTau.metBenchmark_cfi import metBenchmark
from DQMOffline.PFTau.metBenchmark_cfi import matchMetBenchmark

########
# Cor Calo MET
from JetMETCorrections.Configuration.JetCorrectionServicesAllAlgos_cff import *
from JetMETCorrections.Type1MET.MetType1Corrections_cff import metJESCorAK4CaloJet

metMuonJESCorAK4 = metJESCorAK4CaloJet.clone()
metMuonJESCorAK4.inputUncorMetLabel = "caloMetM"

metCorSequence = cms.Sequence(metMuonJESCorAK4)
#########


pfMetBenchmark = metBenchmark.clone()
pfMetBenchmark.InputCollection = 'pfMet'
pfMetBenchmark.BenchmarkLabel = 'pfMet'
pfMetBenchmark.mode = 2

caloMetBenchmark = metBenchmark.clone()
#caloMetBenchmark.InputCollection = 'met'
#caloMetBenchmark.BenchmarkLabel = 'met'
caloMetBenchmark.InputCollection = 'metMuonJESCorAK4'
caloMetBenchmark.BenchmarkLabel = 'metMuonJESCorAK4'
caloMetBenchmark.mode = 2

trueMetBenchmark = metBenchmark.clone()
trueMetBenchmark.InputCollection = 'genMetTrue'
trueMetBenchmark.BenchmarkLabel = 'genMetTrue'
trueMetBenchmark.mode = 2

MatchPfMetBenchmark = matchMetBenchmark.clone()
MatchPfMetBenchmark.InputCollection = 'pfMet'
MatchPfMetBenchmark.MatchCollection = 'genMetTrue'
MatchPfMetBenchmark.mode = 2
MatchPfMetBenchmark.BenchmarkLabel = 'pfMet'

MatchCaloMetBenchmark = matchMetBenchmark.clone()
#MatchCaloMetBenchmark.InputCollection = 'met'
MatchCaloMetBenchmark.InputCollection = 'metMuonJESCorAK4'
MatchCaloMetBenchmark.MatchCollection = 'genMetTrue'
MatchCaloMetBenchmark.mode = 2
#MatchCaloMetBenchmark.BenchmarkLabel = 'met'
MatchCaloMetBenchmark.BenchmarkLabel = 'metMuonJESCorAK4'

UncorrCaloMetBenchmark = metBenchmark.clone()
UncorrCaloMetBenchmark.InputCollection = 'caloMetM'
UncorrCaloMetBenchmark.BenchmarkLabel = 'caloMetM'
UncorrCaloMetBenchmark.mode = 2

metBenchmarkSequence = cms.Sequence( metCorSequence+pfMetBenchmark+caloMetBenchmark+trueMetBenchmark+MatchPfMetBenchmark+MatchCaloMetBenchmark )
metBenchmarkSequenceData = cms.Sequence( metCorSequence+pfMetBenchmark+caloMetBenchmark+UncorrCaloMetBenchmark )
