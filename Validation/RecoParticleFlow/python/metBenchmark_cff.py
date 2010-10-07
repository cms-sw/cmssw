import FWCore.ParameterSet.Config as cms

from DQMOffline.PFTau.metBenchmark_cfi import metBenchmark
from DQMOffline.PFTau.metBenchmark_cfi import matchMetBenchmark

########
# Cor Calo MET
from JetMETCorrections.Configuration.JetCorrectionServicesAllAlgos_cff import *
from JetMETCorrections.Type1MET.MetType1Corrections_cff import metJESCorAK5CaloJet

metMuonJESCorAK5 = metJESCorAK5CaloJet.clone()
metMuonJESCorAK5.inputUncorMetLabel = "corMetGlobalMuons"

metCorSequence = cms.Sequence(metMuonJESCorAK5)
#########


pfMetBenchmark = metBenchmark.clone()
pfMetBenchmark.InputCollection = 'pfMet'
pfMetBenchmark.BenchmarkLabel = 'pfMet'
pfMetBenchmark.mode = 2

caloMetBenchmark = metBenchmark.clone()
#caloMetBenchmark.InputCollection = 'met'
#caloMetBenchmark.BenchmarkLabel = 'met'
caloMetBenchmark.InputCollection = 'metMuonJESCorAK5'
caloMetBenchmark.BenchmarkLabel = 'metMuonJESCorAK5'
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
MatchCaloMetBenchmark.InputCollection = 'metMuonJESCorAK5'
MatchCaloMetBenchmark.MatchCollection = 'genMetTrue'
MatchCaloMetBenchmark.mode = 2
#MatchCaloMetBenchmark.BenchmarkLabel = 'met'
MatchCaloMetBenchmark.BenchmarkLabel = 'metMuonJESCorAK5'

metBenchmarkSequence = cms.Sequence( metCorSequence+pfMetBenchmark+caloMetBenchmark+trueMetBenchmark+MatchPfMetBenchmark+MatchCaloMetBenchmark )
