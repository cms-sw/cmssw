import FWCore.ParameterSet.Config as cms

from DQMOffline.PFTau.metBenchmark_cfi import metBenchmark
from DQMOffline.PFTau.metBenchmark_cfi import matchMetBenchmark

pfMetBenchmark = metBenchmark.clone()
pfMetBenchmark.InputCollection = 'pfMet'
pfMetBenchmark.BenchmarkLabel = 'pfMet'
pfMetBenchmark.mode = 2

caloMetBenchmark = metBenchmark.clone()
caloMetBenchmark.InputCollection = 'met'
caloMetBenchmark.BenchmarkLabel = 'met'
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
MatchCaloMetBenchmark.InputCollection = 'met'
MatchCaloMetBenchmark.MatchCollection = 'genMetTrue'
MatchCaloMetBenchmark.mode = 2
MatchCaloMetBenchmark.BenchmarkLabel = 'met'

metBenchmarkSequence = cms.Sequence( pfMetBenchmark+caloMetBenchmark+trueMetBenchmark+MatchPfMetBenchmark+MatchCaloMetBenchmark )
