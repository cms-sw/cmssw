
# computing of isolation from deposits,
# as provided by Matthew LeBourgeois

from RecoEgamma.EgammaIsolationAlgos.egammaIsolationSequence_cff import *
from RecoEgamma.EgammaIsolationAlgos.egammaIsolationSequencePAT_cff import *

# create both IsoDeposits from the full collection and the reduced collection

eleIsoDepositEcalFromHitsFull = eleIsoDepositEcalFromHits.clone()
eleIsoDepositEcalFromHitsReduced = eleIsoDepositEcalFromHits.clone()

eleIsoDepositEcalFromHitsFull.ExtractorPSet.barrelEcalHits = cms.InputTag("ecalRecHit","EcalRecHitsEB")
eleIsoDepositEcalFromHitsFull.ExtractorPSet.endcapEcalHits = cms.InputTag("ecalRecHit","EcalRecHitsEE")
eleIsoDepositEcalFromHitsReduced.ExtractorPSet.barrelEcalHits = cms.InputTag("reducedEcalRecHitsEB")
eleIsoDepositEcalFromHitsReduced.ExtractorPSet.endcapEcalHits = cms.InputTag("reducedEcalRecHitsEE")

# clone the value map producers for each DR

eleIsoFromDepsTk03                           = eleIsoFromDepsTk.clone()
eleIsoFromDepsTk04                           = eleIsoFromDepsTk.clone()
eleIsoFromDepsEcalFromHitsByCrystalFull03    = eleIsoFromDepsEcalFromHitsByCrystal.clone()
eleIsoFromDepsEcalFromHitsByCrystalFull04    = eleIsoFromDepsEcalFromHitsByCrystal.clone()
eleIsoFromDepsEcalFromHitsByCrystalReduced03 = eleIsoFromDepsEcalFromHitsByCrystal.clone()
eleIsoFromDepsEcalFromHitsByCrystalReduced04 = eleIsoFromDepsEcalFromHitsByCrystal.clone()
eleIsoFromDepsHcalFromTowers03               = eleIsoFromDepsHcalFromTowers.clone()
eleIsoFromDepsHcalFromTowers04               = eleIsoFromDepsHcalFromTowers.clone()

# set the correct delta R
eleIsoFromDepsTk03.deposits[0].deltaR                         = 0.3
eleIsoFromDepsTk04.deposits[0].deltaR                         = 0.4
eleIsoFromDepsEcalFromHitsByCrystalFull03.deposits[0].deltaR  = 0.3
eleIsoFromDepsEcalFromHitsByCrystalFull04.deposits[0].deltaR  = 0.4
eleIsoFromDepsEcalFromHitsByCrystalReduced03.deposits[0].deltaR  = 0.3
eleIsoFromDepsEcalFromHitsByCrystalReduced04.deposits[0].deltaR  = 0.4
eleIsoFromDepsHcalFromTowers03.deposits[0].deltaR             = 0.3
eleIsoFromDepsHcalFromTowers04.deposits[0].deltaR             = 0.4

# change the source on the ECAL hits to make sure to get the reduced or full collection

eleIsoFromDepsEcalFromHitsByCrystalFull03.deposits[0].src = "eleIsoDepositEcalFromHitsFull"
eleIsoFromDepsEcalFromHitsByCrystalFull04.deposits[0].src = "eleIsoDepositEcalFromHitsFull"
eleIsoFromDepsEcalFromHitsByCrystalReduced03.deposits[0].src = "eleIsoDepositEcalFromHitsReduced"
eleIsoFromDepsEcalFromHitsByCrystalReduced04.deposits[0].src = "eleIsoDepositEcalFromHitsReduced"

# the sequence

electronIsoFromDeps = cms.Sequence(
  eleIsoDepositTk*eleIsoDepositEcalFromHitsFull*
  eleIsoDepositEcalFromHitsReduced*
  eleIsoDepositHcalFromTowers*      
  eleIsoFromDepsTk03*
  eleIsoFromDepsTk04*
  eleIsoFromDepsEcalFromHitsByCrystalFull03*
  eleIsoFromDepsEcalFromHitsByCrystalFull04*
  eleIsoFromDepsEcalFromHitsByCrystalReduced03*
  eleIsoFromDepsEcalFromHitsByCrystalReduced04*
  eleIsoFromDepsHcalFromTowers03*
  eleIsoFromDepsHcalFromTowers04
)