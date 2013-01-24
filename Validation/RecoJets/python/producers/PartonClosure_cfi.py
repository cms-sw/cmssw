import FWCore.ParameterSet.Config as cms

#----------------------------------------------
# module to for parton closure
#----------------------------------------------

partonClosure = cms.EDFilter("PartonClosure",
## input
  recs = cms.InputTag("iterativeCone5CaloJets"),
  refs = cms.InputTag("genParticles"),
                             
## histogram directory
  hist = cms.string('partonClosure.hist'),

## pt bins for calibration
  type = cms.int32 (  0),   # comparison type can be (0:ratio/1:rel diff)
  bins = cms.int32 (200),
  min  = cms.double(0.0),                             
  max  = cms.double(2.0),
                             
  binsPt  = cms.vdouble(30.0, 50.0, 70.0, 100.0, 130.0, 160.0, 200.0, 250.0, 300.0),
  binsEta = cms.vdouble(-3.0, -1.4, 0.0, 1.4, 3.0),
                             
## configure visible range and reconstruction quality
  maxDR     = cms.double(  0.3),
  minPtRef  = cms.double(  5.0),
  maxPtRef  = cms.double(999.0),
  minPtRec  = cms.double( 15.0),
  maxPtRec  = cms.double(999.0),
  minEtaRef = cms.double( -3.0),                             
  maxEtaRef = cms.double(  3.0),
  minEtaRec = cms.double( -5.0),
  maxEtaRec = cms.double(  5.0),
                             
  minEmfCaloJet = cms.double( 0.0),
  maxEmfCaloJet = cms.double( 1.0),
  status        = cms.int32 (   3),
  partons       = cms.vint32(1, 2, 3, 4, 5)
# partons       = cms.vint32(5)
)


