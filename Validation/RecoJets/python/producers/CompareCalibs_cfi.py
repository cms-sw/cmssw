import FWCore.ParameterSet.Config as cms

#----------------------------------------------
# module to for gen jet closure
#----------------------------------------------

compareCalibs = cms.EDFilter("CompareCalibs",
## input
    recs = cms.InputTag("uhhCaliIterativeCone5"),
    refs = cms.InputTag("iterativeCone5CaloJets"),
                             
## histogram directory
    hist = cms.string('compareCalibs.hist'),
                             
## pt bins for calibration
    type = cms.int32 (  0),
    bins = cms.int32 (200),
    min  = cms.double(0.0),
    max  = cms.double(2.0),
    binsPt  = cms.vdouble(30.0, 50.0, 70.0, 100.0, 130.0, 160.0, 200.0, 250.0, 300.0),
    binsEta = cms.vdouble(-3.0, -1.4,  0.0,   1.4,   3.0),                             

## configure visible range and reconstruction quality                                 
    maxDR         = cms.double(  0.3),
    minPtRef      = cms.double( 15.0),
    maxPtRef      = cms.double(999.0),
    minPtRec      = cms.double( 15.0),
    maxPtRec      = cms.double(999.0),
    minEtaRef     = cms.double( -5.0),
    maxEtaRef     = cms.double(  5.0),
    minEtaRec     = cms.double( -5.0),
    maxEtaRec     = cms.double(  5.0),                             
    minEmfCaloJet = cms.double( 0.05),
    maxEmfCaloJet = cms.double( 0.95)
)


