import FWCore.ParameterSet.Config as cms

simHcalTTPDigis = cms.EDProducer("HcalTTPDigiProducer",
    HFDigiCollection = cms.InputTag( 'simHcalDigis' ),
    HFSoI            = cms.int32( 4 ),
    maskedChannels   = cms.vuint32( ),
    id               = cms.untracked.int32( 101 ),
    samples          = cms.int32( 5 ), 
    presamples       = cms.int32( 2 ), 
    fwAlgorithm      = cms.int32( 1 ), 
    defTT8           = cms.string("hits>=2"), 
    defTT9           = cms.string("hits>=2:hfp>=1:hfm>=1"), 
    defTT10          = cms.string("hits>=3:hfp>=1:hfm>=1"),
    defTTLocal       = cms.string("hits>=4"),
    iEtaMin          = cms.int32( 33 ), 
    iEtaMax          = cms.int32( 41 ),
    threshold        = cms.uint32( 2 )
)
