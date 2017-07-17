import FWCore.ParameterSet.Config as cms
import muonCustoms

def customise_csc_PostLS1(process):
    process=muonCustoms.customise_csc_PostLS1(process)

    if hasattr(process,'simCscTriggerPrimitiveDigis'):
        process.simCscTriggerPrimitiveDigis.CSCComparatorDigiProducer = cms.InputTag( 'mixData', 'MuonCSCComparatorDigisDM')
        process.simCscTriggerPrimitiveDigis.CSCWireDigiProducer = cms.InputTag( 'mixData', 'MuonCSCWireDigisDM')

    # Switch input for CSCRecHitD to  s i m u l a t e d (and Mixed!)  digis
    if hasattr(process,'csc2DRecHits'):    
        process.csc2DRecHits.wireDigiTag  = cms.InputTag("mixData", "MuonCSCWireDigisDM")
        process.csc2DRecHits.stripDigiTag = cms.InputTag("mixData", "MuonCSCStripDigisDM")

    if hasattr(process,'hltCsc2DRecHits'):
        process.hltCsc2DRecHits.wireDigiTag  = cms.InputTag("mixData","MuonCSCWireDigisDM")
        process.hltCsc2DRecHits.stripDigiTag = cms.InputTag("mixData","MuonCSCStripDigisDM")
    
    return process
