import FWCore.ParameterSet.Config as cms
import os

runOnTheGrid = False
runOnTheIC1 = False

def customise(process):
        process._Process__name="FINALRECO"
        #process.Tracer = cms.Service("Tracer")
        process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )
        #process.options = cms.untracked.PSet( SkipEvent = cms.untracked.vstring('ProductNotFound') )
        
        module_name="overlay"
        process_name="OVERLAY"
        process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )
        
        process.dt1DRecHits.dtDigiLabel = cms.InputTag(module_name,"muonDTDigisDM")
        process.csc2DRecHits.wireDigiTag = cms.InputTag(module_name,"MuonCSCWireDigisDM")
        process.csc2DRecHits.stripDigiTag = cms.InputTag(module_name,"MuonCSCStripDigisDM")
        process.rpcRecHits.rpcDigiLabel = cms.InputTag(module_name,"muonRPCDigisDM")
        process.siPixelClusters.src = cms.InputTag(module_name,"siPixelDigisDM")

        process.ecalPreshowerRecHit.ESdigiCollection = cms.InputTag(module_name,"ESDigiCollectionDM",process_name)
        process.ecalGlobalUncalibRecHit.EBdigiCollection = cms.InputTag(module_name,"EBDigiCollectionDM",process_name)
        process.ecalGlobalUncalibRecHit.EEdigiCollection =  cms.InputTag(module_name,"EEDigiCollectionDM",process_name)

        #process.ecalDetIdToBeRecovered.ebSrFlagCollection=cms.InputTag(module_name)
        #process.ecalDetIdToBeRecovered.eeSrFlagCollection=cms.InputTag(module_name)    
        
        process.hcalnoise.digiCollName = cms.InputTag(module_name,"HBHEDigiCollectionDM")

        process.hbhereco.digiLabel = cms.InputTag(module_name,"HBHEDigiCollectionDM")
        process.hfreco.digiLabel = cms.InputTag(module_name,"HFDigiCollectionDM")
        process.horeco.digiLabel = cms.InputTag(module_name,"HODigiCollectionDM")
        process.zdcreco.digiLabel = cms.InputTag(module_name,"ZDCDigiCollectionDM")
        
        process.siStripZeroSuppression.RawDigiProducersList = cms.VInputTag()
        process.MeasurementTracker.inactiveStripDetectorLabels = cms.VInputTag()
        process.newMeasurementTracker.inactiveStripDetectorLabels = cms.VInputTag()
        process.fourthMeasurementTracker.inactiveStripDetectorLabels = cms.VInputTag()
        process.secMeasurementTracker.inactiveStripDetectorLabels = cms.VInputTag()
        process.fifthMeasurementTracker.inactiveStripDetectorLabels = cms.VInputTag()
        process.thMeasurementTracker.inactiveStripDetectorLabels = cms.VInputTag()
        process.load('Configuration/StandardSequences/Generator_cff')
        process.genParticles.abortOnUnknownPDGCode = False
        process.generation_step = cms.Path(cms.SequencePlaceholder("randomEngineStateProducer")+process.GeneInfo+process.genJetMET) # process.GeneInfo
        process.schedule.insert(0,process.generation_step)

        if runOnTheGrid:
                process.source.fileNames=cms.untracked.vstring(__FILE_NAMES__)
                process.source.skipEvents=cms.untracked.uint32(__SKIP_EVENTS__)
                process.maxEvents.input = cms.untracked.int32(__MAX_EVENTS__)
                process.output.fileName = cms.untracked.string("output.root")
        
        process.siStripClusters.DigiProducersList = cms.VInputTag(
                cms.InputTag(module_name,"siStripDigisZSDM"),
                cms.InputTag(module_name,"siStripDigisVRDM"),
                cms.InputTag(module_name,"siStripDigisPRDM"),
                cms.InputTag(module_name,"siStripDigisSMDM")
        )
                
        print process.dumpPython()  
        process.output.outputCommands = cms.untracked.vstring("keep *_*_*_*")
                                  
        return (process)        



