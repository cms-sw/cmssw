import FWCore.ParameterSet.Config as cms
import os

# cuts: ---------------------------------------------------------------------

runOnTheGrid = False

                        
def customise(process):
        process._Process__name="OVERLAY"
        process.LoadAllDictionaries = cms.Service("LoadAllDictionaries")

        process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )
        #process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) , Rethrow = cms.untracked.vstring('ProductNotFound') )
        
        originalEventProcessName = "SELECTION"
        newPartialEventProcessName = "TAUSIMU"
        recHitCollection = "" # "assocHitsWithHCAL" # "" "assocHitsWithHCAL" "trueHits"
        #newPartialEventProcessName = originalEventProcessName
        settingForoverlayCompleteSecondEvent = True
  
        process.overlay = cms.EDProducer('EventMixingModule',
                #       ZS = ZeroSuppressed
                #       PR = ProcessedRaw
                #       VR = VirginRaw
                #       SM = ScopeMode
                #SiStripDigiInput1 = cms.InputTag("siStripDigis","ZeroSuppressed"),
                SiPixelDigiInput1 = cms.InputTag("siPixelDigis","",originalEventProcessName),
                #SiStripDigiInput1 = cms.VInputTag(cms.InputTag("siStripDigis","ZeroSuppressed",originalEventProcessName),cms.InputTag("siStripZeroSuppression","VirginRaw",originalEventProcessName),cms.InputTag("siStripZeroSuppression","ProcessedRaw",originalEventProcessName),cms.InputTag("siStripZeroSuppression","ScopeMode",originalEventProcessName)),
                SiStripDigiInput1 = cms.VInputTag(
                        cms.InputTag("siStripDigis","ZeroSuppressed",originalEventProcessName)
                        ,cms.InputTag("siStripZeroSuppression","VirginRaw",originalEventProcessName)
                        ,cms.InputTag("siStripZeroSuppression","ProcessedRaw",originalEventProcessName)
                        ,cms.InputTag("siStripZeroSuppression","ScopeMode",originalEventProcessName)
                ),              

                EBDigiInput1 = cms.InputTag("ecalDigis","ebDigis",originalEventProcessName),
                EEDigiInput1 = cms.InputTag("ecalDigis","eeDigis",originalEventProcessName),
                ESDigiInput1 = cms.InputTag("ecalPreshowerDigis","",originalEventProcessName),
        
                HBHEDigiInput1 = cms.InputTag("hcalDigis","",originalEventProcessName),
                HODigiInput1 = cms.InputTag("hcalDigis","",originalEventProcessName),
                HFDigiInput1 = cms.InputTag("hcalDigis","",originalEventProcessName),
                ZDCDigiInput1 = cms.InputTag("hcalDigis","",originalEventProcessName),
        
                DTDigiInput1 = cms.InputTag("muonDTDigis","",originalEventProcessName),
                RPCDigiInput1 = cms.InputTag("muonRPCDigis","",originalEventProcessName),
                CSCStripDigiInput1 = cms.InputTag("muonCSCDigis","MuonCSCStripDigi",originalEventProcessName),
                CSCWireDigiInput1 = cms.InputTag("muonCSCDigis","MuonCSCWireDigi",originalEventProcessName),
                CSCComparatorDigiInput1 = cms.InputTag("muonCSCDigis","MuonCSCComparatorDigi",originalEventProcessName),                        
        
                ##SiStripDigiInput2 = cms.InputTag("siStripDigis","ZeroSuppressed"),
                SiPixelDigiInput2 = cms.InputTag("siPixelDigis","",newPartialEventProcessName),
                #SiStripDigiInput2 = cms.VInputTag(cms.InputTag("siStripDigis","ZeroSuppressed",newPartialEventProcessName),cms.InputTag("siStripZeroSuppression","VirginRaw",newPartialEventProcessName),cms.InputTag("siStripZeroSuppression","ProcessedRaw",newPartialEventProcessName),cms.InputTag("siStripZeroSuppression","ScopeMode",newPartialEventProcessName)),

                #SiStripDigiInput2 = cms.VInputTag(cms.InputTag("siStripDigis","ZeroSuppressed",newPartialEventProcessName),cms.InputTag("siStripZeroSuppression","VirginRaw",newPartialEventProcessName),cms.InputTag("siStripZeroSuppression","ProcessedRaw",newPartialEventProcessName),cms.InputTag("siStripZeroSuppression","ScopeMode",newPartialEventProcessName)),              
                SiStripDigiInput2 = cms.VInputTag(
                        cms.InputTag("siStripDigis","ZeroSuppressed",newPartialEventProcessName)
                        ,cms.InputTag("siStripZeroSuppression","VirginRaw",process._Process__name)
                        ,cms.InputTag("siStripZeroSuppression","ProcessedRaw",process._Process__name)
                        ,cms.InputTag("siStripZeroSuppression","ScopeMode",process._Process__name)
                ),
                                
                EBDigiInput2 = cms.InputTag("ecalDigis","ebDigis",newPartialEventProcessName),
                EEDigiInput2 = cms.InputTag("ecalDigis","eeDigis",newPartialEventProcessName),
                ESDigiInput2 = cms.InputTag("ecalPreshowerDigis","",newPartialEventProcessName),
        
                HBHEDigiInput2 = cms.InputTag("hcalDigis","",newPartialEventProcessName),
                HODigiInput2 = cms.InputTag("hcalDigis","",newPartialEventProcessName),
                HFDigiInput2 = cms.InputTag("hcalDigis","",newPartialEventProcessName),
                ZDCDigiInput2 = cms.InputTag("hcalDigis","",newPartialEventProcessName),
        
                DTDigiInput2 = cms.InputTag("muonDTDigis","",newPartialEventProcessName),
                RPCDigiInput2 = cms.InputTag("muonRPCDigis","",newPartialEventProcessName),
                CSCStripDigiInput2 = cms.InputTag("muonCSCDigis","MuonCSCStripDigi",newPartialEventProcessName),
                CSCWireDigiInput2 = cms.InputTag("muonCSCDigis","MuonCSCWireDigi",newPartialEventProcessName),
                CSCComparatorDigiInput2 = cms.InputTag("muonCSCDigis","MuonCSCComparatorDigi",newPartialEventProcessName),
                 
                ##SiStripDigiInput2 = cms.InputTag("siStripDigis","ZeroSuppressed"),
                #SiPixelDigiInput2 = cms.InputTag("siPixelDigis","",originalEventProcessName),
                #SiStripDigiInput2 = cms.VInputTag(cms.InputTag("siStripDigis","ZeroSuppressed",originalEventProcessName),cms.InputTag("siStripZeroSuppression","VirginRaw",originalEventProcessName),cms.InputTag("siStripZeroSuppression","ProcessedRaw",originalEventProcessName),cms.InputTag("siStripZeroSuppression","ScopeMode",originalEventProcessName)),
                
                #EBDigiInput2 = cms.InputTag("ecalDigis","ebDigis",originalEventProcessName),
                #EEDigiInput2 = cms.InputTag("ecalDigis","eeDigis",originalEventProcessName),
                #ESDigiInput2 = cms.InputTag("ecalPreshowerDigis","",originalEventProcessName),
        
                #HBHEDigiInput2 = cms.InputTag("hcalDigis","",originalEventProcessName),
                #HODigiInput2 = cms.InputTag("hcalDigis","",originalEventProcessName),
                #HFDigiInput2 = cms.InputTag("hcalDigis","",originalEventProcessName),
                #ZDCDigiInput2 = cms.InputTag("hcalDigis","",originalEventProcessName),
        
                #DTDigiInput2 = cms.InputTag("muonDTDigis","",originalEventProcessName),
                #RPCDigiInput2 = cms.InputTag("muonRPCDigis","",originalEventProcessName),
                #CSCStripDigiInput2 = cms.InputTag("muonCSCDigis","MuonCSCStripDigi",originalEventProcessName),
                #CSCWireDigiInput2 = cms.InputTag("muonCSCDigis","MuonCSCWireDigi",originalEventProcessName),
        
                 
                DTDigiOutputProdInstanceLabel = cms.string('muonDTDigisDM'),
                RPCDigiOutputProdInstanceLabel = cms.string('muonRPCDigisDM'),
                CSCStripDigiOutputProdInstanceLabel = cms.string('MuonCSCStripDigisDM'),
                CSCWireDigiOutputProdInstanceLabel = cms.string('MuonCSCWireDigisDM'),
                CSCComparatorDigiOutputProdInstanceLabel = cms.string('MuonCSCComparatorDigisDM'),
                 
                SiPixelDigiOutputProdInstanceLabel = cms.string('siPixelDigisDM'),
                #SiStripDigiOutputProdInstanceLabel = cms.string('siStripDigisDM'),      
        
                HBHEDigiOutputProdInstanceLabel = cms.string('HBHEDigiCollectionDM'),
                HODigiOutputProdInstanceLabel = cms.string('HODigiCollectionDM'),
                HFDigiOutputProdInstanceLabel = cms.string('HFDigiCollectionDM'),
                ZDCDigiOutputProdInstanceLabel = cms.string('ZDCDigiCollectionDM'),
                EBDigiOutputProdInstanceLabel = cms.string('EBDigiCollectionDM'),
                EEDigiOutputProdInstanceLabel = cms.string('EEDigiCollectionDM'),
                ESDigiOutputProdInstanceLabel = cms.string('ESDigiCollectionDM'),

                SiStripDigiOutputProdInstanceLabel = cms.vstring('siStripDigisZSDM','siStripDigisVRDM','siStripDigisPRDM','siStripDigisSMDM'),
                 
                #hitCollection1 = cms.InputTag("selectMuonHits","",originalEventProcessName),
                #hitCollection2 = cms.InputTag("selectMuonHits","",originalEventProcessName),
                hitCollection1 = cms.InputTag("selectMuons",recHitCollection,"SELECTION"),      # GRIDJOB
                hitCollection2 = cms.InputTag("selectMuons",recHitCollection,"SELECTION"), # irrelevant, OVERLAYSTANDALONE

                overlayCompleteSecondEvent = cms.untracked.bool(settingForoverlayCompleteSecondEvent),
                Label = cms.string('')
        )

        process.load("RecoLocalTracker.SiStripZeroSuppression.DefaultAlgorithms_cff")
        process.siStripZeroSuppression = cms.EDFilter("SiStripZeroSuppression",
                Algorithms = process.DefaultAlgorithms,
                RawDigiProducersList = cms.VInputTag(
                        cms.InputTag('siStripDigis','VirginRaw'),
                        cms.InputTag('siStripDigis','ProcessedRaw'),
                        cms.InputTag('siStripDigis','ScopeMode')
                )
        )
        
        process.overlayPath = cms.Path(process.siStripZeroSuppression*process.overlay)

        process.output.SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('overlayPath'))
        process.schedule = cms.Schedule(process.overlayPath, process.L1simulation_step, process.out_step)

        if runOnTheGrid:
                process.source.fileNames=cms.untracked.vstring(__FILE_NAMES__)
                process.source.skipEvents=cms.untracked.uint32(__SKIP_EVENTS__)
                process.maxEvents.input = cms.untracked.int32(__MAX_EVENTS__)
                process.output.fileName = cms.untracked.string("output.root")

        #process.output.SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('filterNumHepMCEventsPath'))
        
        process.output.outputCommands = cms.untracked.vstring(
               "drop *_*_*_*","keep edmHepMCProduct_*_*_*",
               "keep *_*_*_OVERLAY",
               "drop *_siStripZeroSuppression_*_OVERLAY",
               "keep uints_*_*_*",
               "keep FEDRawDataCollection_*_*_*"
        )
        print process.schedule
        return(process)

