import FWCore.ParameterSet.Config as cms

process = cms.Process("PostMixerRec")
process.load("Configuration.StandardSequences.Services_cff")
process.load('FWCore/MessageService/MessageLogger_cfi')
process.load('Configuration/StandardSequences/GeometryPilot2_cff')
process.load('Configuration/StandardSequences/MagneticField_38T_cff')
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.load("Configuration.EventContent.EventContent_cff")

process.load("SimGeneral.DataMixingModule.ReconstructionDM_cff")

process.configurationMetadata = cms.untracked.PSet(
        version = cms.untracked.string('$Revision: 1.1 $'),
            annotation = cms.untracked.string('datamixertest nevts:500'),
            name = cms.untracked.string('PyReleaseValidation')
        )

process.options = cms.untracked.PSet(
        Rethrow = cms.untracked.vstring('ProductNotFound')
        )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(500)
)
process.source = cms.Source("PoolSource",
#    catalog = cms.untracked.string('PoolFileCatalog.xml'),
#    fileNames = cms.untracked.vstring('file:/uscms_data/d1/mikeh/CRAFT_on_SingleNu_Digis.root')
     fileNames = cms.untracked.vstring('file:/uscms_data/d1/mikeh/CRAFT_on_QCD_NewHcal_Digis_v4a.root')
#     /uscms_data/d1/mikeh/CRAFT_on_SingleNu_Digis.root')
)


process.csc2DRecHits.stripDigiTag = 'mix:MuonCSCStripDigisDM'
process.csc2DRecHits.wireDigiTag = 'mix:MuonCSCWireDigisDM'
process.dt1DRecHits.dtDigiLabel = 'mix:muonDTDigisDM'
process.rpcRecHits.rpcDigiLabel = 'mix:muonRPCDigisDM'
# Calo - using RecHits here
process.ecalWeightUncalibRecHit.EBdigiCollection = cms.InputTag("mix","EBDigiCollectionDM")
process.ecalWeightUncalibRecHit.EEdigiCollection = cms.InputTag("mix","EEDigiCollectionDM")
process.ecalPreshowerRecHit.ESdigiCollection = cms.InputTag("mix","ESDigiCollectionDM")
#
#process.hbhereco.digiLabel =  cms.InputTag("mix:HBHEDigiCollection")
#process.horeco.digiLabel =  cms.InputTag("mix:HODigiCollection")
#process.hfreco.digiLabel =  cms.InputTag("mix:HFDigiCollection")
process.hbhereco.digiLabel =  cms.InputTag("mix")
process.horeco.digiLabel =  cms.InputTag("mix")
process.hfreco.digiLabel =  cms.InputTag("mix")
process.zdcreco.digiLabel = cms.InputTag("mix")

# Tracker
#siStripClusters.DigiProducersList???  { 'mix','SiStripDigisDM'}
process.siPixelClusters.src = 'mix:siPixelDigisDM'
process.siStripClusters.DigiProducersList = cms.VPSet(cms.PSet( DigiLabel = cms.string('siStripDigisDM'),
                                                                DigiProducer = cms.string('mix')))

#process.multi5x5PreshowerClusterShape.preshStripEnergyCut = cms.double(200.0) #turn off ES clusters?
#process.photons.scIslandEndcapProducer = cms.InputTag("correctedMulti5x5SuperClusters")

process.myoutput = cms.OutputModule("PoolOutputModule",
#    process.FEVTEventContent,
    outputCommands = process.FEVTDEBUGEventContent.outputCommands,                                
    fileName = cms.untracked.string('file:/uscms_data/d1/mikeh/CRAFT_on_QCD_NewHcal_RECO_v4a.root'),
    dataset = cms.untracked.PSet(
            dataTier = cms.untracked.string(''),
                    filterName = cms.untracked.string('')
                )                                
)

process.myoutput.outputCommands.append('keep *')
#process.myoutput.outputCommands.append('keep HFDataFramesSorted_*_*_*')
#process.myoutput.outputCommands.append('keep HODataFramesSorted_*_*_*')
#process.myoutput.outputCommands.append('keep EBDigiCollection_ecalDigis_*_*')
#process.myoutput.outputCommands.append('keep EEDigiCollection_ecalDigis_*_*')
#process.myoutput.outputCommands.append('keep ESDataFramesSorted_ecalPreshowerDigis_*_*')



# Other statements
process.GlobalTag.globaltag = 'STARTUP_30X::All'

process.p1 = cms.Path(process.reconstruction)
process.outpath = cms.EndPath(process.myoutput)

# Schedule definition
#process.schedule = cms.Schedule(process.p1,process.RECO)
