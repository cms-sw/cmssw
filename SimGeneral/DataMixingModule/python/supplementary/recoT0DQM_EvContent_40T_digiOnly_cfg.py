
import FWCore.ParameterSet.Config as cms

process = cms.Process("Rec")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("CondCore.DBCommon.CondDBSetup_cfi")


process.maxEvents = cms.untracked.PSet(  input = cms.untracked.int32(500) )
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
'/store/data/Commissioning08/HcalHPDNoise/RAW/v1/000/068/021/E293B383-5DA7-DD11-9621-001D09F2437B.root'
#'/store/data/Commissioning08/HcalHPDNoise/RAW/v1/000/068/021/AAFD01F4-67A7-DD11-8EAE-000423D98A44.root'
# '/store/data/Commissioning08/Cosmics/RAW/v1/000/070/664/20A57DB9-79AF-DD11-95AB-000423D99BF2.root'
#        '/store/data/Commissioning08/Cosmics/RAW/v1/000/067/838/006945C8-40A5-DD11-BD7E-001617DBD556.root'
#        '/store/data/Commissioning08/Calo/RAW/v1/000/067/838/001DBF26-91A5-DD11-BA34-000423D98AF0.root'
#     '/store/data/Commissioning08/Cosmics/RAW/v1/000/066/480/3A542A1C-609B-DD11-8D8A-000423D6CA72.root' 
    )
)


process.dump = cms.EDAnalyzer("EventContentAnalyzer")


# output module
#
process.load("Configuration.EventContent.EventContentCosmics_cff")

process.FEVT = cms.OutputModule("PoolOutputModule",
    process.FEVTEventContent,
    dataset = cms.untracked.PSet(dataTier = cms.untracked.string('RECO')),
    fileName = cms.untracked.string('file:/uscms_data/d1/mikeh/promptRecoHCalNoise.root')
)

process.FEVT.outputCommands.append('keep CSCDetIdCSCALCTDigiMuonDigiCollection_muonCSCDigis_MuonCSCALCTDigi_*')
process.FEVT.outputCommands.append('keep CSCDetIdCSCCLCTDigiMuonDigiCollection_muonCSCDigis_MuonCSCCLCTDigi_*')
process.FEVT.outputCommands.append('keep CSCDetIdCSCComparatorDigiMuonDigiCollection_muonCSCDigis_MuonCSCComparatorDigi_*')
process.FEVT.outputCommands.append('keep CSCDetIdCSCCorrelatedLCTDigiMuonDigiCollection_csctfDigis_*_*')
process.FEVT.outputCommands.append('keep CSCDetIdCSCCorrelatedLCTDigiMuonDigiCollection_muonCSCDigis_MuonCSCCorrelatedLCTDigi_*')
process.FEVT.outputCommands.append('keep CSCDetIdCSCRPCDigiMuonDigiCollection_muonCSCDigis_MuonCSCRPCDigi_*')
process.FEVT.outputCommands.append('keep CSCDetIdCSCStripDigiMuonDigiCollection_muonCSCDigis_MuonCSCStripDigi_*')
process.FEVT.outputCommands.append('keep CSCDetIdCSCWireDigiMuonDigiCollection_muonCSCDigis_MuonCSCWireDigi_*')
process.FEVT.outputCommands.append('keep cscL1TrackCSCDetIdCSCCorrelatedLCTDigiMuonDigiCollectionstdpairs_csctfDigis_*_*')
process.FEVT.outputCommands.append('keep DTChamberIdDTLocalTriggerMuonDigiCollection_muonDTDigis_*_*')
process.FEVT.outputCommands.append('keep DTLayerIdDTDigiMuonDigiCollection_muonDTDigis_*_*')
process.FEVT.outputCommands.append('keep intL1CSCSPStatusDigisstdpair_csctfDigis_*_*')
process.FEVT.outputCommands.append('keep L1MuDTChambPhContainer_dttfDigis_*_*')
process.FEVT.outputCommands.append('keep L1MuDTChambThContainer_dttfDigis_*_*')
process.FEVT.outputCommands.append('keep L1MuDTTrackContainer_dttfDigis_DATA_*')
process.FEVT.outputCommands.append('keep PixelDigiedmDetSetVector_siPixelDigis_*_*')
process.FEVT.outputCommands.append('keep RPCDetIdRPCDigiMuonDigiCollection_muonRPCDigis_*_*')
process.FEVT.outputCommands.append('keep HBHEDataFramesSorted_hcalDigis_*_*')
process.FEVT.outputCommands.append('keep HFDataFramesSorted_hcalDigis_*_*')
process.FEVT.outputCommands.append('keep HODataFramesSorted_hcalDigis_*_*')
process.FEVT.outputCommands.append('keep EBDigiCollection_ecalDigis_*_*')
process.FEVT.outputCommands.append('keep EEDigiCollection_ecalDigis_*_*')
process.FEVT.outputCommands.append('keep ESDataFramesSorted_ecalPreshowerDigis_*_*')

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.1 $'),
    name = cms.untracked.string('$Source: /cvs_server/repositories/CMSSW/CMSSW/SimGeneral/DataMixingModule/python/recoT0DQM_EvContent_40T_digiOnly_cfg.py,v $'),
    annotation = cms.untracked.string('CRUZET Prompt Reco with Mag field at 3.8T')
)
process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) ) ## default is false


# Conditions (Global Tag is used here):
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.GlobalTag.connect = "frontier://PromptProd/CMS_COND_21X_GLOBALTAG"
process.GlobalTag.connect = "frontier://FrontierInt/CMS_COND_30X_GLOBALTAG"
process.GlobalTag.globaltag = "CRAFT_30X::All"
process.prefer("GlobalTag")

# Magnetic fiuld: force mag field to be 3.8 tesla
process.load("Configuration.StandardSequences.MagneticField_38T_cff")

#Geometry
process.load("Configuration.StandardSequences.Geometry_cff")

# Real data raw to digi
process.load("Configuration.StandardSequences.RawToDigi_Data_cff")

# reconstruction sequence for Cosmics (local copy)
process.load("SimGeneral.DataMixingModule.ReconstructionLocalCosmics_cff")

# offline DQM
#process.load("DQMOffline.Configuration.DQMOfflineCosmics_cff")
#process.load("DQMServices.Components.MEtoEDMConverter_cff")

#L1 trigger validation
#process.load("L1Trigger.HardwareValidation.L1HardwareValidation_cff")
process.load("L1Trigger.Configuration.L1Config_cff")
process.load("L1TriggerConfig.CSCTFConfigProducers.CSCTFConfigProducer_cfi")
process.load("L1TriggerConfig.CSCTFConfigProducers.L1MuCSCTFConfigurationRcdSrc_cfi")

#process.roadSearchSeedsP5.MaxNumberOfStripClusters = 100


#Paths
process.allPath = cms.Path( process.RawToDigi_woGCT )

process.outpath = cms.EndPath(process.FEVT)
