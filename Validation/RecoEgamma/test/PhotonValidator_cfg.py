import FWCore.ParameterSet.Config as cms

process = cms.Process("TestPhotonValidator")

process.load("RecoEcal.EgammaClusterProducers.geometryForClustering_cff")
process.load("Configuration.StandardSequences.MagneticField_40T_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")
process.load("RecoTracker.GeometryESProducer.TrackerRecoGeometryESProducer_cfi")
process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
process.load("RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi")
process.load("SimGeneral.TrackingAnalysis.trackingParticles_cfi")
process.load("SimTracker.TrackAssociation.TrackAssociatorByHits_cfi")
process.load("SimGeneral.MixingModule.mixNoPU_cfi")
process.load("DQMServices.Components.MEtoEDMConverter_cfi")
process.load("Validation.RecoEgamma.photonValidator_cfi")

process.DQMStore = cms.Service("DQMStore");



#  include "DQMServices/Components/data/MessageLogger.cfi"
#  service = LoadAllDictionaries {}



process.maxEvents = cms.untracked.PSet(
 input = cms.untracked.int32(5000)
)


from Validation.RecoEgamma.photonValidator_cfi import *
photonValidation.OutputFileName = 'PhotonValidationRelVal210_H130GGgluonfusion.root'



process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(

# official RelVal 210 single Photons pt=10GeV
#'/store/relval/CMSSW_2_1_0/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V5_v1/0000/16A83B09-3660-DD11-80F4-000423D6CAF2.root',
#'/store/relval/CMSSW_2_1_0/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V5_v1/0000/8EC7DD51-0E60-DD11-B142-000423D6BA18.root',
#'/store/relval/CMSSW_2_1_0/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V5_v1/0001/A053CB4F-A760-DD11-80A0-0016177CA7A0.root'

# official RelVal 210 single Photons pt=35GeV
#'/store/relval/CMSSW_2_1_0/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V5_v1/0000/A67C9E97-1760-DD11-9999-001617E30D40.root',
#'/store/relval/CMSSW_2_1_0/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V5_v1/0000/AA302489-1560-DD11-B07F-001617DBD224.root',
#'/store/relval/CMSSW_2_1_0/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V5_v1/0000/EA15BBD9-1760-DD11-85C3-000423D9989E.root',
#'/store/relval/CMSSW_2_1_0/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V5_v1/0001/86442119-A760-DD11-A69F-001617C3B70E.root'  

# Official RelVal 210 H130GGgluonfusion

'/store/relval/CMSSW_2_1_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0000/626823C8-0461-DD11-AEAC-003048767E4B.root',
'/store/relval/CMSSW_2_1_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0001/1607149A-0D61-DD11-AD4C-0018F3D0968E.root',
'/store/relval/CMSSW_2_1_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0001/209DDAC6-0661-DD11-B2E1-00304875A9D7.root',
'/store/relval/CMSSW_2_1_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0001/346C64E6-0661-DD11-8EC3-001A92971B78.root',
'/store/relval/CMSSW_2_1_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0001/3A27EE79-0561-DD11-8AA4-0018F3D09612.root',
'/store/relval/CMSSW_2_1_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0001/3CAB0D11-0A61-DD11-95CD-001A928116BC.root',
'/store/relval/CMSSW_2_1_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0001/42EE600F-0661-DD11-A409-001A92971ADC.root',
'/store/relval/CMSSW_2_1_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0001/4427A171-0561-DD11-A792-00304875AAE3.root',
'/store/relval/CMSSW_2_1_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0001/46DABCA4-0B61-DD11-8499-001A92971B04.root',
'/store/relval/CMSSW_2_1_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0001/4C7713A2-0561-DD11-A7A2-001A92971B68.root',
'/store/relval/CMSSW_2_1_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0001/4E11E857-0C61-DD11-B5B6-001A92971B54.root',
'/store/relval/CMSSW_2_1_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0001/5A08B5DF-0961-DD11-A988-00304872538F.root',
'/store/relval/CMSSW_2_1_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0001/5C233E22-0D61-DD11-8E13-003048754D09.root',
'/store/relval/CMSSW_2_1_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0001/605FD004-1261-DD11-8495-001731230FC9.root',
'/store/relval/CMSSW_2_1_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0001/6E59C4E6-0661-DD11-86E2-0018F3D09682.root',
'/store/relval/CMSSW_2_1_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0001/8A1704AD-0D61-DD11-A4D2-003048723C0B.root',
'/store/relval/CMSSW_2_1_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0001/8A3C7299-0B61-DD11-A0E8-0018F3D096E6.root',
'/store/relval/CMSSW_2_1_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0001/96377209-1261-DD11-9FF6-003048756275.root',
'/store/relval/CMSSW_2_1_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0001/966E0CDA-0661-DD11-8363-003048769DBB.root',
'/store/relval/CMSSW_2_1_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0001/96D201C2-0661-DD11-BEEB-001A928116EA.root',
'/store/relval/CMSSW_2_1_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0001/9A795024-0A61-DD11-8DA8-001A92810AA6.root',
'/store/relval/CMSSW_2_1_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0001/9C4CC0A1-0B61-DD11-B6CF-001A92971B54.root',
'/store/relval/CMSSW_2_1_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0001/9E2DEE44-0C61-DD11-BD9A-0018F3D096E6.root',
'/store/relval/CMSSW_2_1_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0001/A627B4D9-0961-DD11-846B-003048726C93.root',
'/store/relval/CMSSW_2_1_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0001/ACA61FBA-0B61-DD11-AA2F-001A92810AA6.root',
'/store/relval/CMSSW_2_1_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0001/B06F0510-0861-DD11-91D9-001A928116C0.root',
'/store/relval/CMSSW_2_1_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0001/B2DD55A1-0B61-DD11-ABDD-0018F3D096D8.root',
'/store/relval/CMSSW_2_1_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0001/B2EDC404-0661-DD11-ACD8-001A92811744.root',
'/store/relval/CMSSW_2_1_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0001/D20B8AC6-0961-DD11-9DA5-0017312310E7.root',
'/store/relval/CMSSW_2_1_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0001/DA1C9DD6-0961-DD11-81C0-00304875AB5D.root',
'/store/relval/CMSSW_2_1_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0001/EC9B92DA-0661-DD11-A808-003048767DD9.root',
'/store/relval/CMSSW_2_1_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0001/EE465A60-0C61-DD11-9916-0018F3D0969A.root',
'/store/relval/CMSSW_2_1_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0001/EEEBC177-0B61-DD11-A842-001A92811734.root',
'/store/relval/CMSSW_2_1_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0001/F4104BF7-0961-DD11-8B8F-001731AF68BB.root',
'/store/relval/CMSSW_2_1_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0002/4E7534D5-6A61-DD11-B76C-001A92971B9A.root'



)

)



from SimTracker.TrackAssociation.TrackAssociatorByHits_cfi import *
import SimTracker.TrackAssociation.TrackAssociatorByHits_cfi
#TrackAssociatorByHits.AbsoluteNumberOfHits = True
#TrackAssociatorByHits.Cut_RecoToSim = 3
#TrackAssociatorByHits.Quality_SimToReco = 3
TrackAssociatorByHits.Cut_RecoToSim = 0.5
TrackAssociatorByHits.Quality_SimToReco = 0.5


process.FEVT = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring("keep *_MEtoEDMConverter_*_*"),
    fileName = cms.untracked.string('pippo.root')
)

process.p1 = cms.Path(process.photonValidation)
process.schedule = cms.Schedule(process.p1)



