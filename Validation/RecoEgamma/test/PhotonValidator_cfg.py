import FWCore.ParameterSet.Config as cms

process = cms.Process("TestPhotonValidator")
process.load('Configuration/StandardSequences/GeometryPilot2_cff')
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
#process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")
process.load("RecoTracker.GeometryESProducer.TrackerRecoGeometryESProducer_cfi")
process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
process.load("RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi")
process.load("SimGeneral.TrackingAnalysis.trackingParticles_cfi")
process.load("SimTracker.TrackAssociation.TrackAssociatorByHits_cfi")
process.load("SimGeneral.MixingModule.mixNoPU_cfi")
process.load("TrackingTools.TransientTrack.TransientTrackBuilder_cfi")
process.load("DQMServices.Components.MEtoEDMConverter_cfi")
process.load("Validation.RecoEgamma.photonValidator_cfi")
process.load("Validation.RecoEgamma.tpSelection_cfi")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'IDEAL_31X::All'


process.DQMStore = cms.Service("DQMStore");



#  include "DQMServices/Components/data/MessageLogger.cfi"
#  service = LoadAllDictionaries {}



process.maxEvents = cms.untracked.PSet(
# input = cms.untracked.int32(3000)
)



from Validation.RecoEgamma.photonValidator_cfi import *
photonValidation.OutputFileName = 'PhotonValidationRelVal310pre10_SingleGammaPt10reRecoNoFid.root'
#photonValidation.OutputFileName = 'PhotonValidationRelVal310pre10_SingleGammaPt35_reRecoNoFid.root'
#photonValidation.OutputFileName = 'PhotonValidationRelVal310pre10_SingleGammaFlatPt10_100.root'
#photonValidation.OutputFileName = 'PhotonValidationRelVal310pre10_QCD_Pt_50_80.root'
#photonValidation.OutputFileName = 'PhotonValidationRelVal310pre10_GammaJets_Pt_80_120.root'
#photonValidation.OutputFileName = 'PhotonValidationRelVal310pre10_H130GGgluonfusion.root'

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(

'file:/tmp/nancy/CMSSW_3_1_0_pre10/src/RecoEgamma/EgammaPhotonProducers/SingleGammaPt10_RAW2DIGI_RECO.root'


#        'file:/tmp/nancy/CMSSW_3_1_0_pre10/src/RecoEgamma/EgammaPhotonProducers/SingleGammaPt35_RAW2DIGI_RECO_1.root',
#        'file:/tmp/nancy/CMSSW_3_1_0_pre10/src/RecoEgamma/EgammaPhotonProducers/SingleGammaPt35_RAW2DIGI_RECO_2.root'
    
        
# official RelVal 310pre10 single Photons pt=10GeV

# '/store/relval/CMSSW_3_1_0_pre10/RelValSingleGammaPt10/GEN-SIM-RECO/IDEAL_31X_v1/0009/2616F5E5-0558-DE11-9690-0019B9F7312C.root',
# '/store/relval/CMSSW_3_1_0_pre10/RelValSingleGammaPt10/GEN-SIM-RECO/IDEAL_31X_v1/0008/3047114C-5D57-DE11-9DFF-001D09F242EF.root'

# official RelVal 310pre10 single Photons pt=35GeV            
#        '/store/relval/CMSSW_3_1_0_pre10/RelValSingleGammaPt35/GEN-SIM-RECO/IDEAL_31X_v1/0008/E60958D5-0458-DE11-B0B7-000423D98E30.root',
#        '/store/relval/CMSSW_3_1_0_pre10/RelValSingleGammaPt35/GEN-SIM-RECO/IDEAL_31X_v1/0008/C0188C68-4157-DE11-AE7E-001D09F250AF.root'



# official RelVal 310pre10 RelValH130GGgluonfusion
#        '/store/relval/CMSSW_3_1_0_pre10/RelValH130GGgluonfusion/GEN-SIM-RECO/STARTUP_31X_v1/0008/F8A6F666-7657-DE11-A787-001617E30D12.root',
#        '/store/relval/CMSSW_3_1_0_pre10/RelValH130GGgluonfusion/GEN-SIM-RECO/STARTUP_31X_v1/0008/F6A18A14-6E57-DE11-B30C-001D09F28C1E.root',
#        '/store/relval/CMSSW_3_1_0_pre10/RelValH130GGgluonfusion/GEN-SIM-RECO/STARTUP_31X_v1/0008/DC5ABCCF-0458-DE11-A9B7-001D09F2983F.root',
#        '/store/relval/CMSSW_3_1_0_pre10/RelValH130GGgluonfusion/GEN-SIM-RECO/STARTUP_31X_v1/0008/B8DAC6B9-7357-DE11-ABF4-000423D992DC.root'
        
# official RelVal 310pre10 GammaJets_Pt_80_120



    ),
                            
                            
    secondaryFileNames = cms.untracked.vstring(

# official RelVal 310pre10 single Photons pt=10GeV    
#        '/store/relval/CMSSW_3_1_0_pre10/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0008/DEF9D1CD-0458-DE11-A0E0-001D09F23C73.root',
#        '/store/relval/CMSSW_3_1_0_pre10/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0008/B21BA97F-5957-DE11-AE5D-001D09F241F0.root',
#        '/store/relval/CMSSW_3_1_0_pre10/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0008/841E75D6-5C57-DE11-945B-001D09F24F1F.root'

# official RelVal 310pre10 single Photons pt=35GeV
#       '/store/relval/CMSSW_3_1_0_pre10/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0008/9C1E04D7-0458-DE11-9CC3-001D09F2AD84.root',
#        '/store/relval/CMSSW_3_1_0_pre10/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0008/88BEEB58-4257-DE11-8DAF-001D09F252F3.root',
#        '/store/relval/CMSSW_3_1_0_pre10/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0008/4CF9B8C8-4257-DE11-A09D-001D09F24493.root'

      
# official RelVal 310pre10 RelValH130GGgluonfusion

#        '/store/relval/CMSSW_3_1_0_pre10/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0008/E8FC5B3A-6B57-DE11-B61B-001617DBD556.root',
#        '/store/relval/CMSSW_3_1_0_pre10/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0008/CAECC52F-7657-DE11-8D61-001D09F28E80.root',
#        '/store/relval/CMSSW_3_1_0_pre10/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0008/C882525A-7657-DE11-9B97-001D09F24047.root',
#        '/store/relval/CMSSW_3_1_0_pre10/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0008/C6826BA0-7357-DE11-8C1E-001D09F29524.root',
#        '/store/relval/CMSSW_3_1_0_pre10/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0008/C4CE2082-6F57-DE11-8A4F-001D09F29533.root',
#        '/store/relval/CMSSW_3_1_0_pre10/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0008/C4AD2E24-7757-DE11-A021-001D09F24489.root',
#        '/store/relval/CMSSW_3_1_0_pre10/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0008/80B9BF85-6257-DE11-9927-001617DBD556.root',
#        '/store/relval/CMSSW_3_1_0_pre10/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0008/7CAD8F26-7557-DE11-9CDE-001D09F23A6B.root',
#        '/store/relval/CMSSW_3_1_0_pre10/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0008/58915F08-7757-DE11-8D46-000423D6B48C.root',
#        '/store/relval/CMSSW_3_1_0_pre10/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0008/3CDBDC70-6D57-DE11-8E19-001D09F291D2.root',
#        '/store/relval/CMSSW_3_1_0_pre10/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0008/0C8AD1BD-7057-DE11-9672-000423D98950.root',
#        '/store/relval/CMSSW_3_1_0_pre10/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0008/0AF4BEA2-7257-DE11-AEEC-001D09F24489.root',
#        '/store/relval/CMSSW_3_1_0_pre10/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0008/00C92E25-0558-DE11-9B2F-001D09F2960F.root'

# official RelVal 310pre10 GammaJets_Pt_80_120

    
    )
 )



from SimTracker.TrackAssociation.TrackAssociatorByHits_cfi import *
import SimTracker.TrackAssociation.TrackAssociatorByHits_cfi
#TrackAssociatorByHits.AbsoluteNumberOfHits = True
#TrackAssociatorByHits.Cut_RecoToSim = 3
#TrackAssociatorByHits.Quality_SimToReco = 3
TrackAssociatorByHits.Cut_RecoToSim = 0.5
TrackAssociatorByHits.Quality_SimToReco = 0.5



## For single gamma fla pt =10-150
#photonValidation.minPhoEtCut = 10
#photonValidation.eMax  = 300
#photonValidation.etMax = 30
#photonValidation.etScale = 0.10
#photonValidation.signal = True

## For single gamma pt =10
photonValidation.minPhoEtCut = 10
photonValidation.eMax  = 100
photonValidation.etMax = 50
photonValidation.etScale = 0.20
photonValidation.signal = True

## For single gamma pt = 35
#photonValidation.minPhoEtCut = 35
#photonValidation.eMax  = 300
#photonValidation.etMax = 50
#photonValidation.etScale = 0.20
#photonValidation.signal = True

## For gam Jet and higgs
#photonValidation.minPhoEtCut = 20
#photonValidation.eMax  = 500
#photonValidation.etMax = 500
#photonValidation.signal = True
## same for all

photonValidation.convTrackMinPtCut = 1.
photonValidation.useTP = True
photonValidation.rBin = 48
photonValidation.eoverpMin = 0.
photonValidation.eoverpMax = 5.


process.FEVT = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring("keep *_MEtoEDMConverter_*_*"),
    fileName = cms.untracked.string('pippo.root')
)


#process.p1 = cms.Path(process.photonValidation)
process.p1 = cms.Path(process.tpSelection*process.photonValidation)
#process.p1 = cms.Path(process.mix*process.trackingParticles*process.tpSelection*process.photonValidation)
process.schedule = cms.Schedule(process.p1)



