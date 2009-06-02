import FWCore.ParameterSet.Config as cms

process = cms.Process("TestPhotonValidator")

#process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")


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
# input = cms.untracked.int32(10000)
)



from Validation.RecoEgamma.photonValidator_cfi import *
#photonValidation.OutputFileName = 'PhotonValidationRelVal310pre8_SingleGammaPt35.root'
#photonValidation.OutputFileName = 'PhotonValidationRelVal310pre8_SingleGammaPt10.root'
#photonValidation.OutputFileName = 'PhotonValidationRelVal310pre8_SingleGammaFlatPt10_100.root'
#photonValidation.OutputFileName = 'PhotonValidationRelVal310pre8_QCD_Pt_50_80.root'
photonValidation.OutputFileName = 'PhotonValidationRelVal310pre8_GammaJets_Pt_80_120.root'
#photonValidation.OutputFileName = 'PhotonValidationRelVal310pre8_H130GGgluonfusion.root'

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(


        
        #Private production with 310pre8 of single photons flat pt
        
# official RelVal 310pre8 single Photons pt=10GeV    

# '/store/relval/CMSSW_3_1_0_pre8/RelValSingleGammaPt10/GEN-SIM-RECO/IDEAL_31X_v1/0006/C404B8D4-DB4D-DE11-B072-000423D6BA18.root'

# official RelVal 310pre8 single Photons pt=35GeV

# '/store/relval/CMSSW_3_1_0_pre8/RelValSingleGammaPt35/GEN-SIM-RECO/IDEAL_31X_v1/0006/58B2A3FD-DA4D-DE11-979A-001D09F251FE.root',
# '/store/relval/CMSSW_3_1_0_pre8/RelValSingleGammaPt35/GEN-SIM-RECO/IDEAL_31X_v1/0005/D0BD3013-514D-DE11-B32D-001D09F29146.root'     



# official RelVal 310pre8 RelValH130GGgluonfusion

#        '/store/relval/CMSSW_3_1_0_pre8/RelValH130GGgluonfusion/GEN-SIM-RECO/STARTUP_31X_v1/0006/84505318-A64D-DE11-8A91-0019DB2F3F9A.root',
#        '/store/relval/CMSSW_3_1_0_pre8/RelValH130GGgluonfusion/GEN-SIM-RECO/STARTUP_31X_v1/0006/4056579A-DB4D-DE11-817B-001D09F2924F.root',
#        '/store/relval/CMSSW_3_1_0_pre8/RelValH130GGgluonfusion/GEN-SIM-RECO/STARTUP_31X_v1/0006/22DAA72D-A64D-DE11-80C7-001D09F291D2.root',
#        '/store/relval/CMSSW_3_1_0_pre8/RelValH130GGgluonfusion/GEN-SIM-RECO/STARTUP_31X_v1/0005/C2B24B62-A54D-DE11-BF96-001D09F251E0.root'
    
# official RelVal 310pre8 GammaJets_Pt_80_120

        '/store/relval/CMSSW_3_1_0_pre8/RelValGammaJets_Pt_80_120/GEN-SIM-RECO/STARTUP_31X_v1/0006/FE7D57C4-DA4D-DE11-8FBE-001617DBCF6A.root',
        '/store/relval/CMSSW_3_1_0_pre8/RelValGammaJets_Pt_80_120/GEN-SIM-RECO/STARTUP_31X_v1/0005/F6776509-874D-DE11-9997-001D09F242EA.root',
        '/store/relval/CMSSW_3_1_0_pre8/RelValGammaJets_Pt_80_120/GEN-SIM-RECO/STARTUP_31X_v1/0005/F62F7BCC-874D-DE11-84CE-001D09F244DE.root',
        '/store/relval/CMSSW_3_1_0_pre8/RelValGammaJets_Pt_80_120/GEN-SIM-RECO/STARTUP_31X_v1/0005/AECAD622-884D-DE11-A702-001D09F2437B.root',
        '/store/relval/CMSSW_3_1_0_pre8/RelValGammaJets_Pt_80_120/GEN-SIM-RECO/STARTUP_31X_v1/0005/54163678-874D-DE11-87E4-001D09F2437B.root'

    ),
                            
                            
    secondaryFileNames = cms.untracked.vstring(

# official RelVal 310pre8 single Photons pt=10GeV    
#      '/store/relval/CMSSW_3_1_0_pre8/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0006/AC9F7329-B24D-DE11-9935-001D09F291D2.root',
#      '/store/relval/CMSSW_3_1_0_pre8/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0006/36A5DAFA-DA4D-DE11-9F3F-001D09F24691.root',
#      '/store/relval/CMSSW_3_1_0_pre8/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0006/146E2CEB-B54D-DE11-BD7B-001D09F242EA.root'

# official RelVal 310pre8 single Photons pt=35GeV

#        '/store/relval/CMSSW_3_1_0_pre8/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0006/0806C70D-DB4D-DE11-8ED8-001D09F253FC.root',
#        '/store/relval/CMSSW_3_1_0_pre8/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0005/A21DD565-504D-DE11-87A2-001D09F23A34.root',
#        '/store/relval/CMSSW_3_1_0_pre8/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0005/7C9ACDA3-4A4D-DE11-9B42-001D09F2527B.root'
      
# official RelVal 310pre8 RelValH130GGgluonfusion

#        '/store/relval/CMSSW_3_1_0_pre8/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0006/E8A5562E-A64D-DE11-BC9D-000423D98920.root',
#        '/store/relval/CMSSW_3_1_0_pre8/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0006/AE7288FD-DA4D-DE11-AAC0-001D09F2B30B.root',
#        '/store/relval/CMSSW_3_1_0_pre8/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0006/A031B21B-A64D-DE11-BE45-001D09F2AD7F.root',
#        '/store/relval/CMSSW_3_1_0_pre8/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0006/8ED0C13D-A64D-DE11-B66D-001D09F2525D.root',
#        '/store/relval/CMSSW_3_1_0_pre8/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0006/22372517-A64D-DE11-B15A-000423D990CC.root',
#        '/store/relval/CMSSW_3_1_0_pre8/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0006/0CD74F1C-A64D-DE11-A476-001D09F24FEC.root',
#        '/store/relval/CMSSW_3_1_0_pre8/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0005/D68ABCD8-A44D-DE11-8561-001D09F25109.root',
#        '/store/relval/CMSSW_3_1_0_pre8/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0005/AEE1A72C-A54D-DE11-A413-001D09F244BB.root',
#        '/store/relval/CMSSW_3_1_0_pre8/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0005/9ADFC4E3-A54D-DE11-A86A-001D09F25208.root',
#        '/store/relval/CMSSW_3_1_0_pre8/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0005/9442C19E-A54D-DE11-8A2A-000423D944F8.root',
#        '/store/relval/CMSSW_3_1_0_pre8/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0005/30AC8B42-A54D-DE11-8D72-0019DB2F3F9A.root',
#        '/store/relval/CMSSW_3_1_0_pre8/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0005/1EA492C3-A54D-DE11-A8C2-001D09F291D2.root',
#        '/store/relval/CMSSW_3_1_0_pre8/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0005/04388317-A64D-DE11-BBCA-001D09F24691.root'



# official RelVal 310pre8 GammaJets_Pt_80_120
        '/store/relval/CMSSW_3_1_0_pre8/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0006/4ED22FCA-DA4D-DE11-99AD-001D09F25109.root',
        '/store/relval/CMSSW_3_1_0_pre8/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0005/FE8075A6-874D-DE11-B9CA-001D09F231B0.root',
        '/store/relval/CMSSW_3_1_0_pre8/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0005/E29B17F1-874D-DE11-ABA5-001617E30E2C.root',
        '/store/relval/CMSSW_3_1_0_pre8/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0005/E0252B93-864D-DE11-A364-001617C3B66C.root',
        '/store/relval/CMSSW_3_1_0_pre8/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0005/DE967DCB-874D-DE11-AA6B-0016177CA7A0.root',
        '/store/relval/CMSSW_3_1_0_pre8/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0005/C06928FE-864D-DE11-99D4-001D09F34488.root',
        '/store/relval/CMSSW_3_1_0_pre8/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0005/B0ABB4B7-874D-DE11-89CC-001D09F2AF96.root',
        '/store/relval/CMSSW_3_1_0_pre8/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0005/9E7BCB29-874D-DE11-AD2F-0019B9F72F97.root',
        '/store/relval/CMSSW_3_1_0_pre8/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0005/767866FF-874D-DE11-9A41-001617DBD556.root',
        '/store/relval/CMSSW_3_1_0_pre8/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0005/5483E574-874D-DE11-84A7-001D09F28C1E.root',
        '/store/relval/CMSSW_3_1_0_pre8/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0005/4256437E-874D-DE11-A640-001617C3B76E.root',
        '/store/relval/CMSSW_3_1_0_pre8/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0005/2A11B86B-874D-DE11-ADFA-001D09F24259.root',
        '/store/relval/CMSSW_3_1_0_pre8/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0005/145084EF-864D-DE11-9E5C-001D09F253FC.root',
        '/store/relval/CMSSW_3_1_0_pre8/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0005/068CABEE-874D-DE11-AB70-000423D6B358.root'
    
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
#photonValidation.minPhoEtCut = 10
#photonValidation.eMax  = 100
#photonValidation.etMax = 50
#photonValidation.etScale = 0.20
#photonValidation.signal = True

## For single gamma pt = 35
#photonValidation.minPhoEtCut = 35
#photonValidation.eMax  = 300
#photonValidation.etMax = 50
#photonValidation.etScale = 0.20
#photonValidation.signal = True

## For gam Jet and higgs
photonValidation.minPhoEtCut = 20
photonValidation.eMax  = 500
photonValidation.etMax = 500
photonValidation.signal = True
## same for all
photonValidation.convTrackMinPtCut = 1.
photonValidation.rBin = 48
photonValidation.eoverpMin = 0.
photonValidation.eoverpMax = 5.


process.FEVT = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring("keep *_MEtoEDMConverter_*_*"),
    fileName = cms.untracked.string('pippo.root')
)

process.p1 = cms.Path(process.tpSelection*process.photonValidation)
#process.p1 = cms.Path(process.mix*process.trackingParticles*process.tpSelection*process.photonValidation)
process.schedule = cms.Schedule(process.p1)



