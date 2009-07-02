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
process.GlobalTag.globaltag = 'MC_31X_V1::All'


process.DQMStore = cms.Service("DQMStore");
process.load("DQMServices.Components.DQMStoreStats_cfi")
from DQMServices.Components.DQMStoreStats_cfi import *
dqmStoreStats.runOnEndJob = cms.untracked.bool(True)


#  include "DQMServices/Components/data/MessageLogger.cfi"
#  service = LoadAllDictionaries {}



process.maxEvents = cms.untracked.PSet(
# input = cms.untracked.int32(3000)
)



from Validation.RecoEgamma.photonValidator_cfi import *
#photonValidation.OutputFileName = 'PhotonValidationRelVal310_SingleGammaPt10.root'
#photonValidation.OutputFileName = 'PhotonValidationRelVal310_SingleGammaPt35.root'
#photonValidation.OutputFileName = 'PhotonValidationRelVal310_SingleGammaFlatPt10_100.root'
photonValidation.OutputFileName = 'PhotonValidationRelVal310_H130GGgluonfusion.root'
#photonValidation.OutputFileName = 'PhotonValidationRelVal310_GammaJets_Pt_80_120.root'
#photonValidation.OutputFileName = 'PhotonValidationRelVal310_QCD_Pt_50_80.root'



process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(

    
        
# official RelVal 310 single Photons pt=10GeV

#        '/store/relval/CMSSW_3_1_0/RelValSingleGammaPt10/GEN-SIM-RECO/MC_31X_V1-v1/0001/960522C2-DE66-DE11-8C70-001617E30D40.root'
# official RelVal 310 single Photons pt=35GeV            
#        '/store/relval/CMSSW_3_1_0/RelValSingleGammaPt35/GEN-SIM-RECO/MC_31X_V1-v1/0001/7A56FB59-DE66-DE11-955B-001D09F24EC0.root',
#        '/store/relval/CMSSW_3_1_0/RelValSingleGammaPt35/GEN-SIM-RECO/MC_31X_V1-v1/0001/1CF7616A-C066-DE11-930E-001D09F2543D.root'
  
# official RelVal 310 single Photons Flat pt 10-100GeV


        
# official RelVal 310 RelValH130GGgluonfusion
        '/store/relval/CMSSW_3_1_0/RelValH130GGgluonfusion/GEN-SIM-RECO/STARTUP31X_V1-v1/0002/206777EA-DE66-DE11-BDCB-000423D98B08.root',
        '/store/relval/CMSSW_3_1_0/RelValH130GGgluonfusion/GEN-SIM-RECO/STARTUP31X_V1-v1/0001/6CC60C13-C166-DE11-BE59-001D09F29321.root',
        '/store/relval/CMSSW_3_1_0/RelValH130GGgluonfusion/GEN-SIM-RECO/STARTUP31X_V1-v1/0001/3CB10493-C066-DE11-8C5D-001D09F2B30B.root',
        '/store/relval/CMSSW_3_1_0/RelValH130GGgluonfusion/GEN-SIM-RECO/STARTUP31X_V1-v1/0001/289E8DFE-C066-DE11-8317-001D09F2523A.root',
        '/store/relval/CMSSW_3_1_0/RelValH130GGgluonfusion/GEN-SIM-RECO/STARTUP31X_V1-v1/0001/06D0B3EC-BF66-DE11-A798-001D09F2441B.root'

        
# official RelVal 310 GammaJets_Pt_80_120
#        '/store/relval/CMSSW_3_1_0/RelValGammaJets_Pt_80_120/GEN-SIM-RECO/STARTUP31X_V1-v1/0002/2AAA6789-DF66-DE11-AE21-001D09F24024.root',
#        '/store/relval/CMSSW_3_1_0/RelValGammaJets_Pt_80_120/GEN-SIM-RECO/STARTUP31X_V1-v1/0001/E62A1021-4F66-DE11-9AC2-001D09F29533.root',
#        '/store/relval/CMSSW_3_1_0/RelValGammaJets_Pt_80_120/GEN-SIM-RECO/STARTUP31X_V1-v1/0001/522DD728-4E66-DE11-8738-001D09F2924F.root',
#        '/store/relval/CMSSW_3_1_0/RelValGammaJets_Pt_80_120/GEN-SIM-RECO/STARTUP31X_V1-v1/0001/3CF955C7-4D66-DE11-962A-001D09F29146.root',
#        '/store/relval/CMSSW_3_1_0/RelValGammaJets_Pt_80_120/GEN-SIM-RECO/STARTUP31X_V1-v1/0001/1883AC54-4F66-DE11-B02A-001D09F24D8A.root'
 
    ),
                            
                            
    secondaryFileNames = cms.untracked.vstring(

# official RelVal 310 single Photons pt=10GeV    
#        '/store/relval/CMSSW_3_1_0/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0001/9CE8F0CF-4666-DE11-B1C5-001D09F23F2A.root',
#        '/store/relval/CMSSW_3_1_0/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0001/8ECD1E9D-3D66-DE11-831E-001D09F29114.root',
#        '/store/relval/CMSSW_3_1_0/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0001/24F0057A-DE66-DE11-86F7-000423D60FF6.root'
        
 
# official RelVal 310 single Photons pt=35GeV

#        '/store/relval/CMSSW_3_1_0/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0002/CE1C3FA9-E066-DE11-B117-001617DC1F70.root',
#        '/store/relval/CMSSW_3_1_0/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0001/74778F0E-B766-DE11-881B-001D09F23D1D.root',
#        '/store/relval/CMSSW_3_1_0/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0001/5C964CCF-BE66-DE11-B265-001D09F23F2A.root'
        
        
# official RelVal 310 single Photons Flat pt 10-100GeV


# official RelVal 310 RelValH130GGgluonfusion
        '/store/relval/CMSSW_3_1_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0001/EA01A495-C066-DE11-A31E-001D09F25217.root',
        '/store/relval/CMSSW_3_1_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0001/DEE7492C-BE66-DE11-9668-001D09F250AF.root',
        '/store/relval/CMSSW_3_1_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0001/DE569060-C066-DE11-9436-000423D98800.root',
        '/store/relval/CMSSW_3_1_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0001/D097BC0F-C166-DE11-BF0B-001D09F232B9.root',
        '/store/relval/CMSSW_3_1_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0001/CCE6B364-C066-DE11-B8F7-001D09F24FEC.root',
        '/store/relval/CMSSW_3_1_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0001/BA660EEC-C066-DE11-9D06-001D09F292D1.root',
        '/store/relval/CMSSW_3_1_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0001/A2EF98E7-BF66-DE11-A50C-001D09F2525D.root',
        '/store/relval/CMSSW_3_1_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0001/86FC01FB-C066-DE11-B5C0-001D09F29533.root',
        '/store/relval/CMSSW_3_1_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0001/7011BA66-B766-DE11-A5BC-000423D98930.root',
        '/store/relval/CMSSW_3_1_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0001/64FA5FC1-C066-DE11-A857-0019B9F72F97.root',
        '/store/relval/CMSSW_3_1_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0001/2A9281CE-C066-DE11-94FB-001D09F24399.root',
        '/store/relval/CMSSW_3_1_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0001/10498604-C166-DE11-A2C8-001D09F2924F.root',
        '/store/relval/CMSSW_3_1_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0001/0E3E3455-DE66-DE11-81EB-000423D9890C.root'


# official RelVal 310 GammaJets_Pt_80_120
#        '/store/relval/CMSSW_3_1_0/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0002/7CA4D29B-DE66-DE11-91D3-000423D94C68.root',
#        '/store/relval/CMSSW_3_1_0/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0001/F0BF6757-4F66-DE11-B2EC-001D09F25393.root',
#        '/store/relval/CMSSW_3_1_0/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0001/EE18B620-4F66-DE11-86C7-0019B9F72BFF.root',
#        '/store/relval/CMSSW_3_1_0/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0001/D2273EB5-4E66-DE11-8CA4-001D09F23D1D.root',
#        '/store/relval/CMSSW_3_1_0/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0001/CAC4BAD1-4D66-DE11-8B2D-000423D6006E.root',
#        '/store/relval/CMSSW_3_1_0/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0001/C8319120-4F66-DE11-806A-001D09F24934.root',
#        '/store/relval/CMSSW_3_1_0/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0001/BA45AC28-4E66-DE11-86E6-001D09F25401.root',
#        '/store/relval/CMSSW_3_1_0/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0001/AE77FC29-4F66-DE11-8774-001D09F282F5.root',
#        '/store/relval/CMSSW_3_1_0/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0001/ACF5A083-4D66-DE11-94CD-001D09F29524.root',
#        '/store/relval/CMSSW_3_1_0/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0001/94AF12C5-4D66-DE11-9169-001D09F25393.root',
#        '/store/relval/CMSSW_3_1_0/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0001/86AE89E0-4C66-DE11-912E-001D09F28F11.root',
#        '/store/relval/CMSSW_3_1_0/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0001/284F7CB8-4E66-DE11-A58E-001D09F282F5.root',
#        '/store/relval/CMSSW_3_1_0/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0001/1ADFB858-4F66-DE11-8C30-001D09F29146.root',
#        '/store/relval/CMSSW_3_1_0/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0001/0CA62601-4E66-DE11-ADB0-001D09F23D1D.root'


    
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
#photonValidation.etMax = 300
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



