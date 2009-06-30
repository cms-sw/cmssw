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



#  include "DQMServices/Components/data/MessageLogger.cfi"
#  service = LoadAllDictionaries {}



process.maxEvents = cms.untracked.PSet(
# input = cms.untracked.int32(3000)
)



from Validation.RecoEgamma.photonValidator_cfi import *
#photonValidation.OutputFileName = 'PhotonValidationRelVal310pre11_SingleGammaPt10.root'
#photonValidation.OutputFileName = 'PhotonValidationRelVal310pre11_SingleGammaPt35.root'
photonValidation.OutputFileName = 'PhotonValidationRelVal310pre11_SingleGammaFlatPt10_100.root'
#photonValidation.OutputFileName = 'PhotonValidationRelVal310pre11_H130GGgluonfusion.root'
#photonValidation.OutputFileName = 'PhotonValidationRelVal310pre11_GammaJets_Pt_80_120.root'
#photonValidation.OutputFileName = 'PhotonValidationRelVal310pre11_QCD_Pt_50_80.root'



process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(

    
        
# official RelVal 310pre11 single Photons pt=10GeV

#'/store/relval/CMSSW_3_1_0_pre11/RelValSingleGammaPt10/GEN-SIM-RECO/MC_31X_V1-v1/0000/F844D40A-A964-DE11-9580-001D09F23A3E.root'
    
# official RelVal 310pre11 single Photons pt=35GeV            

#'/store/relval/CMSSW_3_1_0_pre11/RelValSingleGammaPt35/GEN-SIM-RECO/MC_31X_V1-v1/0000/D0C7F2F0-A864-DE11-A44F-001D09F34488.root',
#'/store/relval/CMSSW_3_1_0_pre11/RelValSingleGammaPt35/GEN-SIM-RECO/MC_31X_V1-v1/0000/023306E8-EC64-DE11-974D-001D09F28755.root'


# official RelVal 310pre11 single Photons Flat pt 10-100GeV


'/store/relval/CMSSW_3_1_0_pre11/RelValSingleGammaFlatPt10To100/GEN-SIM-RECO/MC_31X_V1-v1/0001/7866FD2A-F064-DE11-AE01-00304876A15B.root',
'/store/relval/CMSSW_3_1_0_pre11/RelValSingleGammaFlatPt10To100/GEN-SIM-RECO/MC_31X_V1-v1/0000/44097A69-B264-DE11-BBA5-0018F3D095EE.root'

        
# official RelVal 310pre11 RelValH130GGgluonfusion
#'/store/relval/CMSSW_3_1_0_pre11/RelValH130GGgluonfusion/GEN-SIM-RECO/STARTUP31X_V1-v1/0000/C2A35ED3-7364-DE11-83BE-001D09F2AF1E.root',
#'/store/relval/CMSSW_3_1_0_pre11/RelValH130GGgluonfusion/GEN-SIM-RECO/STARTUP31X_V1-v1/0000/A6D6068F-AD64-DE11-91B7-001D09F232B9.root',
#'/store/relval/CMSSW_3_1_0_pre11/RelValH130GGgluonfusion/GEN-SIM-RECO/STARTUP31X_V1-v1/0000/9E11DFFC-B164-DE11-812C-001D09F252E9.root',
#'/store/relval/CMSSW_3_1_0_pre11/RelValH130GGgluonfusion/GEN-SIM-RECO/STARTUP31X_V1-v1/0000/8E7BF1F1-9164-DE11-90D0-001D09F29524.root',
#'/store/relval/CMSSW_3_1_0_pre11/RelValH130GGgluonfusion/GEN-SIM-RECO/STARTUP31X_V1-v1/0000/0EEA012E-ED64-DE11-8D2A-001617C3B70E.root'

        
# official RelVal 310pre11 GammaJets_Pt_80_120

# '/store/relval/CMSSW_3_1_0_pre11/RelValGammaJets_Pt_80_120/GEN-SIM-RECO/STARTUP31X_V1-v1/0000/FC7DC211-8164-DE11-B74A-001D09F28EC1.root',
# '/store/relval/CMSSW_3_1_0_pre11/RelValGammaJets_Pt_80_120/GEN-SIM-RECO/STARTUP31X_V1-v1/0000/AC0B9DBD-B264-DE11-B787-0019B9F72CC2.root',
# '/store/relval/CMSSW_3_1_0_pre11/RelValGammaJets_Pt_80_120/GEN-SIM-RECO/STARTUP31X_V1-v1/0000/AA53252D-ED64-DE11-9DED-001D09F28F11.root',
# '/store/relval/CMSSW_3_1_0_pre11/RelValGammaJets_Pt_80_120/GEN-SIM-RECO/STARTUP31X_V1-v1/0000/52B10569-AD64-DE11-8FD0-001D09F232B9.root',
# '/store/relval/CMSSW_3_1_0_pre11/RelValGammaJets_Pt_80_120/GEN-SIM-RECO/STARTUP31X_V1-v1/0000/249419A0-9964-DE11-B7EB-001D09F24FEC.root'
 
    ),
                            
                            
    secondaryFileNames = cms.untracked.vstring(

# official RelVal 310pre11 single Photons pt=10GeV    

# '/store/relval/CMSSW_3_1_0_pre11/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0000/CA5CA038-9D64-DE11-B83B-001617E30CC8.root',
# '/store/relval/CMSSW_3_1_0_pre11/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0000/6439D1FA-EC64-DE11-B807-000423D99264.root',
# '/store/relval/CMSSW_3_1_0_pre11/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0000/505E4A05-6364-DE11-A2A7-001D09F25456.root'
 
# official RelVal 310pre11 single Photons pt=35GeV

#'/store/relval/CMSSW_3_1_0_pre11/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0000/EA6FE75B-6464-DE11-B272-000423D98804.root',
#'/store/relval/CMSSW_3_1_0_pre11/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0000/AAB3E9A4-9C64-DE11-97F5-001D09F282F5.root',
#'/store/relval/CMSSW_3_1_0_pre11/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0000/62EAC420-ED64-DE11-8E71-001D09F231C9.root'


# official RelVal 310pre11 single Photons Flat pt 10-100GeV

  '/store/relval/CMSSW_3_1_0_pre11/RelValSingleGammaFlatPt10To100/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0001/D6E76E16-F064-DE11-A180-0030486792BA.root',
  '/store/relval/CMSSW_3_1_0_pre11/RelValSingleGammaFlatPt10To100/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0000/F89CC81D-B364-DE11-9CEB-003048D15D22.root',
  '/store/relval/CMSSW_3_1_0_pre11/RelValSingleGammaFlatPt10To100/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0000/90DDC066-B264-DE11-88EA-001BFCDBD184.root',
  '/store/relval/CMSSW_3_1_0_pre11/RelValSingleGammaFlatPt10To100/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0000/566BFDA4-B164-DE11-B025-001A92811702.root',
  '/store/relval/CMSSW_3_1_0_pre11/RelValSingleGammaFlatPt10To100/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0000/282B0EA3-B164-DE11-B6F8-001A92971B64.root'

# official RelVal 310pre11 RelValH130GGgluonfusion

#'/store/relval/CMSSW_3_1_0_pre11/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0000/DA994D02-ED64-DE11-8007-0019DB29C614.root',
#'/store/relval/CMSSW_3_1_0_pre11/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0000/D68387BC-7264-DE11-A64F-001D09F25438.root',
#'/store/relval/CMSSW_3_1_0_pre11/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0000/CE3B25FB-7C64-DE11-8AD6-001D09F253D4.root',
#'/store/relval/CMSSW_3_1_0_pre11/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0000/848938FD-A964-DE11-A814-001D09F2512C.root',
#'/store/relval/CMSSW_3_1_0_pre11/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0000/78BD393E-AE64-DE11-868F-001D09F29321.root',
#'/store/relval/CMSSW_3_1_0_pre11/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0000/760F2D60-5D64-DE11-AEBA-001D09F2525D.root',
#'/store/relval/CMSSW_3_1_0_pre11/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0000/66EEC670-4D64-DE11-B0C2-000423D99996.root',
#'/store/relval/CMSSW_3_1_0_pre11/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0000/4E842403-8F64-DE11-863D-000423D99B3E.root',
#'/store/relval/CMSSW_3_1_0_pre11/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0000/46305028-AE64-DE11-9BE8-001D09F241B9.root',
#'/store/relval/CMSSW_3_1_0_pre11/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0000/0EE465D4-9864-DE11-88F5-000423D9870C.root',
#'/store/relval/CMSSW_3_1_0_pre11/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0000/08552BA0-B064-DE11-8184-001D09F28D4A.root',
#'/store/relval/CMSSW_3_1_0_pre11/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0000/06BE6194-9C64-DE11-A795-001D09F24DDF.root',
#'/store/relval/CMSSW_3_1_0_pre11/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0000/0254AEFA-8464-DE11-811B-001D09F2AD7F.root'


# official RelVal 310pre11 GammaJets_Pt_80_120

#'/store/relval/CMSSW_3_1_0_pre11/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0000/DC292942-AE64-DE11-B9CE-001D09F24EE3.root',
#'/store/relval/CMSSW_3_1_0_pre11/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0000/CC2F46A0-9C64-DE11-B14B-001D09F23F2A.root',
#'/store/relval/CMSSW_3_1_0_pre11/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0000/B08EBBCF-5064-DE11-9775-001D09F25325.root',
#'/store/relval/CMSSW_3_1_0_pre11/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0000/92D30771-AA64-DE11-948A-001D09F231C9.root',
#'/store/relval/CMSSW_3_1_0_pre11/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0000/8EEAC691-9964-DE11-96F6-001617DBD224.root',
#'/store/relval/CMSSW_3_1_0_pre11/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0000/84EF5240-8764-DE11-8FEA-001D09F295A1.root',
#'/store/relval/CMSSW_3_1_0_pre11/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0000/7EA36B14-B064-DE11-9D25-001D09F2AF1E.root',
#'/store/relval/CMSSW_3_1_0_pre11/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0000/3ECC49C8-A864-DE11-9890-001D09F28D4A.root',
#'/store/relval/CMSSW_3_1_0_pre11/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0000/3C9EF5BF-B264-DE11-9956-0030487D1BCC.root',
#'/store/relval/CMSSW_3_1_0_pre11/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0000/3214DF37-8164-DE11-A708-001D09F251D1.root',
#'/store/relval/CMSSW_3_1_0_pre11/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0000/2651E624-7064-DE11-BA62-001D09F24303.root',
#'/store/relval/CMSSW_3_1_0_pre11/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0000/16A79491-B464-DE11-A95E-001D09F24498.root',
#'/store/relval/CMSSW_3_1_0_pre11/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0000/165BD7F2-9364-DE11-A47C-001D09F2AF96.root',
#'/store/relval/CMSSW_3_1_0_pre11/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0000/02D3A025-ED64-DE11-A78C-001D09F2841C.root'

    
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
photonValidation.minPhoEtCut = 10
photonValidation.eMax  = 300
photonValidation.etMax = 300
photonValidation.etScale = 0.10
photonValidation.signal = True

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



