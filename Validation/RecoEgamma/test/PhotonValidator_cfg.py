import FWCore.ParameterSet.Config as cms

process = cms.Process("TestPhotonValidator")
process.load('Configuration/StandardSequences/GeometryPilot2_cff')
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")
process.load("RecoTracker.GeometryESProducer.TrackerRecoGeometryESProducer_cfi")
process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
process.load("RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi")
process.load("SimGeneral.MixingModule.mixNoPU_cfi")
process.load("TrackingTools.TransientTrack.TransientTrackBuilder_cfi")
process.load("DQMServices.Components.MEtoEDMConverter_cfi")
process.load("Validation.RecoEgamma.photonValidationSequence_cff")
process.load("Validation.RecoEgamma.photonPostprocessing_cfi")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'MC_31X_V9::All'


process.DQMStore = cms.Service("DQMStore");
process.load("DQMServices.Components.DQMStoreStats_cfi")
from DQMServices.Components.DQMStoreStats_cfi import *
dqmStoreStats.runOnEndJob = cms.untracked.bool(True)



process.maxEvents = cms.untracked.PSet(
#input = cms.untracked.int32(10)
)



from Validation.RecoEgamma.photonValidationSequence_cff import *
from Validation.RecoEgamma.photonPostprocessing_cfi import *

photonValidation.OutputMEsInRootFile = True
#photonValidation.OutputFileName = 'PhotonValidationRelVal330_SingleGammaPt10.root'
photonValidation.OutputFileName = 'PhotonValidationRelVal330_SingleGammaPt35.root'
#photonValidation.OutputFileName = 'PhotonValidationRelVal330_SingleGammaFlatPt10_100.root'
#photonValidation.OutputFileName = 'PhotonValidationRelVal330_H130GGgluonfusion.root'
#photonValidation.OutputFileName = 'PhotonValidationRelVal330_GammaJets_Pt_80_120.root'
#photonValidation.OutputFileName = 'PhotonValidationRelVal330_QCD_Pt_80_120.root'

photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(

    
        
# official RelVal 330 single Photons pt=10GeV
# '/store/relval/CMSSW_3_3_0/RelValSingleGammaPt10/GEN-SIM-RECO/MC_31X_V9-v1/0009/F6476C34-75B7-DE11-A407-001D09F28755.root',
# '/store/relval/CMSSW_3_3_0/RelValSingleGammaPt10/GEN-SIM-RECO/MC_31X_V9-v1/0009/6E994B34-22B7-DE11-8A71-001D09F28F11.root'


# official RelVal 330 single Photons pt=35GeV            


 '/store/relval/CMSSW_3_3_0/RelValSingleGammaPt35/GEN-SIM-RECO/MC_31X_V9-v1/0009/38B8D900-75B7-DE11-8893-001D09F2910A.root',
 '/store/relval/CMSSW_3_3_0/RelValSingleGammaPt35/GEN-SIM-RECO/MC_31X_V9-v1/0008/E2A42FA3-8CB6-DE11-9FB6-001D09F24682.root'


# official RelVal 330 single Photons Flat pt 10-100GeV


        
# official RelVal 330 RelValH130GGgluonfusion
# '/store/relval/CMSSW_3_3_0/RelValH130GGgluonfusion/GEN-SIM-RECO/STARTUP31X_V8-v1/0009/E60407BB-24B7-DE11-AC69-001D09F29169.root',
# '/store/relval/CMSSW_3_3_0/RelValH130GGgluonfusion/GEN-SIM-RECO/STARTUP31X_V8-v1/0009/D630EDA3-25B7-DE11-84F4-001D09F24934.root',
# '/store/relval/CMSSW_3_3_0/RelValH130GGgluonfusion/GEN-SIM-RECO/STARTUP31X_V8-v1/0009/C231EEC1-23B7-DE11-92E4-0019B9F70468.root',
# '/store/relval/CMSSW_3_3_0/RelValH130GGgluonfusion/GEN-SIM-RECO/STARTUP31X_V8-v1/0009/90A8F208-75B7-DE11-98E0-001D09F295A1.root',
# '/store/relval/CMSSW_3_3_0/RelValH130GGgluonfusion/GEN-SIM-RECO/STARTUP31X_V8-v1/0008/9232C3EA-1DB7-DE11-A0E4-001D09F2AF96.root'
 
        
# official RelVal 330 GammaJets_Pt_80_120

# official RelVal 330 QCD_Pt_80_120


# '/store/relval/CMSSW_3_3_0/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V9-v1/0009/880AA097-75B7-DE11-B848-001D09F23C73.root',
# '/store/relval/CMSSW_3_3_0/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V9-v1/0008/CC9C9FAC-86B6-DE11-8A1B-001D09F24FEC.root',
# '/store/relval/CMSSW_3_3_0/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V9-v1/0008/A44075CB-7FB6-DE11-8010-000423D98868.root',
# '/store/relval/CMSSW_3_3_0/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V9-v1/0008/6E433585-84B6-DE11-A3E1-001D09F2437B.root',
# '/store/relval/CMSSW_3_3_0/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V9-v1/0008/285A7AC1-82B6-DE11-BA29-001D09F2525D.root',
# '/store/relval/CMSSW_3_3_0/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V9-v1/0008/12755919-88B6-DE11-8FF2-000423D996C8.root',
# '/store/relval/CMSSW_3_3_0/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V9-v1/0008/04B93222-89B6-DE11-93E9-001D09F29524.root'
 

 
    ),
                            
                            
    secondaryFileNames = cms.untracked.vstring(

# official RelVal 330 single Photons pt=10GeV    

#        '/store/relval/CMSSW_3_3_0/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0009/D8504B33-75B7-DE11-BA0A-001D09F24FEC.root',
#        '/store/relval/CMSSW_3_3_0/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0009/5EE6322A-21B7-DE11-AACA-001D09F2960F.root',
#        '/store/relval/CMSSW_3_3_0/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0009/26A09364-22B7-DE11-95E3-0019B9F70468.root'
        
    
# official RelVal 330 single Photons pt=35GeV

        '/store/relval/CMSSW_3_3_0/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0008/D4425DF8-8DB6-DE11-B613-001D09F282F5.root',
        '/store/relval/CMSSW_3_3_0/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0008/C81D2B6E-8AB6-DE11-906D-001D09F25109.root',
        '/store/relval/CMSSW_3_3_0/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0008/1CB473F3-8BB6-DE11-902B-000423D94990.root'

# official RelVal 330 single Photons Flat pt 10-100GeV


# official RelVal 330 RelValH130GGgluonfusion
#        '/store/relval/CMSSW_3_3_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V8-v1/0009/FCF13148-23B7-DE11-80E8-001D09F2A690.root',
#        '/store/relval/CMSSW_3_3_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V8-v1/0009/F8B012C3-25B7-DE11-9F69-001D09F28F1B.root',
#        '/store/relval/CMSSW_3_3_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V8-v1/0009/E8F7F230-23B7-DE11-9539-0019B9F70468.root',
#        '/store/relval/CMSSW_3_3_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V8-v1/0009/C64BFC48-25B7-DE11-B800-0019B9F72CE5.root',
#        '/store/relval/CMSSW_3_3_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V8-v1/0009/740F185B-26B7-DE11-B73C-0019B9F72F97.root',
#        '/store/relval/CMSSW_3_3_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V8-v1/0009/6EF9F602-75B7-DE11-B653-001D09F28D54.root',
#        '/store/relval/CMSSW_3_3_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V8-v1/0009/6E4CBC3B-24B7-DE11-9DA2-0019B9F72CE5.root',
#        '/store/relval/CMSSW_3_3_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V8-v1/0009/58794B0D-25B7-DE11-9450-001D09F2841C.root',
#        '/store/relval/CMSSW_3_3_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V8-v1/0009/54FF7E21-24B7-DE11-84E3-001D09F29619.root',
#        '/store/relval/CMSSW_3_3_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V8-v1/0009/38161AD6-24B7-DE11-8AC9-001D09F253D4.root',
#        '/store/relval/CMSSW_3_3_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V8-v1/0008/8C42FD91-1EB7-DE11-85B6-001D09F248F8.root',
#        '/store/relval/CMSSW_3_3_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V8-v1/0008/7ACC86AC-1BB7-DE11-938D-001D09F290CE.root',
#        '/store/relval/CMSSW_3_3_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V8-v1/0008/30810D88-1CB7-DE11-8FB5-001D09F2437B.root'


# official RelVal 330 GammaJets_Pt_80_120

# official RelVal 330 QCD_Pt_80_120

#        '/store/relval/CMSSW_3_3_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0009/D6C17CFC-74B7-DE11-AB70-001D09F292D1.root',
#        '/store/relval/CMSSW_3_3_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0008/E8967CA5-86B6-DE11-A98B-000423D990CC.root',
#        '/store/relval/CMSSW_3_3_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0008/A6D46FD2-88B6-DE11-9A62-001D09F295FB.root',
#        '/store/relval/CMSSW_3_3_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0008/A6581A1E-87B6-DE11-ABC8-001D09F24D4E.root',
#        '/store/relval/CMSSW_3_3_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0008/986674A5-7DB6-DE11-B624-001D09F2AD84.root',
#        '/store/relval/CMSSW_3_3_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0008/8CD25F56-81B6-DE11-BFCA-001D09F297EF.root',
#        '/store/relval/CMSSW_3_3_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0008/8AF01E8C-89B6-DE11-AA50-000423D996C8.root',
#        '/store/relval/CMSSW_3_3_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0008/888A2A74-80B6-DE11-A90D-001D09F28F0C.root',
#        '/store/relval/CMSSW_3_3_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0008/82606045-86B6-DE11-BEBD-001D09F25109.root',
#        '/store/relval/CMSSW_3_3_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0008/7C232926-89B6-DE11-87E0-000423D996C8.root',
#        '/store/relval/CMSSW_3_3_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0008/6AFFC2DB-7EB6-DE11-BC1B-001D09F2B2CF.root',
#        '/store/relval/CMSSW_3_3_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0008/64E7D72F-84B6-DE11-B9C1-001D09F2AF96.root',
#        '/store/relval/CMSSW_3_3_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0008/462CE89F-82B6-DE11-ABCE-001D09F34488.root',
#        '/store/relval/CMSSW_3_3_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0008/26CC8AAE-83B6-DE11-8952-001D09F28F25.root',
#        '/store/relval/CMSSW_3_3_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0008/240A57B3-85B6-DE11-B626-0019B9F6C674.root',
#        '/store/relval/CMSSW_3_3_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0008/16421F6E-88B6-DE11-9EF2-000423D996C8.root',
#        '/store/relval/CMSSW_3_3_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0008/06BCF7D6-87B6-DE11-801E-001D09F27067.root'


    
    )
 )


photonPostprocessing.rBin = 48

## For single gamma fla pt =10-150
#photonValidation.eMax  = 300
#photonValidation.etMax = 300
#photonValidation.etScale = 0.10


## For single gamma pt =10
#photonValidation.eMax  = 100
#photonValidation.etMax = 50
#photonValidation.etScale = 0.20
#photonPostprocessing.eMax  = 100
#photonPostprocessing.etMax = 50


## For single gamma pt = 35
photonValidation.eMax  = 300
photonValidation.etMax = 50
photonValidation.etScale = 0.20
photonValidation.dCotCutOn = False
photonValidation.dCotCutValue = 0.15
#photonValidation.likelihoodCut = 0.90

## For gam Jet and higgs
#photonValidation.eMax  = 500
#photonValidation.etMax = 500
#photonPostprocessing.eMax  = 500
#photonPostprocessing.etMax = 500




process.FEVT = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring("keep *_MEtoEDMConverter_*_*"),
    fileName = cms.untracked.string('pippo.root')
)


#process.p1 = cms.Path(process.photonValidation)
process.p1 = cms.Path(process.tpSelection*process.photonValidationSequence*process.photonPostprocessing*process.dqmStoreStats)
#process.p1 = cms.Path(process.mix*process.trackingParticles*process.tpSelection*process.photonValidation)
process.schedule = cms.Schedule(process.p1)



