import FWCore.ParameterSet.Config as cms

process = cms.Process("TestPhotonValidator")

#process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")


process.load('Configuration/StandardSequences/GeometryPilot2_cff')
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
#process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
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


process.DQMStore = cms.Service("DQMStore");



#  include "DQMServices/Components/data/MessageLogger.cfi"
#  service = LoadAllDictionaries {}



process.maxEvents = cms.untracked.PSet(
# input = cms.untracked.int32(1000)
)


from Validation.RecoEgamma.photonValidator_cfi import *
#photonValidation.OutputFileName = 'PhotonValidationRelVal310pre5_SingleGammaPt35.root'
photonValidation.OutputFileName = 'PhotonValidationRelVal310pre5_SingleGammaPt10.root'
#photonValidation.OutputFileName = 'PhotonValidationRelVal310pre5_SingleGammaFlatPt10_100.root'
#photonValidation.OutputFileName = 'PhotonValidationRelVal310pre5_QCD_Pt_50_80.root'
#photonValidation.OutputFileName = 'PhotonValidationRelVal310pre5_GammaJets_Pt_80_120.root'
#photonValidation.OutputFileName = 'PhotonValidationRelVal310pre5_H130GGgluonfusion.root'

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(

#'file:/build/nancy/CMSSW_3_1_0_pre5/src/RecoEgamma/SingleGammaPt35_cfi_GEN_SIM_DIGI_L1_DIGI2RAW_RAW2DIGI_RECO.root'

  # official RelVal 310pre5 single Photons pt=35GeV

#'file:/data/test/CMSSW_3_1_0_pre5/src/RecoEgamma/SingleGammaPt35_ReReco.root'
#'file:/data/test/CMSSW_3_1_0_pre5/src/RecoEgamma/SingleGammaPt35_ReRecoOldEcalIsoCut.root'

  # official RelVal 310pre5 single Photons pt=10GeV    
    '/store/relval/CMSSW_3_1_0_pre5/RelValSingleGammaPt10/GEN-SIM-RECO/IDEAL_31X_v1/0000/6EFC1418-0C2C-DE11-94BB-000423D9890C.root'


  # official RelVal 310pre5 RelValH130GGgluonfusion


  # official RelVal 310pre5 GammaJets_Pt_80_120

    ),
                            
                            
    secondaryFileNames = cms.untracked.vstring(
  # official RelVal 310pre5 single Photons pt=35GeV


    # official RelVal 310pre5 single Photons pt=10GeV    

    '/store/relval/CMSSW_3_1_0_pre5/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0000/14CFB193-9E2B-DE11-9155-000423D98800.root',
    '/store/relval/CMSSW_3_1_0_pre5/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0000/6011C343-9F2B-DE11-91FC-000423D944F8.root',
    '/store/relval/CMSSW_3_1_0_pre5/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0000/80CE3900-0C2C-DE11-851B-000423D94908.root'



    # official RelVal 310pre5 RelValH130GGgluonfusion

# official RelVal 310pre5 GammaJets_Pt_80_120

    
    )
 )



from SimTracker.TrackAssociation.TrackAssociatorByHits_cfi import *
import SimTracker.TrackAssociation.TrackAssociatorByHits_cfi
#TrackAssociatorByHits.AbsoluteNumberOfHits = True
#TrackAssociatorByHits.Cut_RecoToSim = 3
#TrackAssociatorByHits.Quality_SimToReco = 3
TrackAssociatorByHits.Cut_RecoToSim = 0.5
TrackAssociatorByHits.Quality_SimToReco = 0.5



## For single gamma pt =10
photonValidation.minPhoEtCut = 10
photonValidation.eMax  = 100
photonValidation.etMax = 50
photonValidation.etScale = 0.20

## For single gamma pt = 35
#photonValidation.minPhoEtCut = 35
#photonValidation.eMax  = 300
#photonValidation.etMax = 50
#photonValidation.etScale = 0.20

## For gam Jet and higgs
#photonValidation.minPhoEtCut = 10
#photonValidation.eMax  = 500
#photonValidation.etMax = 500
## same for all
photonValidation.convTrackMinPtCut = 1.
photonValidation.rBin = 48
photonValidation.eoverpMin = 0.
photonValidation.eoverpMax = 5.
photonValidation.signal = True

process.FEVT = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring("keep *_MEtoEDMConverter_*_*"),
    fileName = cms.untracked.string('pippo.root')
)

process.p1 = cms.Path(process.tpSelection*process.photonValidation)
#process.p1 = cms.Path(process.mix*process.trackingParticles*process.tpSelection*process.photonValidation)
process.schedule = cms.Schedule(process.p1)



