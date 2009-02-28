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
photonValidation.OutputFileName = 'PhotonValidationRelVal300pre7_SingleGammaPt35TrackPtCut.root'
#photonValidation.OutputFileName = 'PhotonValidationRelVal300pre7_SingleGammaFlatPt10_100.root'
#photonValidation.OutputFileName = 'PhotonValidationRelVal300pre7_QCD_Pt_50_80.root'
#photonValidation.OutputFileName = 'PhotonValidationRelVal300pre7_GammaJets_Pt_80_120.root'
#photonValidation.OutputFileName = 'test.root'



process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(

 # official RelVal 300pre7 single Photons pt=35GeV
    '/store/relval/CMSSW_3_0_0_pre7/RelValSingleGammaPt35/GEN-SIM-RECO/IDEAL_30X_v1/0006/B662239B-6AE8-DD11-9A5E-00304879FA4A.root'

    # official RelVal 300pre7 single Photons flat pt =10-100
#    '/store/relval/CMSSW_3_0_0_pre7/RelValSingleGammaFlatPt10To100/GEN-SIM-RECO/IDEAL_30X_v1/0001/D48A44ED-73E9-DD11-BDD9-0018F3D0968E.root',
#    '/store/relval/CMSSW_3_0_0_pre7/RelValSingleGammaFlatPt10To100/GEN-SIM-RECO/IDEAL_30X_v1/0002/9437241C-1BEA-DD11-A5B5-0018F3D09684.root'

    # official RelVal 300pre7 QCD_Pt_50_80
#    '/store/relval/CMSSW_3_0_0_pre7/RelValQCD_Pt_50_80/GEN-SIM-RECO/STARTUP_30X_v1/0001/282A48F1-73E9-DD11-BD25-001BFCDBD19E.root',
#    '/store/relval/CMSSW_3_0_0_pre7/RelValQCD_Pt_50_80/GEN-SIM-RECO/STARTUP_30X_v1/0001/B89FEC1F-73E9-DD11-95D0-003048678B14.root',
#    '/store/relval/CMSSW_3_0_0_pre7/RelValQCD_Pt_50_80/GEN-SIM-RECO/STARTUP_30X_v1/0001/DC8FAF67-72E9-DD11-9F9C-001731AF6B85.root',
#    '/store/relval/CMSSW_3_0_0_pre7/RelValQCD_Pt_50_80/GEN-SIM-RECO/STARTUP_30X_v1/0002/56016A1B-7DE9-DD11-BD38-003048767DE5.root',
#    '/store/relval/CMSSW_3_0_0_pre7/RelValQCD_Pt_50_80/GEN-SIM-RECO/STARTUP_30X_v1/0002/663CDA67-7FE9-DD11-973B-001731230FC9.root',
#    '/store/relval/CMSSW_3_0_0_pre7/RelValQCD_Pt_50_80/GEN-SIM-RECO/STARTUP_30X_v1/0002/6C9A58C1-7AE9-DD11-8473-00304867BED8.root',
#    '/store/relval/CMSSW_3_0_0_pre7/RelValQCD_Pt_50_80/GEN-SIM-RECO/STARTUP_30X_v1/0002/7E9DAE45-7CE9-DD11-A82A-003048678AF4.root',
#    '/store/relval/CMSSW_3_0_0_pre7/RelValQCD_Pt_50_80/GEN-SIM-RECO/STARTUP_30X_v1/0002/C23EDAC2-1AEA-DD11-B093-00304875ABE9.root'

    # official RelVal 300pre7 GammaJets_80_120
#    '/store/relval/CMSSW_3_0_0_pre7/RelValGammaJets_Pt_80_120/GEN-SIM-RECO/STARTUP_30X_v1/0006/0675C039-5EE8-DD11-A127-000423D94E70.root',
#    '/store/relval/CMSSW_3_0_0_pre7/RelValGammaJets_Pt_80_120/GEN-SIM-RECO/STARTUP_30X_v1/0006/4847CC41-5EE8-DD11-BC5B-001D09F248FD.root',
#    '/store/relval/CMSSW_3_0_0_pre7/RelValGammaJets_Pt_80_120/GEN-SIM-RECO/STARTUP_30X_v1/0006/985B66CD-6AE8-DD11-9877-000423D99394.root',
#    '/store/relval/CMSSW_3_0_0_pre7/RelValGammaJets_Pt_80_120/GEN-SIM-RECO/STARTUP_30X_v1/0006/BC73B5DE-5FE8-DD11-B6B3-001D09F2426D.root'
    

    ),
                            
                            
    secondaryFileNames = cms.untracked.vstring(
   # official RelVal 300pre7 single Photons pt=35GeV 

    '/store/relval/CMSSW_3_0_0_pre7/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0006/1EF2AD87-6AE8-DD11-8EA2-001D09F290BF.root',
    '/store/relval/CMSSW_3_0_0_pre7/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0006/B2726469-31E8-DD11-9A79-001617C3B76E.root',
    '/store/relval/CMSSW_3_0_0_pre7/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0006/B2B8CF43-32E8-DD11-9E9B-000423D99F3E.root'
  
   # official RelVal 300pre7 single Photons pt=10-100 
#    '/store/relval/CMSSW_3_0_0_pre7/RelValSingleGammaFlatPt10To100/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0001/28FAE958-72E9-DD11-BABE-003048678E94.root',
#    '/store/relval/CMSSW_3_0_0_pre7/RelValSingleGammaFlatPt10To100/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0001/5415B6EE-73E9-DD11-BE64-001A92971AA8.root',
#    '/store/relval/CMSSW_3_0_0_pre7/RelValSingleGammaFlatPt10To100/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0001/B8431867-75E9-DD11-872A-003048678B18.root',
#    '/store/relval/CMSSW_3_0_0_pre7/RelValSingleGammaFlatPt10To100/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0001/C28E0A22-73E9-DD11-9B44-001731AF67E7.root',
#    '/store/relval/CMSSW_3_0_0_pre7/RelValSingleGammaFlatPt10To100/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0002/BCA910FA-1AEA-DD11-9986-0018F3D096DE.root'

    # official RelVal 300pre7 QCD_Pt_50_80
#       '/store/relval/CMSSW_3_0_0_pre7/RelValQCD_Pt_50_80/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/3A7E81D7-70E9-DD11-B652-001731AF6A8D.root',
#        '/store/relval/CMSSW_3_0_0_pre7/RelValQCD_Pt_50_80/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/5A95F826-73E9-DD11-A88E-001A92810A98.root',
#        '/store/relval/CMSSW_3_0_0_pre7/RelValQCD_Pt_50_80/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/5EFA6F98-71E9-DD11-B94F-001731AF684B.root',
#        '/store/relval/CMSSW_3_0_0_pre7/RelValQCD_Pt_50_80/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/7437801F-73E9-DD11-81E1-003048678FFA.root',
#        '/store/relval/CMSSW_3_0_0_pre7/RelValQCD_Pt_50_80/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/74A6A3EB-73E9-DD11-9F46-001A92971BB4.root',
#        '/store/relval/CMSSW_3_0_0_pre7/RelValQCD_Pt_50_80/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/82A2D9AA-74E9-DD11-8900-001731AF66BF.root',
#        '/store/relval/CMSSW_3_0_0_pre7/RelValQCD_Pt_50_80/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/8E4F77E6-73E9-DD11-9927-00304867C0FC.root',
#        '/store/relval/CMSSW_3_0_0_pre7/RelValQCD_Pt_50_80/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/DAFE10E8-73E9-DD11-B45C-001A92971B36.root',
#        '/store/relval/CMSSW_3_0_0_pre7/RelValQCD_Pt_50_80/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/E0569B60-72E9-DD11-BB89-00304867D836.root',
#        '/store/relval/CMSSW_3_0_0_pre7/RelValQCD_Pt_50_80/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/ECA11D70-75E9-DD11-9A6E-001731AF68B5.root',
#        '/store/relval/CMSSW_3_0_0_pre7/RelValQCD_Pt_50_80/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/F02B6363-72E9-DD11-BB9E-0018F3D09688.root',
#        '/store/relval/CMSSW_3_0_0_pre7/RelValQCD_Pt_50_80/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/FA7C8C66-72E9-DD11-9E1C-00173199E924.root',
#        '/store/relval/CMSSW_3_0_0_pre7/RelValQCD_Pt_50_80/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0002/041218A7-1AEA-DD11-8AAB-0018F3D0968C.root',
#        '/store/relval/CMSSW_3_0_0_pre7/RelValQCD_Pt_50_80/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0002/0C26C2AF-7EE9-DD11-BD17-001731A28319.root',
#        '/store/relval/CMSSW_3_0_0_pre7/RelValQCD_Pt_50_80/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0002/32A53A5A-7FE9-DD11-BD67-003048678FF6.root',
#        '/store/relval/CMSSW_3_0_0_pre7/RelValQCD_Pt_50_80/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0002/3A6AB20D-7DE9-DD11-87FF-0018F3D0962E.root',
#        '/store/relval/CMSSW_3_0_0_pre7/RelValQCD_Pt_50_80/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0002/5ADFB7F8-79E9-DD11-9E64-0030486791F2.root',
#        '/store/relval/CMSSW_3_0_0_pre7/RelValQCD_Pt_50_80/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0002/72DB5586-7BE9-DD11-AECC-001731AF6B7D.root',
#        '/store/relval/CMSSW_3_0_0_pre7/RelValQCD_Pt_50_80/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0002/72DC33E3-80E9-DD11-92EA-003048D25B68.root',
#        '/store/relval/CMSSW_3_0_0_pre7/RelValQCD_Pt_50_80/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0002/929F2582-7BE9-DD11-BC58-001BFCDBD182.root',
#        '/store/relval/CMSSW_3_0_0_pre7/RelValQCD_Pt_50_80/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0002/B2CEEA44-7CE9-DD11-8C7A-00304867920A.root',
#        '/store/relval/CMSSW_3_0_0_pre7/RelValQCD_Pt_50_80/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0002/BE782C48-7CE9-DD11-ACDD-0018F3D09620.root',
#        '/store/relval/CMSSW_3_0_0_pre7/RelValQCD_Pt_50_80/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0002/C6D9BD10-7DE9-DD11-B7B2-00304867BEE4.root',
#        '/store/relval/CMSSW_3_0_0_pre7/RelValQCD_Pt_50_80/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0002/E2C5D1D0-7DE9-DD11-86E6-003048679076.root',
#        '/store/relval/CMSSW_3_0_0_pre7/RelValQCD_Pt_50_80/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0002/E612C8C3-7AE9-DD11-B7A2-001A92971B12.root',
#        '/store/relval/CMSSW_3_0_0_pre7/RelValQCD_Pt_50_80/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0002/E8024BDD-7DE9-DD11-8FA5-003048754C6B.root'

    # official RelVal 300pre7 GammaJets_80_120
#    '/store/relval/CMSSW_3_0_0_pre7/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0006/04C2AF36-5EE8-DD11-B248-000423D996B4.root',
#    '/store/relval/CMSSW_3_0_0_pre7/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0006/0ABA9EFD-5EE8-DD11-B457-001D09F23A84.root',
#    '/store/relval/CMSSW_3_0_0_pre7/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0006/180BC7D4-5FE8-DD11-8766-001D09F23A20.root',
#    '/store/relval/CMSSW_3_0_0_pre7/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0006/24720043-5EE8-DD11-872B-001D09F24682.root',
#    '/store/relval/CMSSW_3_0_0_pre7/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0006/30EFA037-5EE8-DD11-A200-001D09F24F65.root',
#    '/store/relval/CMSSW_3_0_0_pre7/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0006/620DD295-60E8-DD11-A1DD-000423D94494.root',
#    '/store/relval/CMSSW_3_0_0_pre7/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0006/6656A03B-5EE8-DD11-979F-0019B9F71A6B.root',
#    '/store/relval/CMSSW_3_0_0_pre7/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0006/78595C38-5EE8-DD11-A6E3-000423D94AA8.root',
#    '/store/relval/CMSSW_3_0_0_pre7/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0006/9A788057-6BE8-DD11-9DF3-001D09F2924F.root',
#    '/store/relval/CMSSW_3_0_0_pre7/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0006/A84F9439-5EE8-DD11-BDF5-001D09F290CE.root',
#    '/store/relval/CMSSW_3_0_0_pre7/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0006/B2EDD437-5EE8-DD11-90E4-001D09F24D67.root',
#    '/store/relval/CMSSW_3_0_0_pre7/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0006/C22E344B-5EE8-DD11-8D8A-001D09F29619.root',
#    '/store/relval/CMSSW_3_0_0_pre7/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0006/EC17B8CA-60E8-DD11-AB03-001D09F29597.root'
    
    
    )
 )



from SimTracker.TrackAssociation.TrackAssociatorByHits_cfi import *
import SimTracker.TrackAssociation.TrackAssociatorByHits_cfi
#TrackAssociatorByHits.AbsoluteNumberOfHits = True
#TrackAssociatorByHits.Cut_RecoToSim = 3
#TrackAssociatorByHits.Quality_SimToReco = 3
TrackAssociatorByHits.Cut_RecoToSim = 0.5
TrackAssociatorByHits.Quality_SimToReco = 0.5

photonValidation.minPhoEtCut = 10
photonValidation.eMax = 500
photonValidation.etMax = 150
photonValidation.convTrackMinPtCut = 1.


process.FEVT = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring("keep *_MEtoEDMConverter_*_*"),
    fileName = cms.untracked.string('pippo.root')
)

process.p1 = cms.Path(process.tpSelection*process.photonValidation)
#process.p1 = cms.Path(process.mix*process.trackingParticles*process.tpSelection*process.photonValidation)
process.schedule = cms.Schedule(process.p1)



