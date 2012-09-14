import FWCore.ParameterSet.Config as cms

process = cms.Process("HSCPAnalysis")
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) )

process.GlobalTag.globaltag = 'START42_V9::All'

process.source = cms.Source("PoolSource",
   fileNames = cms.untracked.vstring(
'file:/uscmst1b_scratch/lpc1/lpcphys/jchen/HSCPRawData/STEP2_RAW2DIGI_L1Reco_RECO_PU_50_1_w2Z.root'
   )
)
process.source.inputCommands = cms.untracked.vstring("keep *", "drop *_MEtoEDMConverter_*_*")



########################################################################
process.load('SUSYBSMAnalysis.Skimming.EXOHSCP_cff')
process.load('SUSYBSMAnalysis.Skimming.EXOHSCP_EventContent_cfi')
process.load("SUSYBSMAnalysis.HSCP.HSCParticleProducerFromSkim_cff")  #IF RUNNING ON HSCP SKIM

########################################################################  SPECIAL CASE FOR MC

process.GlobalTag.toGet = cms.VPSet(
   cms.PSet( record = cms.string('SiStripDeDxMip_3D_Rcd'),
            tag = cms.string('MC7TeV_Deco_3D_Rcd_38X'),
            connect = cms.untracked.string("sqlite_file:MC7TeV_Deco_SiStripDeDxMip_3D_Rcd.db")),
)

process.load("PhysicsTools.HepMCCandAlgos.genParticles_cfi")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.genParticles.abortOnUnknownPDGCode = cms.untracked.bool(False)

process.generalTracksSkim.filter       = cms.bool(False)
process.HSCParticleProducer.filter     = cms.bool(False)
process.DedxFilter.filter              = cms.bool(False)
#process.HSCPTreeBuilder.reccordGenInfo = cms.untracked.bool(True)

process.dedxHarm2.calibrationPath      = cms.string("file:Gains.root")
process.dedxTru40.calibrationPath      = cms.string("file:Gains.root")
process.dedxProd.calibrationPath       = cms.string("file:Gains.root")
process.dedxASmi.calibrationPath       = cms.string("file:Gains.root")
process.dedxNPHarm2.calibrationPath    = cms.string("file:Gains.root")
process.dedxNPTru40.calibrationPath    = cms.string("file:Gains.root")
process.dedxNSHarm2.calibrationPath    = cms.string("file:Gains.root")
process.dedxNSTru40.calibrationPath    = cms.string("file:Gains.root")
process.dedxNPProd.calibrationPath     = cms.string("file:Gains.root")
process.dedxNPASmi.calibrationPath     = cms.string("file:Gains.root")

process.dedxHarm2.UseCalibration       = cms.bool(True)
process.dedxTru40.UseCalibration       = cms.bool(True)
process.dedxProd.UseCalibration        = cms.bool(True)
process.dedxASmi.UseCalibration        = cms.bool(True)
process.dedxNPHarm2.UseCalibration     = cms.bool(True)
process.dedxNPTru40.UseCalibration     = cms.bool(True)
process.dedxNSHarm2.UseCalibration     = cms.bool(True)
process.dedxNSTru40.UseCalibration     = cms.bool(True)
process.dedxNPProd.UseCalibration      = cms.bool(True)
process.dedxNPASmi.UseCalibration      = cms.bool(True)


process.dedxHarm2.MeVperADCStrip = cms.double(3.61e-06*265)
process.dedxTru40.MeVperADCStrip = cms.double(3.61e-06*265) 
process.dedxProd.MeVperADCStrip = cms.double(3.61e-06*265)   
process.dedxASmi.MeVperADCStrip = cms.double(3.61e-06*265)   
process.dedxNPHarm2.MeVperADCStrip = cms.double(3.61e-06*265)
process.dedxNPTru40.MeVperADCStrip = cms.double(3.61e-06*265)
process.dedxNSHarm2.MeVperADCStrip = cms.double(3.61e-06*265)
process.dedxNSTru40.MeVperADCStrip = cms.double(3.61e-06*265)
process.dedxNPProd.MeVperADCStrip = cms.double(3.61e-06*265)  
process.dedxNPASmi.MeVperADCStrip = cms.double(3.61e-06*265)   

process.load("RecoLocalMuon.DTSegment.dt4DSegments_MTPatternReco4D_LinearDriftFromDBLoose_cfi")
process.dt4DSegments.Reco4DAlgoConfig.Reco2DAlgoConfig.AlphaMaxPhi = 1.0
process.dt4DSegments.Reco4DAlgoConfig.Reco2DAlgoConfig.AlphaMaxTheta = 0.9
process.dt4DSegments.Reco4DAlgoConfig.Reco2DAlgoConfig.segmCleanerMode = 2
process.dt4DSegments.Reco4DAlgoConfig.Reco2DAlgoConfig.MaxChi2 = 1.0
process.dt4DSegmentsMT = process.dt4DSegments.clone()
process.dt4DSegmentsMT.Reco4DAlgoConfig.recAlgoConfig.stepTwoFromDigi = True
process.dt4DSegmentsMT.Reco4DAlgoConfig.Reco2DAlgoConfig.recAlgoConfig.stepTwoFromDigi = True

process.muontiming.TimingFillerParameters.DTTimingParameters.MatchParameters.DTsegments = "dt4DSegmentsMT"
process.muontiming.TimingFillerParameters.DTTimingParameters.HitsMin = 3
process.muontiming.TimingFillerParameters.DTTimingParameters.RequireBothProjections = False
process.muontiming.TimingFillerParameters.DTTimingParameters.DropTheta = True
process.muontiming.TimingFillerParameters.DTTimingParameters.DoWireCorr = True
process.muontiming.TimingFillerParameters.DTTimingParameters.MatchParameters.DTradius = 1.0


########################################################################
process.nEventsBefSkim  = cms.EDProducer("EventCountProducer")
process.nEventsBefEDM   = cms.EDProducer("EventCountProducer")
########################################################################


process.OUT = cms.OutputModule("PoolOutputModule",
     outputCommands = cms.untracked.vstring(
         "drop *",
         'keep EventAux_*_*_*',
         'keep LumiSummary_*_*_*',
         'keep edmMergeableCounter_*_*_*',
         "keep *_genParticles_*_HSCPAnalysis",
         "keep GenEventInfoProduct_generator_*_*",
         "keep *_offlinePrimaryVertices_*_*",
#         "keep *_csc2DRecHits_*_*",
#         "keep *_cscSegments_*_*",
#         "keep *_dt1DRecHits_*_*",
#         "keep *_rpcRecHits_*_*",
#         "keep *_dt4DSegments_*_*",
         "keep SiStripClusteredmNewDetSetVector_generalTracksSkim_*_*",
         "keep SiPixelClusteredmNewDetSetVector_generalTracksSkim_*_*",
#         "keep *_reducedHSCPhbhereco_*_*",
#         "keep *_reducedHSCPEcalRecHitsEB_*_*",
#         "keep *_reducedHSCPEcalRecHitsEE_*_*",
         "keep *_TrackRefitter_*_*",
         "drop TrajectorysToOnerecoTracksAssociation_TrackRefitter__",
         "keep *_standAloneMuons_*_*",
         "drop recoTracks_standAloneMuons__*",
         "keep *_globalMuons_*_*",
         "keep *_muonsSkim_*_*",
#         "keep L1GlobalTriggerReadoutRecord_gtDigis_*_*",
         "keep edmTriggerResults_TriggerResults_*_*",
         "keep recoPFJets_ak5PFJets__*",
         "keep recoPFMETs_pfMet__*",
#         "keep recoCaloJets_ak5CaloJets__*",
         "keep *_HSCParticleProducer_*_*",
         "keep *_HSCPIsolation01__*",
         "keep *_HSCPIsolation03__*",
         "keep *_HSCPIsolation05__*",
         "keep *_dedx*_*_HSCPAnalysis",
         "keep *_muontiming_*_HSCPAnalysis",
		 "keep *_g4SimHits_StoppedParticles*_*",		 
         "keep triggerTriggerEvent_hltTriggerSummaryAOD_*_*",
         "keep PileupSummaryInfos_addPileupInfo_*_*"
    ),
    fileName = cms.untracked.string('HSCP.root'),
#    SelectEvents = cms.untracked.PSet(
#       SelectEvents = cms.vstring('p1')
#    ),
)

########################################################################


#LOOK AT SD PASSED PATH IN ORDER to avoid as much as possible duplicated events (make the merging of .root file faster)
process.p1 = cms.Path(process.nEventsBefSkim + process.genParticles + process.exoticaHSCPSeq + process.nEventsBefEDM + process.dt4DSegmentsMT * process.HSCParticleProducerSeq)
process.endPath = cms.EndPath(process.OUT)


