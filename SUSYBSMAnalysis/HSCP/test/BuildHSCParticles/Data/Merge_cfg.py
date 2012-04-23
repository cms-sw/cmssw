import FWCore.ParameterSet.Config as cms
process = cms.Process("MergeHLT")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
process.load("FWCore.MessageService.MessageLogger_cfi")

process.MessageLogger.cerr.FwkReport.reportEvery = 5000
process.source = cms.Source("PoolSource",
   fileNames = cms.untracked.vstring(
#XXX_INPUT_XXX
"file:Run191001_ElHAD/res/HSCP_10_1_0IQ.root",
"file:Run191001_ElHAD/res/HSCP_11_2_cta.root",
"file:Run191001_ElHAD/res/HSCP_12_1_TPZ.root",
"file:Run191001_ElHAD/res/HSCP_13_1_yvV.root",
"file:Run191001_ElHAD/res/HSCP_14_2_H2l.root",
"file:Run191001_ElHAD/res/HSCP_15_1_dMm.root",
"file:Run191001_ElHAD/res/HSCP_16_1_Ong.root",
"file:Run191001_ElHAD/res/HSCP_17_1_FWz.root",
"file:Run191001_ElHAD/res/HSCP_18_1_0no.root",
"file:Run191001_ElHAD/res/HSCP_19_1_v65.root",
"file:Run191001_ElHAD/res/HSCP_1_1_tue.root",
"file:Run191001_ElHAD/res/HSCP_20_1_Z3m.root",
"file:Run191001_ElHAD/res/HSCP_21_1_n4o.root",
"file:Run191001_ElHAD/res/HSCP_22_1_gmf.root",
"file:Run191001_ElHAD/res/HSCP_23_1_FEQ.root",
"file:Run191001_ElHAD/res/HSCP_24_1_edI.root",
"file:Run191001_ElHAD/res/HSCP_25_1_r0o.root",
"file:Run191001_ElHAD/res/HSCP_26_1_7kM.root",
"file:Run191001_ElHAD/res/HSCP_27_1_70m.root",
"file:Run191001_ElHAD/res/HSCP_28_1_tyQ.root",
"file:Run191001_ElHAD/res/HSCP_29_1_KGI.root",
"file:Run191001_ElHAD/res/HSCP_2_1_1z3.root",
"file:Run191001_ElHAD/res/HSCP_30_1_Ps4.root",
"file:Run191001_ElHAD/res/HSCP_31_1_g7g.root",
"file:Run191001_ElHAD/res/HSCP_32_1_rlo.root",
"file:Run191001_ElHAD/res/HSCP_33_1_8KH.root",
"file:Run191001_ElHAD/res/HSCP_34_1_iNi.root",
"file:Run191001_ElHAD/res/HSCP_35_1_FFj.root",
"file:Run191001_ElHAD/res/HSCP_36_1_9T7.root",
"file:Run191001_ElHAD/res/HSCP_37_1_Ljy.root",
"file:Run191001_ElHAD/res/HSCP_3_1_yRE.root",
"file:Run191001_ElHAD/res/HSCP_4_1_Hi0.root",
"file:Run191001_ElHAD/res/HSCP_5_1_lNJ.root",
"file:Run191001_ElHAD/res/HSCP_6_1_9lY.root",
"file:Run191001_ElHAD/res/HSCP_7_1_ZGK.root",
"file:Run191001_ElHAD/res/HSCP_8_1_hwJ.root",
"file:Run191001_ElHAD/res/HSCP_9_1_2KW.root",
"file:Run191001_SingleMu/res/HSCP_10_2_GGY.root",
"file:Run191001_SingleMu/res/HSCP_11_2_Of2.root",
"file:Run191001_SingleMu/res/HSCP_12_1_eny.root",
"file:Run191001_SingleMu/res/HSCP_13_1_acD.root",
"file:Run191001_SingleMu/res/HSCP_14_1_Qv8.root",
"file:Run191001_SingleMu/res/HSCP_15_1_fv0.root",
"file:Run191001_SingleMu/res/HSCP_16_2_vuL.root",
"file:Run191001_SingleMu/res/HSCP_17_1_eJd.root",
"file:Run191001_SingleMu/res/HSCP_18_1_6qH.root",
"file:Run191001_SingleMu/res/HSCP_19_1_TRG.root",
"file:Run191001_SingleMu/res/HSCP_1_2_CkN.root",
"file:Run191001_SingleMu/res/HSCP_20_1_6Nq.root",
"file:Run191001_SingleMu/res/HSCP_21_1_y6y.root",
"file:Run191001_SingleMu/res/HSCP_22_1_7Nk.root",
"file:Run191001_SingleMu/res/HSCP_24_1_Gm6.root",
"file:Run191001_SingleMu/res/HSCP_25_1_ryX.root",
"file:Run191001_SingleMu/res/HSCP_26_1_TJm.root",
"file:Run191001_SingleMu/res/HSCP_27_1_BKI.root",
"file:Run191001_SingleMu/res/HSCP_28_1_FPN.root",
"file:Run191001_SingleMu/res/HSCP_29_1_ZJw.root",
"file:Run191001_SingleMu/res/HSCP_2_1_Yki.root",
"file:Run191001_SingleMu/res/HSCP_30_1_wYj.root",
"file:Run191001_SingleMu/res/HSCP_31_1_xxq.root",
"file:Run191001_SingleMu/res/HSCP_32_1_1VH.root",
"file:Run191001_SingleMu/res/HSCP_33_1_oKL.root",
"file:Run191001_SingleMu/res/HSCP_34_1_zJ9.root",
"file:Run191001_SingleMu/res/HSCP_35_1_tuE.root",
"file:Run191001_SingleMu/res/HSCP_36_1_cgQ.root",
"file:Run191001_SingleMu/res/HSCP_37_1_K8i.root",
"file:Run191001_SingleMu/res/HSCP_38_1_RO7.root",
"file:Run191001_SingleMu/res/HSCP_39_1_pjy.root",
"file:Run191001_SingleMu/res/HSCP_3_1_Onb.root",
"file:Run191001_SingleMu/res/HSCP_40_1_E3H.root",
"file:Run191001_SingleMu/res/HSCP_41_1_zQU.root",
"file:Run191001_SingleMu/res/HSCP_42_1_XDq.root",
"file:Run191001_SingleMu/res/HSCP_43_1_9s8.root",
"file:Run191001_SingleMu/res/HSCP_4_1_WMz.root",
"file:Run191001_SingleMu/res/HSCP_5_1_TzZ.root",
"file:Run191001_SingleMu/res/HSCP_6_1_unb.root",
"file:Run191001_SingleMu/res/HSCP_7_1_vAd.root",
"file:Run191001_SingleMu/res/HSCP_8_1_cHk.root",
"file:Run191001_SingleMu/res/HSCP_9_1_4w4.root",

   )
)

process.HSCPHLTDuplicate = cms.EDFilter("HSCPHLTFilter",
   RemoveDuplicates = cms.bool(True),
   TriggerProcess   = cms.string("HLT"),
   MuonTrigger1Mask    = cms.int32(1),  #Activated
   PFMetTriggerMask    = cms.int32(1),  #Activated
)

process.HSCPHLTFilterPFMET = cms.EDFilter("HSCPHLTFilter",
   RemoveDuplicates = cms.bool(False),
   TriggerProcess   = cms.string("HLT"),
   MuonTrigger1Mask    = cms.int32(0),  #Activated
   PFMetTriggerMask    = cms.int32(1),  #Activated
)


process.HSCPHLTFilterSingleMU = cms.EDFilter("HSCPHLTFilter",
   RemoveDuplicates = cms.bool(False),
   TriggerProcess  = cms.string("HLT"),
   MuonTrigger1Mask    = cms.int32(1),  #Activated
   PFMetTriggerMask    = cms.int32(0),  #Activated
)


process.Filter      = cms.Path(process.HSCPHLTDuplicate   )
process.HscpPathPFMet = cms.Path(process.HSCPHLTFilterPFMET   )
process.HscpPathSingleMu  = cms.Path(process.HSCPHLTFilterSingleMU    )


process.Out = cms.OutputModule("PoolOutputModule",
     outputCommands = cms.untracked.vstring(
         "drop *",
         'keep EventAux_*_*_*',
         'keep LumiSummary_*_*_*',
         'keep edmMergeableCounter_*_*_*',
         "keep *_genParticles_*_*",
         "keep GenEventInfoProduct_generator_*_*",
         "keep *_offlinePrimaryVertices_*_*",
         #"keep *_cscSegments_*_*",
         #"keep *_rpcRecHits_*_*",
         #"keep *_dt4DSegments_*_*",
         "keep SiStripClusteredmNewDetSetVector_generalTracksSkim_*_*",
         "keep SiPixelClusteredmNewDetSetVector_generalTracksSkim_*_*",
         #"keep *_reducedHSCPhbhereco_*_*",      #
         #"keep *_reducedHSCPEcalRecHitsEB_*_*", #
         #"keep *_reducedHSCPEcalRecHitsEE_*_*", #
         "keep *_TrackRefitter_*_*",
         "drop TrajectorysToOnerecoTracksAssociation_TrackRefitter__",
         "keep *_standAloneMuons_*_*",
         #"drop recoTracks_standAloneMuons__*",
         "keep *_globalMuons_*_*",  #
         "keep *_muonsSkim_*_*",
         "keep edmTriggerResults_TriggerResults_*_*",
         "keep recoPFJets_ak5PFJets__*", #
         "keep recoPFMETs_pfMet__*",     #
         "keep *_HSCParticleProducer_*_*",
         "keep *_HSCPIsolation01__*",
         "keep *_HSCPIsolation03__*",
         "keep *_HSCPIsolation05__*",
         "keep *_dedx*_*_HSCPAnalysis",
         "keep *_muontiming_*_HSCPAnalysis",
         "keep triggerTriggerEvent_hltTriggerSummaryAOD_*_*",
    ),
#    fileName = cms.untracked.string('/uscmst1b_scratch/lpc1/3DayLifetime/farrell/NewDTError/XXX_OUTPUT_XXX.root'),
    fileName = cms.untracked.string('file:191XXX.root'),
    SelectEvents = cms.untracked.PSet(
       SelectEvents = cms.vstring('Filter')
    ),
)

process.endPath = cms.EndPath(process.Out)

#process.schedule = cms.Schedule(process.Filter, process.HscpPathPFMet, process.HscpPathSingleMu, process.endPath)
process.schedule = cms.Schedule(process.Filter, process.endPath)


