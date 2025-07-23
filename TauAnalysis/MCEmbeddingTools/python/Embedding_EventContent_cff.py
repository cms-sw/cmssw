import FWCore.ParameterSet.Config as cms

TauEmbSelection = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_selectedMuonsForEmbedding_*_SELECT',
        'keep *_patMuonsAfterID_*_SELECT',
        'keep *_slimmedMuons_*_SELECT',
        'keep *_slimmedMuonTrackExtras_*_SELECT',
        'keep recoVertexs_offlineSlimmedPrimaryVertices_*_SELECT',
        'keep *_firstStepPrimaryVertices_*_SELECT',
        'keep *_ecalDrivenElectronSeeds_*_SELECT'
    )
)

TauEmbCleaning = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'drop *_*_*_SELECT',
        'drop recoIsoDepositedmValueMap_muIsoDepositTk_*_*',
        'drop recoIsoDepositedmValueMap_muIsoDepositTkDisplaced_*_*',
        'drop *_ctppsProtons_*_*',
        'drop *_ctppsLocalTrackLiteProducer_*_*',
        'drop *_ctppsDiamondLocalTracks_*_*',
        'drop *_ctppsDiamondRecHits_*_*',
        'drop *_ctppsDiamondRawToDigi_*_*',
        'drop *_ctppsPixelLocalTracks_*_*',
        'drop *_ctppsPixelRecHits_*_*',
        'drop *_ctppsPixelClusters_*_*',
        'drop *_ctppsPixelDigis_*_*',
        'drop *_totemRPLocalTrackFitter_*_*',
        'drop *_totemRPUVPatternFinder_*_*',
        'drop *_totemRPRecHitProducer_*_*',
        'drop *_totemRPClusterProducer_*_*',
        'drop *_totemRPRawToDigi_*_*',
        'drop *_muonSimClassifier_*_*',
        'keep *_patMuonsAfterID_*_SELECT',
        'keep *_slimmedMuons_*_SELECT',
        'keep *_selectedMuonsForEmbedding_*_SELECT',
        'keep recoVertexs_offlineSlimmedPrimaryVertices_*_SELECT',
        'keep *_firstStepPrimaryVertices_*_SELECT',
        'keep *_offlineBeamSpot_*_SELECT',
        'keep *_l1extraParticles_*_SELECT',
        'keep TrajectorySeeds_*_*_*',
        'keep recoElectronSeeds_*_*_*',
        'keep *_generalTracks_*_LHEembeddingCLEAN',
        'keep *_generalTracks_*_CLEAN',
        'keep *_cosmicsVetoTracksRaw_*_LHEembeddingCLEAN',
        'keep *_cosmicsVetoTracksRaw_*_CLEAN',
        'keep *_electronGsfTracks_*_LHEembeddingCLEAN',
        'keep *_electronGsfTracks_*_CLEAN',
        'keep *_lowPtGsfEleGsfTracks_*_LHEembeddingCLEAN',
        'keep *_lowPtGsfEleGsfTracks_*_CLEAN',
        'keep *_displacedTracks_*_LHEembeddingCLEAN',
        'keep *_displacedTracks_*_CLEAN',
        'keep *_ckfOutInTracksFromConversions_*_LHEembeddingCLEAN',
        'keep *_ckfOutInTracksFromConversions_*_CLEAN',
        'keep *_muons1stStep_*_LHEembeddingCLEAN',
        'keep *_muons1stStep_*_CLEAN',
        'keep *_displacedMuons1stStep_*_LHEembeddingCLEAN',
        'keep *_displacedMuons1stStep_*_CLEAN',
        'keep *_conversions_*_LHEembeddingCLEAN',
        'keep *_conversions_*_CLEAN',
        'keep *_allConversions_*_LHEembeddingCLEAN',
        'keep *_allConversions_*_CLEAN',
        'keep *_particleFlowTmp_*_LHEembeddingCLEAN',
        'keep *_particleFlowTmp_*_CLEAN',
        'keep *_ecalDigis_*_LHEembeddingCLEAN',
        'keep *_ecalDigis_*_CLEAN',
        'keep *_hcalDigis_*_LHEembeddingCLEAN',
        'keep *_hcalDigis_*_CLEAN',
        'keep *_ecalRecHit_*_LHEembeddingCLEAN',
        'keep *_ecalRecHit_*_CLEAN',
        'keep *_ecalPreshowerRecHit_*_LHEembeddingCLEAN',
        'keep *_ecalPreshowerRecHit_*_CLEAN',
        'keep *_hbhereco_*_LHEembeddingCLEAN',
        'keep *_hbhereco_*_CLEAN',
        'keep *_horeco_*_LHEembeddingCLEAN',
        'keep *_horeco_*_CLEAN',
        'keep *_hfreco_*_LHEembeddingCLEAN',
        'keep *_hfreco_*_CLEAN',
        'keep *_standAloneMuons_*_LHEembeddingCLEAN',
        'keep *_glbTrackQual_*_LHEembeddingCLEAN',
        'keep *_externalLHEProducer_*_LHEembedding',
        'keep *_externalLHEProducer_*_LHEembeddingCLEAN'
    )
)

TauEmbSimGen = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_*_*_LHEembeddingCLEAN',
        'keep *_*_*_SELECT',
        'drop *_muonReducedTrackExtras_*_*',
        'drop *_*_uncleanedConversions_*',
        'drop *_diamondSampicLocalTracks_*_*',
        'keep *_*_unsmeared_*'
    )
)

TauEmbSimHLT = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_*_*_SELECT',
        'keep *_*_*_LHEembeddingCLEAN',
        'keep *_*_unsmeared_SIMembeddingpreHLT',
        'keep DcsStatuss_hltScalersRawToDigi_*_*'
    )
)

TauEmbSimReco = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_*_*_LHEembeddingCLEAN',
        'keep *_*_*_SELECT',
        'keep *_genParticles_*_SIMembedding',
        'keep *_standAloneMuons_*_SIMembedding',
        'keep *_glbTrackQual_*_SIMembedding',
        'keep *_generator_*_SIMembedding',
        'keep *_addPileupInfo_*_SIMembedding',
        'keep *_selectedMuonsForEmbedding_*_*',
        'keep *_slimmedAddPileupInfo_*_*',
        'keep *_embeddingHltPixelVertices_*_*',
        'keep *_*_vertexPosition_*',
        'keep recoMuons_muonsFromCosmics_*_*',
        'keep recoTracks_cosmicMuons1Leg_*_*',
        'keep recoMuons_muonsFromCosmics1Leg_*_*',
        'keep *_muonDTDigis_*_*',
        'keep *_muonCSCDigis_*_*',
        'keep TrajectorySeeds_*_*_*',
        'keep recoElectronSeeds_*_*_*',
        'drop recoIsoDepositedmValueMap_muIsoDepositTk_*_*',
        'drop recoIsoDepositedmValueMap_muIsoDepositTkDisplaced_*_*',
        'drop *_ctppsProtons_*_*',
        'drop *_ctppsLocalTrackLiteProducer_*_*',
        'drop *_ctppsDiamondLocalTracks_*_*',
        'drop *_ctppsDiamondRecHits_*_*',
        'drop *_ctppsDiamondRawToDigi_*_*',
        'drop *_ctppsPixelLocalTracks_*_*',
        'drop *_ctppsPixelRecHits_*_*',
        'drop *_ctppsPixelClusters_*_*',
        'drop *_ctppsPixelDigis_*_*',
        'drop *_totemRPLocalTrackFitter_*_*',
        'drop *_totemRPUVPatternFinder_*_*',
        'drop *_totemRPRecHitProducer_*_*',
        'drop *_totemRPClusterProducer_*_*',
        'drop *_totemRPRawToDigi_*_*',
        'drop *_muonSimClassifier_*_*',
        'keep *_generalTracks_*_SIMembedding',
        'keep *_cosmicsVetoTracksRaw_*_SIMembedding',
        'keep *_electronGsfTracks_*_SIMembedding',
        'keep *_lowPtGsfEleGsfTracks_*_SIMembedding',
        'keep *_displacedTracks_*_SIMembedding',
        'keep *_ckfOutInTracksFromConversions_*_SIMembedding',
        'keep *_muons1stStep_*_SIMembedding',
        'keep *_displacedMuons1stStep_*_SIMembedding',
        'keep *_conversions_*_SIMembedding',
        'keep *_allConversions_*_SIMembedding',
        'keep *_particleFlowTmp_*_SIMembedding',
        'keep *_ecalDigis_*_SIMembedding',
        'keep *_hcalDigis_*_SIMembedding',
        'keep *_ecalRecHit_*_SIMembedding',
        'keep *_ecalPreshowerRecHit_*_SIMembedding',
        'keep *_hbhereco_*_SIMembedding',
        'keep *_horeco_*_SIMembedding',
        'keep *_hfreco_*_SIMembedding',
        'keep *_offlinePrimaryVertices_*_SIMembedding',
        'keep *_*_unsmeared_SIMembeddingpreHLT',
        'keep *_hltScalersRawToDigi_*_SIMembeddingHLT'
    )
)

TauEmbMerge = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'drop *_*_*_SELECT',
        'keep *_prunedGenParticles_*_MERGE',
        'keep *_generator_*_SIMembeddingpreHLT',
        'keep *_generator_*_SIMembeddingHLT',
        'keep *_generator_*_SIMembedding',
        'keep *_selectedMuonsForEmbedding_*_*',
        'keep *_unpackedPatTrigger_*_*',
        'keep patPackedGenParticles_packedGenParticles_*_*',
        'keep recoGenParticles_prunedGenParticles_*_*',
        'keep *_packedPFCandidateToGenAssociation_*_*',
        'keep *_lostTracksToGenAssociation_*_*',
        'keep LHEEventProduct_*_*_*',
        'keep GenFilterInfo_*_*_*',
        'keep GenLumiInfoHeader_generator_*_*',
        'keep GenLumiInfoProduct_*_*_*',
        'keep GenEventInfoProduct_generator_*_*',
        'keep recoGenParticles_genPUProtons_*_*',
        'keep *_slimmedGenJetsFlavourInfos_*_*',
        'keep *_slimmedGenJets__*',
        'keep *_slimmedGenJetsAK8__*',
        'keep *_slimmedGenJetsAK8SoftDropSubJets__*',
        'keep *_genMetTrue_*_*',
        'keep LHERunInfoProduct_*_*_*',
        'keep GenRunInfoProduct_*_*_*',
        'keep *_genParticles_xyz0_*',
        'keep *_genParticles_t0_*'
    )
)

TauEmbNano = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep edmTriggerResults_*_*_SIMembeddingpreHLT',
        'keep edmTriggerResults_*_*_SIMembeddingHLT',
        'keep edmTriggerResults_*_*_SIMembedding',
        'keep edmTriggerResults_*_*_MERGE',
        'keep edmTriggerResults_*_*_NANO'
    )
)