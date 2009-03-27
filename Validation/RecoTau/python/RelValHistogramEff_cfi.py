import FWCore.ParameterSet.Config as cms

TauEfficiencies = cms.EDAnalyzer("DQMHistEffProducer",
    plots = cms.PSet(
# REGULAR PFTAU EFFICIENCIES CALCULATION
      PFTauIDMatchingEfficiencies = cms.PSet(
        numerator = cms.string('RecoTauV/fixedConePFTauProducer_Matched/fixedConePFTauProducerMatched_vs_#PAR#TauVisible'),
        denominator = cms.string('RecoTauV/fixedConePFTauProducer_ReferenceCollection/nRef_Taus_vs_#PAR#TauVisible'),
        efficiency = cms.string('RecoTauV/fixedConePFTauProducer_Matched/PFJetMatchingEff#PAR#'),
        parameter = cms.vstring('pt', 'eta', 'phi', 'energy')
      ),
      PFTauIDLeadingTrackFindEfficiencies = cms.PSet(
        numerator = cms.string('RecoTauV/fixedConePFTauProducer_fixedConePFTauDiscriminationByLeadingTrackFinding/fixedConePFTauDiscriminationByLeadingTrackFinding_vs_#PAR#TauVisible'),
        denominator = cms.string('RecoTauV/fixedConePFTauProducer_ReferenceCollection/nRef_Taus_vs_#PAR#TauVisible'),
        efficiency = cms.string('RecoTauV/fixedConePFTauProducer_fixedConePFTauDiscriminationByLeadingTrackFinding/LeadingTrackFindingEff#PAR#'),
        parameter = cms.vstring('pt', 'eta', 'phi', 'energy')
      ),
      PFTauIDLeadingTrackPtCutEfficiencies = cms.PSet(
        numerator = cms.string('RecoTauV/fixedConePFTauProducer_fixedConePFTauDiscriminationByLeadingTrackPtCut/fixedConePFTauDiscriminationByLeadingTrackPtCut_vs_#PAR#TauVisible'),
        denominator = cms.string('RecoTauV/fixedConePFTauProducer_ReferenceCollection/nRef_Taus_vs_#PAR#TauVisible'),
        efficiency = cms.string('RecoTauV/fixedConePFTauProducer_fixedConePFTauDiscriminationByLeadingTrackPtCut/LeadingTrackPtCutEff#PAR#'),
        parameter = cms.vstring('pt', 'eta', 'phi', 'energy')
      ),
      PFTauIDTrackIsolationEfficienies = cms.PSet(
        numerator = cms.string('RecoTauV/fixedConePFTauProducer_fixedConePFTauDiscriminationByTrackIsolation/fixedConePFTauDiscriminationByTrackIsolation_vs_#PAR#TauVisible'),
        denominator = cms.string('RecoTauV/fixedConePFTauProducer_ReferenceCollection/nRef_Taus_vs_#PAR#TauVisible'),
        efficiency = cms.string('RecoTauV/fixedConePFTauProducer_fixedConePFTauDiscriminationByTrackIsolation/TrackIsolationEff#PAR#'),
        parameter = cms.vstring('pt', 'eta', 'phi', 'energy')
      ),
      PFTauIDECALIsolationEfficienies = cms.PSet(
        numerator = cms.string('RecoTauV/fixedConePFTauProducer_fixedConePFTauDiscriminationByECALIsolation/fixedConePFTauDiscriminationByECALIsolation_vs_#PAR#TauVisible'),
        denominator = cms.string('RecoTauV/fixedConePFTauProducer_ReferenceCollection/nRef_Taus_vs_#PAR#TauVisible'),
        efficiency = cms.string('RecoTauV/fixedConePFTauProducer_fixedConePFTauDiscriminationByECALIsolation/ECALIsolationEff#PAR#'),
        parameter = cms.vstring('pt', 'eta', 'phi', 'energy')
      ),
      PFTauIDMuonRejectionEfficiencies = cms.PSet(
        numerator = cms.string('RecoTauV/fixedConePFTauProducer_fixedConePFTauDiscriminationAgainstElectron/fixedConePFTauDiscriminationAgainstElectron_vs_#PAR#TauVisible'),
        denominator = cms.string('RecoTauV/fixedConePFTauProducer_ReferenceCollection/nRef_Taus_vs_#PAR#TauVisible'),
        efficiency = cms.string('RecoTauV/fixedConePFTauProducer_fixedConePFTauDiscriminationAgainstElectron/AgainstElectronEff#PAR#'),
        parameter = cms.vstring('pt', 'eta', 'phi', 'energy')
      ),
      PFTauIDElectronRejectionEfficiencies = cms.PSet(
        numerator = cms.string('RecoTauV/fixedConePFTauProducer_fixedConePFTauDiscriminationAgainstMuon/fixedConePFTauDiscriminationAgainstMuon_vs_#PAR#TauVisible'),
        denominator = cms.string('RecoTauV/fixedConePFTauProducer_ReferenceCollection/nRef_Taus_vs_#PAR#TauVisible'),
        efficiency = cms.string('RecoTauV/fixedConePFTauProducer_fixedConePFTauDiscriminationAgainstMuon/AgainstMuonEff#PAR#'),
        parameter = cms.vstring('pt', 'eta', 'phi', 'energy')
      ),
# PFTAUHIGHEFFICIENCY EFFICIENCY CALCULATION
      PFTauHighEfficiencyIDMatchingEfficiencies = cms.PSet(
        numerator = cms.string('RecoTauV/shrinkingConePFTauProducer_Matched/shrinkingConePFTauProducerMatched_vs_#PAR#TauVisible'),
        denominator = cms.string('RecoTauV/shrinkingConePFTauProducer_ReferenceCollection/nRef_Taus_vs_#PAR#TauVisible'),
        efficiency = cms.string('RecoTauV/shrinkingConePFTauProducer_Matched/PFJetMatchingEff#PAR#'),
        parameter = cms.vstring('pt', 'eta', 'phi', 'energy')
      ),
      PFTauHighEfficiencyIDLeadingTrackFindEfficiencies = cms.PSet(
        numerator = cms.string('RecoTauV/shrinkingConePFTauProducer_shrinkingConePFTauDiscriminationByLeadingTrackFinding/shrinkingConePFTauDiscriminationByLeadingTrackFinding_vs_#PAR#TauVisible'),
        denominator = cms.string('RecoTauV/shrinkingConePFTauProducer_ReferenceCollection/nRef_Taus_vs_#PAR#TauVisible'),
        efficiency = cms.string('RecoTauV/shrinkingConePFTauProducer_shrinkingConePFTauDiscriminationByLeadingTrackFinding/LeadingTrackFindingEff#PAR#'),
        parameter = cms.vstring('pt', 'eta', 'phi', 'energy')
      ),
      PFTauHighEfficiencyIDLeadingTrackPtCutEfficiencies = cms.PSet(
        numerator = cms.string('RecoTauV/shrinkingConePFTauProducer_shrinkingConePFTauDiscriminationByLeadingTrackPtCut/shrinkingConePFTauDiscriminationByLeadingTrackPtCut_vs_#PAR#TauVisible'),
        denominator = cms.string('RecoTauV/shrinkingConePFTauProducer_ReferenceCollection/nRef_Taus_vs_#PAR#TauVisible'),
        efficiency = cms.string('RecoTauV/shrinkingConePFTauProducer_shrinkingConePFTauDiscriminationByLeadingTrackPtCut/LeadingTrackPtCutEff#PAR#'),
        parameter = cms.vstring('pt', 'eta', 'phi', 'energy')
      ),
      PFTauHighEfficiencyIDTrackIsolationEfficienies = cms.PSet(
        numerator = cms.string('RecoTauV/shrinkingConePFTauProducer_shrinkingConePFTauDiscriminationByTrackIsolation/shrinkingConePFTauDiscriminationByTrackIsolation_vs_#PAR#TauVisible'),
        denominator = cms.string('RecoTauV/shrinkingConePFTauProducer_ReferenceCollection/nRef_Taus_vs_#PAR#TauVisible'),
        efficiency = cms.string('RecoTauV/shrinkingConePFTauProducer_shrinkingConePFTauDiscriminationByTrackIsolation/TrackIsolationEff#PAR#'),
        parameter = cms.vstring('pt', 'eta', 'phi', 'energy')
      ),
      PFTauHighEfficiencyIDECALIsolationEfficienies = cms.PSet(
        numerator = cms.string('RecoTauV/shrinkingConePFTauProducer_shrinkingConePFTauDiscriminationByECALIsolation/shrinkingConePFTauDiscriminationByECALIsolation_vs_#PAR#TauVisible'),
        denominator = cms.string('RecoTauV/shrinkingConePFTauProducer_ReferenceCollection/nRef_Taus_vs_#PAR#TauVisible'),
        efficiency = cms.string('RecoTauV/shrinkingConePFTauProducer_shrinkingConePFTauDiscriminationByECALIsolation/ECALIsolationEff#PAR#'),
        parameter = cms.vstring('pt', 'eta', 'phi', 'energy')
      ),
      PFTauHighEfficiencyIDMuonRejectionEfficiencies = cms.PSet(
        numerator = cms.string('RecoTauV/shrinkingConePFTauProducer_shrinkingConePFTauDiscriminationAgainstElectron/shrinkingConePFTauDiscriminationAgainstElectron_vs_#PAR#TauVisible'),
        denominator = cms.string('RecoTauV/shrinkingConePFTauProducer_ReferenceCollection/nRef_Taus_vs_#PAR#TauVisible'),
        efficiency = cms.string('RecoTauV/shrinkingConePFTauProducer_shrinkingConePFTauDiscriminationAgainstElectron/AgainstElectronEff#PAR#'),
        parameter = cms.vstring('pt', 'eta', 'phi', 'energy')
      ),
      PFTauHighEfficiencyIDElectronRejectionEfficiencies = cms.PSet(
        numerator = cms.string('RecoTauV/shrinkingConePFTauProducer_shrinkingConePFTauDiscriminationAgainstMuon/shrinkingConePFTauDiscriminationAgainstMuon_vs_#PAR#TauVisible'),
        denominator = cms.string('RecoTauV/shrinkingConePFTauProducer_ReferenceCollection/nRef_Taus_vs_#PAR#TauVisible'),
        efficiency = cms.string('RecoTauV/shrinkingConePFTauProducer_shrinkingConePFTauDiscriminationAgainstMuon/AgainstMuonEff#PAR#'),
        parameter = cms.vstring('pt', 'eta', 'phi', 'energy')
      ),
# PFTAUHIGHEFFICIENCY_LEADING_PION EFFICIENCY CALCULATION
      PFTauHighEfficiencyLeadingPionIDMatchingEfficiencies = cms.PSet(
        numerator = cms.string('RecoTauV/shrinkingConePFTauProducerLeadingPion_Matched/shrinkingConePFTauProducerMatched_vs_#PAR#TauVisible'),
        denominator = cms.string('RecoTauV/shrinkingConePFTauProducerLeadingPion_ReferenceCollection/nRef_Taus_vs_#PAR#TauVisible'),
        efficiency = cms.string('RecoTauV/shrinkingConePFTauProducerLeadingPion_Matched/PFJetMatchingEff#PAR#'),
        parameter = cms.vstring('pt', 'eta', 'phi', 'energy')
      ),
      PFTauHighEfficiencyLeadingPionIDLeadingPionPtCutEfficiencies = cms.PSet(
        numerator = cms.string('RecoTauV/shrinkingConePFTauProducerLeadingPion_shrinkingConePFTauDiscriminationByLeadingPionPtCut/shrinkingConePFTauDiscriminationByLeadingPionPtCut_vs_#PAR#TauVisible'),
        denominator = cms.string('RecoTauV/shrinkingConePFTauProducerLeadingPion_ReferenceCollection/nRef_Taus_vs_#PAR#TauVisible'),
        efficiency = cms.string('RecoTauV/shrinkingConePFTauProducerLeadingPion_shrinkingConePFTauDiscriminationByLeadingPionPtCut/LeadingPionPtCutEff#PAR#'),
        parameter = cms.vstring('pt', 'eta', 'phi', 'energy')
      ),
      PFTauHighEfficiencyLeadingPionIDTrackIsolationEfficienies = cms.PSet(
        numerator = cms.string('RecoTauV/shrinkingConePFTauProducerLeadingPion_shrinkingConePFTauDiscriminationByTrackIsolationUsingLeadingPion/shrinkingConePFTauDiscriminationByTrackIsolationUsingLeadingPion_vs_#PAR#TauVisible'),
        denominator = cms.string('RecoTauV/shrinkingConePFTauProducerLeadingPion_ReferenceCollection/nRef_Taus_vs_#PAR#TauVisible'),
        efficiency = cms.string('RecoTauV/shrinkingConePFTauProducerLeadingPion_shrinkingConePFTauDiscriminationByTrackIsolationUsingLeadingPion/TrackIsolationEff#PAR#'),
        parameter = cms.vstring('pt', 'eta', 'phi', 'energy')
      ),
      PFTauHighEfficiencyLeadingPionIDECALIsolationEfficienies = cms.PSet(
        numerator = cms.string('RecoTauV/shrinkingConePFTauProducerLeadingPion_shrinkingConePFTauDiscriminationByECALIsolationUsingLeadingPion/shrinkingConePFTauDiscriminationByECALIsolationUsingLeadingPion_vs_#PAR#TauVisible'),
        denominator = cms.string('RecoTauV/shrinkingConePFTauProducerLeadingPion_ReferenceCollection/nRef_Taus_vs_#PAR#TauVisible'),
        efficiency = cms.string('RecoTauV/shrinkingConePFTauProducerLeadingPion_shrinkingConePFTauDiscriminationByECALIsolationUsingLeadingPion/ECALIsolationEff#PAR#'),
        parameter = cms.vstring('pt', 'eta', 'phi', 'energy')
      ),
      PFTauHighEfficiencyLeadingPionIDMuonRejectionEfficiencies = cms.PSet(
        numerator = cms.string('RecoTauV/shrinkingConePFTauProducerLeadingPion_shrinkingConePFTauDiscriminationAgainstElectron/shrinkingConePFTauDiscriminationAgainstElectron_vs_#PAR#TauVisible'),
        denominator = cms.string('RecoTauV/shrinkingConePFTauProducerLeadingPion_ReferenceCollection/nRef_Taus_vs_#PAR#TauVisible'),
        efficiency = cms.string('RecoTauV/shrinkingConePFTauProducerLeadingPion_shrinkingConePFTauDiscriminationAgainstElectron/AgainstElectronEff#PAR#'),
        parameter = cms.vstring('pt', 'eta', 'phi', 'energy')
      ),
      PFTauHighEfficiencyLeadingPionIDElectronRejectionEfficiencies = cms.PSet(
        numerator = cms.string('RecoTauV/shrinkingConePFTauProducerLeadingPion_shrinkingConePFTauDiscriminationAgainstMuon/shrinkingConePFTauDiscriminationAgainstMuon_vs_#PAR#TauVisible'),
        denominator = cms.string('RecoTauV/shrinkingConePFTauProducerLeadingPion_ReferenceCollection/nRef_Taus_vs_#PAR#TauVisible'),
        efficiency = cms.string('RecoTauV/shrinkingConePFTauProducerLeadingPion_shrinkingConePFTauDiscriminationAgainstMuon/AgainstMuonEff#PAR#'),
        parameter = cms.vstring('pt', 'eta', 'phi', 'energy')
      ),      
# CALOTAU EFFICIENCY CALCULATIONS      
      CaloTauIDMatchingEfficiencies = cms.PSet(
        numerator = cms.string('RecoTauV/caloRecoTauProducer_Matched/caloRecoTauProducerMatched_vs_#PAR#TauVisible'),
        denominator = cms.string('RecoTauV/caloRecoTauProducer_ReferenceCollection/nRef_Taus_vs_#PAR#TauVisible'),
        efficiency = cms.string('RecoTauV/caloRecoTauProducer_Matched/CaloJetMatchingEff#PAR#'),
        parameter = cms.vstring('pt', 'eta', 'phi', 'energy')
      ),
      CaloTauIDLeadingTrackFindEfficiencies = cms.PSet(
        numerator = cms.string('RecoTauV/caloRecoTauProducer_caloRecoTauDiscriminationByLeadingTrackFinding/caloRecoTauDiscriminationByLeadingTrackFinding_vs_#PAR#TauVisible'),
        denominator = cms.string('RecoTauV/caloRecoTauProducer_ReferenceCollection/nRef_Taus_vs_#PAR#TauVisible'),
        efficiency = cms.string('RecoTauV/caloRecoTauProducer_caloRecoTauDiscriminationByLeadingTrackFinding/LeadingTrackFindingEff#PAR#'),
        parameter = cms.vstring('pt', 'eta', 'phi', 'energy')
      ),
      CaloTauIDLeadingTrackPtCutEfficiencies = cms.PSet(
        numerator = cms.string('RecoTauV/caloRecoTauProducer_caloRecoTauDiscriminationByLeadingTrackPtCut/caloRecoTauDiscriminationByLeadingTrackPtCut_vs_#PAR#TauVisible'),
        denominator = cms.string('RecoTauV/caloRecoTauProducer_ReferenceCollection/nRef_Taus_vs_#PAR#TauVisible'),
        efficiency = cms.string('RecoTauV/caloRecoTauProducer_caloRecoTauDiscriminationByLeadingTrackPtCut/LeadingTrackPtCutEff#PAR#'),
        parameter = cms.vstring('pt', 'eta', 'phi', 'energy')
      ),
      CaloTauIDIsolationEfficienies = cms.PSet(
        numerator = cms.string('RecoTauV/caloRecoTauProducer_caloRecoTauDiscriminationByIsolation/caloRecoTauDiscriminationByIsolation_vs_#PAR#TauVisible'),
        denominator = cms.string('RecoTauV/caloRecoTauProducer_ReferenceCollection/nRef_Taus_vs_#PAR#TauVisible'),
        efficiency = cms.string('RecoTauV/caloRecoTauProducer_caloRecoTauDiscriminationByIsolation/IsolationEff#PAR#'),
        parameter = cms.vstring('pt', 'eta', 'phi', 'energy')
      ),
      CaloTauIDMuonRejectionEfficiencies = cms.PSet(
        numerator = cms.string('RecoTauV/caloRecoTauProducer_caloRecoTauDiscriminationAgainstElectron/caloRecoTauDiscriminationAgainstElectron_vs_#PAR#TauVisible'),
        denominator = cms.string('RecoTauV/caloRecoTauProducer_ReferenceCollection/nRef_Taus_vs_#PAR#TauVisible'),
        efficiency = cms.string('RecoTauV/caloRecoTauProducer_caloRecoTauDiscriminationAgainstElectron/AgainstElectronEff#PAR#'),
        parameter = cms.vstring('pt', 'eta', 'phi', 'energy')
      )      
    )                                
#    outputFileName = cms.string('CMSSW_2_2_0_HadronicTauOneAndThreeProng_ALL.root'),
)


 
