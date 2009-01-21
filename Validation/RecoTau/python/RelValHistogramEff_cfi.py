import FWCore.ParameterSet.Config as cms

TauEfficiencies = cms.EDAnalyzer("DQMHistEffProducer",
    plots = cms.PSet(
      PFTauIDMatchingEfficiencies = cms.PSet(
        numerator = cms.string('RecoTauV/pfRecoTauProducer_Matched/pfRecoTauProducerMatched_vs_#PAR#TauVisible'),
        denominator = cms.string('RecoTauV/pfRecoTauProducer_ReferenceCollection/nRef_Taus_vs_#PAR#TauVisible'),
        efficiency = cms.string('RecoTauV/pfRecoTauProducer_Matched/PFJetMatchingEff#PAR#'),
        parameter = cms.vstring('pt', 'eta', 'phi', 'energy')
      ),
      PFTauIDLeadingTrackFindEfficiencies = cms.PSet(
        numerator = cms.string('RecoTauV/pfRecoTauProducer_pfRecoTauDiscriminationByLeadingTrackFinding/pfRecoTauDiscriminationByLeadingTrackFinding_vs_#PAR#TauVisible'),
        denominator = cms.string('RecoTauV/pfRecoTauProducer_ReferenceCollection/nRef_Taus_vs_#PAR#TauVisible'),
        efficiency = cms.string('RecoTauV/pfRecoTauProducer_pfRecoTauDiscriminationByLeadingTrackFinding/LeadingTrackFindingEff#PAR#'),
        parameter = cms.vstring('pt', 'eta', 'phi', 'energy')
      ),
      PFTauIDLeadingTrackPtCutEfficiencies = cms.PSet(
        numerator = cms.string('RecoTauV/pfRecoTauProducer_pfRecoTauDiscriminationByLeadingTrackPtCut/pfRecoTauDiscriminationByLeadingTrackPtCut_vs_#PAR#TauVisible'),
        denominator = cms.string('RecoTauV/pfRecoTauProducer_ReferenceCollection/nRef_Taus_vs_#PAR#TauVisible'),
        efficiency = cms.string('RecoTauV/pfRecoTauProducer_pfRecoTauDiscriminationByLeadingTrackPtCut/LeadingTrackPtCutEff#PAR#'),
        parameter = cms.vstring('pt', 'eta', 'phi', 'energy')
      ),
      PFTauIDTrackIsolationEfficienies = cms.PSet(
        numerator = cms.string('RecoTauV/pfRecoTauProducer_pfRecoTauDiscriminationByTrackIsolation/pfRecoTauDiscriminationByTrackIsolation_vs_#PAR#TauVisible'),
        denominator = cms.string('RecoTauV/pfRecoTauProducer_ReferenceCollection/nRef_Taus_vs_#PAR#TauVisible'),
        efficiency = cms.string('RecoTauV/pfRecoTauProducer_pfRecoTauDiscriminationByTrackIsolation/TrackIsolationEff#PAR#'),
        parameter = cms.vstring('pt', 'eta', 'phi', 'energy')
      ),
      PFTauIDECALIsolationEfficienies = cms.PSet(
        numerator = cms.string('RecoTauV/pfRecoTauProducer_pfRecoTauDiscriminationByECALIsolation/pfRecoTauDiscriminationByECALIsolation_vs_#PAR#TauVisible'),
        denominator = cms.string('RecoTauV/pfRecoTauProducer_ReferenceCollection/nRef_Taus_vs_#PAR#TauVisible'),
        efficiency = cms.string('RecoTauV/pfRecoTauProducer_pfRecoTauDiscriminationByECALIsolation/ECALIsolationEff#PAR#'),
        parameter = cms.vstring('pt', 'eta', 'phi', 'energy')
      ),
      PFTauIDMuonRejectionEfficiencies = cms.PSet(
        numerator = cms.string('RecoTauV/pfRecoTauProducer_pfRecoTauDiscriminationAgainstElectron/pfRecoTauDiscriminationAgainstElectron_vs_#PAR#TauVisible'),
        denominator = cms.string('RecoTauV/pfRecoTauProducer_ReferenceCollection/nRef_Taus_vs_#PAR#TauVisible'),
        efficiency = cms.string('RecoTauV/pfRecoTauProducer_pfRecoTauDiscriminationAgainstElectron/AgainstElectronEff#PAR#'),
        parameter = cms.vstring('pt', 'eta', 'phi', 'energy')
      ),
      PFTauIDElectronRejectionEfficiencies = cms.PSet(
        numerator = cms.string('RecoTauV/pfRecoTauProducer_pfRecoTauDiscriminationAgainstMuon/pfRecoTauDiscriminationAgainstMuon_vs_#PAR#TauVisible'),
        denominator = cms.string('RecoTauV/pfRecoTauProducer_ReferenceCollection/nRef_Taus_vs_#PAR#TauVisible'),
        efficiency = cms.string('RecoTauV/pfRecoTauProducer_pfRecoTauDiscriminationAgainstMuon/AgainstMuonEff#PAR#'),
        parameter = cms.vstring('pt', 'eta', 'phi', 'energy')
      ),
      PFTauHighEfficiencyIDMatchingEfficiencies = cms.PSet(
        numerator = cms.string('RecoTauV/pfRecoTauProducerHighEfficiency_Matched/pfRecoTauProducerHighEfficiencyMatched_vs_#PAR#TauVisible'),
        denominator = cms.string('RecoTauV/pfRecoTauProducerHighEfficiency_ReferenceCollection/nRef_Taus_vs_#PAR#TauVisible'),
        efficiency = cms.string('RecoTauV/pfRecoTauProducerHighEfficiency_Matched/PFJetMatchingEff#PAR#'),
        parameter = cms.vstring('pt', 'eta', 'phi', 'energy')
      ),
      PFTauHighEfficiencyIDLeadingTrackFindEfficiencies = cms.PSet(
        numerator = cms.string('RecoTauV/pfRecoTauProducerHighEfficiency_pfRecoTauDiscriminationByLeadingTrackFindingHighEfficiency/pfRecoTauDiscriminationByLeadingTrackFindingHighEfficiency_vs_#PAR#TauVisible'),
        denominator = cms.string('RecoTauV/pfRecoTauProducerHighEfficiency_ReferenceCollection/nRef_Taus_vs_#PAR#TauVisible'),
        efficiency = cms.string('RecoTauV/pfRecoTauProducerHighEfficiency_pfRecoTauDiscriminationByLeadingTrackFindingHighEfficiency/LeadingTrackFindingEff#PAR#'),
        parameter = cms.vstring('pt', 'eta', 'phi', 'energy')
      ),
      PFTauHighEfficiencyIDLeadingTrackPtCutEfficiencies = cms.PSet(
        numerator = cms.string('RecoTauV/pfRecoTauProducerHighEfficiency_pfRecoTauDiscriminationByLeadingTrackPtCutHighEfficiency/pfRecoTauDiscriminationByLeadingTrackPtCutHighEfficiency_vs_#PAR#TauVisible'),
        denominator = cms.string('RecoTauV/pfRecoTauProducerHighEfficiency_ReferenceCollection/nRef_Taus_vs_#PAR#TauVisible'),
        efficiency = cms.string('RecoTauV/pfRecoTauProducerHighEfficiency_pfRecoTauDiscriminationByLeadingTrackPtCutHighEfficiency/LeadingTrackPtCutEff#PAR#'),
        parameter = cms.vstring('pt', 'eta', 'phi', 'energy')
      ),
      PFTauHighEfficiencyIDTrackIsolationEfficienies = cms.PSet(
        numerator = cms.string('RecoTauV/pfRecoTauProducerHighEfficiency_pfRecoTauDiscriminationByTrackIsolationHighEfficiency/pfRecoTauDiscriminationByTrackIsolationHighEfficiency_vs_#PAR#TauVisible'),
        denominator = cms.string('RecoTauV/pfRecoTauProducerHighEfficiency_ReferenceCollection/nRef_Taus_vs_#PAR#TauVisible'),
        efficiency = cms.string('RecoTauV/pfRecoTauProducerHighEfficiency_pfRecoTauDiscriminationByTrackIsolationHighEfficiency/TrackIsolationEff#PAR#'),
        parameter = cms.vstring('pt', 'eta', 'phi', 'energy')
      ),
      PFTauHighEfficiencyIDECALIsolationEfficienies = cms.PSet(
        numerator = cms.string('RecoTauV/pfRecoTauProducerHighEfficiency_pfRecoTauDiscriminationByECALIsolationHighEfficiency/pfRecoTauDiscriminationByECALIsolationHighEfficiency_vs_#PAR#TauVisible'),
        denominator = cms.string('RecoTauV/pfRecoTauProducerHighEfficiency_ReferenceCollection/nRef_Taus_vs_#PAR#TauVisible'),
        efficiency = cms.string('RecoTauV/pfRecoTauProducerHighEfficiency_pfRecoTauDiscriminationByECALIsolationHighEfficiency/ECALIsolationEff#PAR#'),
        parameter = cms.vstring('pt', 'eta', 'phi', 'energy')
      ),
      PFTauHighEfficiencyIDMuonRejectionEfficiencies = cms.PSet(
        numerator = cms.string('RecoTauV/pfRecoTauProducerHighEfficiency_pfRecoTauDiscriminationAgainstElectronHighEfficiency/pfRecoTauDiscriminationAgainstElectronHighEfficiency_vs_#PAR#TauVisible'),
        denominator = cms.string('RecoTauV/pfRecoTauProducerHighEfficiency_ReferenceCollection/nRef_Taus_vs_#PAR#TauVisible'),
        efficiency = cms.string('RecoTauV/pfRecoTauProducerHighEfficiency_pfRecoTauDiscriminationAgainstElectronHighEfficiency/AgainstElectronEff#PAR#'),
        parameter = cms.vstring('pt', 'eta', 'phi', 'energy')
      ),
      PFTauHighEfficiencyIDElectronRejectionEfficiencies = cms.PSet(
        numerator = cms.string('RecoTauV/pfRecoTauProducerHighEfficiency_pfRecoTauDiscriminationAgainstMuonHighEfficiency/pfRecoTauDiscriminationAgainstMuonHighEfficiency_vs_#PAR#TauVisible'),
        denominator = cms.string('RecoTauV/pfRecoTauProducerHighEfficiency_ReferenceCollection/nRef_Taus_vs_#PAR#TauVisible'),
        efficiency = cms.string('RecoTauV/pfRecoTauProducerHighEfficiency_pfRecoTauDiscriminationAgainstMuonHighEfficiency/AgainstMuonEff#PAR#'),
        parameter = cms.vstring('pt', 'eta', 'phi', 'energy')
      ),
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
                            
saveTauEff = cms.EDAnalyzer("DQMSimpleFileSaver",
  outputFileName = cms.string('CMSSW_2_2_3_tauGenJets.root')
)

saveTauEffFast = cms.EDAnalyzer("DQMSimpleFileSaver",
  outputFileName = cms.string('FastSim_CMSSW_2_2_3_tauGenJets.root')
)




 
