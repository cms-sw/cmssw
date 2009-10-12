
import FWCore.ParameterSet.Config as cms

from Validation.RecoTau.plotterTauTagValidation_cfi import *


##################################################
#
#   The plotting of all the PFTau ID efficiencies
#
##################################################
plotPFTauEfficiencies = cms.EDAnalyzer("DQMHistPlotter",
  standardDrawingStuff,
  standardCompareTestAndReference,
  drawJobs = cms.PSet(                                     
    PFJetMatchingEff = cms.PSet(
      plots = cms.PSet(
        dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/fixedConePFTauProducer_Matched/PFJetMatchingEff#PAR#'),
        processes = cms.vstring('test', 'reference')
      ),
      parameter = cms.vstring('pt', 'eta', 'phi', 'energy'),
      xAxis = cms.string('#PAR#'),
      yAxis = cms.string('efficiency'),
      legend = cms.string('efficiency'),
      labels = cms.vstring('pt', 'eta'),
      drawOptionSet = cms.string('efficiency')
    ),
    LeadingTrackPtCutEff = cms.PSet(
      plots = cms.PSet(
        dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/fixedConePFTauProducer_fixedConePFTauDiscriminationByLeadingTrackPtCut/LeadingTrackPtCutEff#PAR#'),
        processes = cms.vstring('test', 'reference')
      ),  
      parameter = cms.vstring('pt', 'eta', 'phi', 'energy'),
      xAxis = cms.string('#PAR#'),
      yAxis = cms.string('efficiency'),
      legend = cms.string('efficiency'),
      labels = cms.vstring('pt', 'eta'),
      drawOptionSet = cms.string('efficiency')
    ),
    TrackIsolationEff = cms.PSet(
      plots = cms.PSet(
        dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/fixedConePFTauProducer_fixedConePFTauDiscriminationByTrackIsolation/TrackIsolationEff#PAR#'),
        processes = cms.vstring('test', 'reference')
      ),
      parameter = cms.vstring('pt', 'eta', 'phi', 'energy'),
      xAxis = cms.string('#PAR#'),
      yAxis = cms.string('efficiency'),
      legend = cms.string('efficiency'),
      labels = cms.vstring('pt', 'eta'),
      drawOptionSet = cms.string('efficiency')
    ),
    ECALIsolationEff = cms.PSet(
      plots = cms.PSet(
        dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/fixedConePFTauProducer_fixedConePFTauDiscriminationByECALIsolation/ECALIsolationEff#PAR#'),
        processes = cms.vstring('test', 'reference')
      ),
      parameter = cms.vstring('pt', 'eta', 'phi', 'energy'),
      xAxis = cms.string('#PAR#'),
      yAxis = cms.string('efficiency'),
      legend = cms.string('efficiency'),
      labels = cms.vstring('pt', 'eta'),
      drawOptionSet = cms.string('efficiency')
    ),
    AgainstElectronEff = cms.PSet(
      plots = cms.PSet(
        dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/fixedConePFTauProducer_fixedConePFTauDiscriminationAgainstElectron/AgainstElectronEff#PAR#'), 
        processes = cms.vstring('test', 'reference')
      ),
      parameter = cms.vstring('pt', 'eta', 'phi', 'energy'),
      xAxis = cms.string('#PAR#'),
      yAxis = cms.string('efficiency'),
      legend = cms.string('efficiency'),
      labels = cms.vstring('pt', 'eta'),
      drawOptionSet = cms.string('efficiency')
    ),
    AgainstMuonEff = cms.PSet(
      plots = cms.PSet(
        dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/fixedConePFTauProducer_fixedConePFTauDiscriminationAgainstMuon/AgainstMuonEff#PAR#'), 
        processes = cms.vstring('test', 'reference')
      ),
      parameter = cms.vstring('pt', 'eta', 'phi', 'energy'),
      xAxis = cms.string('#PAR#'),
      yAxis = cms.string('efficiency'),
      legend = cms.string('efficiency'),
      labels = cms.vstring('pt', 'eta'),
      drawOptionSet = cms.string('efficiency')
    ),
    TauIdEffStepByStep = cms.PSet(
      plots = cms.VPSet(
        cms.PSet(
          dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/fixedConePFTauProducer_Matched/PFJetMatchingEff#PAR#'),
          process = cms.string('test'),
          drawOptionEntry = cms.string('eff_overlay01'),
          legendEntry = cms.string('PFJet Matching')
        ),    
        cms.PSet(
          dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/fixedConePFTauProducer_fixedConePFTauDiscriminationByLeadingTrackPtCut/LeadingTrackPtCutEff#PAR#'),
          process = cms.string('test'),
          drawOptionEntry = cms.string('eff_overlay02'),
          legendEntry = cms.string('Lead Track Finding')
        ),
        cms.PSet(
          dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/fixedConePFTauProducer_fixedConePFTauDiscriminationByTrackIsolation/TrackIsolationEff#PAR#'),
          process = cms.string('test'),
          drawOptionEntry = cms.string('eff_overlay03'),
          legendEntry = cms.string('Track Iso.')
        ),
        cms.PSet(
          dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/fixedConePFTauProducer_fixedConePFTauDiscriminationByECALIsolation/ECALIsolationEff#PAR#'),
          process = cms.string('test'),
          drawOptionEntry = cms.string('eff_overlay04'),
          legendEntry = cms.string('Track + Gamma Iso.')
        ),
        cms.PSet(
          dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/fixedConePFTauProducer_fixedConePFTauDiscriminationAgainstElectron/AgainstElectronEff#PAR#'),
          process = cms.string('test'),
          drawOptionEntry = cms.string('eff_overlay05'),
          legendEntry = cms.string('Electron Rejection')
        ),
        cms.PSet(
          dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/fixedConePFTauProducer_fixedConePFTauDiscriminationAgainstMuon/AgainstMuonEff#PAR#'),
          process = cms.string('test'),
          drawOptionEntry = cms.string('eff_overlay06'),
          legendEntry = cms.string('Muon Rejection')
        )
      ),
      parameter = cms.vstring('pt', 'eta', 'phi', 'energy'),
      title = cms.string('TauId step by step efficiencies'),
      xAxis = cms.string('#PAR#'),
      yAxis = cms.string('efficiency'),
      legend = cms.string('efficiency_overlay'),
      labels = cms.vstring('pt', 'eta')
    )
  ),
  outputFilePath = cms.string('./fixedConePFTauProducer/'),
)                    

##################################################
#
#   The plotting of all the PFTauHighEfficiencies ID efficiencies
#
##################################################

plotPFTauHighEfficiencyEfficiencies = cms.EDAnalyzer("DQMHistPlotter",
  standardDrawingStuff,
  standardCompareTestAndReference,
  drawJobs = cms.PSet(                                     
    PFJetMatchingEff = cms.PSet(
      plots = cms.PSet(
        dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/shrinkingConePFTauProducer_Matched/PFJetMatchingEff#PAR#'),
        processes = cms.vstring('test', 'reference')
      ),
      parameter = cms.vstring('pt', 'eta', 'phi', 'energy'),
      xAxis = cms.string('#PAR#'),
      yAxis = cms.string('efficiency'),
      legend = cms.string('efficiency'),
      labels = cms.vstring('pt', 'eta'),
      drawOptionSet = cms.string('efficiency')
    ),
    LeadingTrackPtCutEff = cms.PSet(
      plots = cms.PSet(
        dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/shrinkingConePFTauProducer_shrinkingConePFTauDiscriminationByLeadingTrackPtCut/LeadingTrackPtCutEff#PAR#'),
        processes = cms.vstring('test', 'reference')
      ),  
      parameter = cms.vstring('pt', 'eta', 'phi', 'energy'),
      xAxis = cms.string('#PAR#'),
      yAxis = cms.string('efficiency'),
      legend = cms.string('efficiency'),
      labels = cms.vstring('pt', 'eta'),
      drawOptionSet = cms.string('efficiency')
    ),
    TrackIsolationEff = cms.PSet(
      plots = cms.PSet(
        dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/shrinkingConePFTauProducer_shrinkingConePFTauDiscriminationByTrackIsolation/TrackIsolationEff#PAR#'),
        processes = cms.vstring('test', 'reference')
      ),
      parameter = cms.vstring('pt', 'eta', 'phi', 'energy'),
      xAxis = cms.string('#PAR#'),
      yAxis = cms.string('efficiency'),
      legend = cms.string('efficiency'),
      labels = cms.vstring('pt', 'eta'),
      drawOptionSet = cms.string('efficiency')
    ),
    ECALIsolationEff = cms.PSet(
      plots = cms.PSet(
        dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/shrinkingConePFTauProducer_shrinkingConePFTauDiscriminationByECALIsolation/ECALIsolationEff#PAR#'),
        processes = cms.vstring('test', 'reference')
      ),
      parameter = cms.vstring('pt', 'eta', 'phi', 'energy'),
      xAxis = cms.string('#PAR#'),
      yAxis = cms.string('efficiency'),
      legend = cms.string('efficiency'),
      labels = cms.vstring('pt', 'eta'),
      drawOptionSet = cms.string('efficiency')
    ),    
    AgainstElectronEff = cms.PSet(
      plots = cms.PSet(
        dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/shrinkingConePFTauProducer_shrinkingConePFTauDiscriminationAgainstElectron/AgainstElectronEff#PAR#'), 
        processes = cms.vstring('test', 'reference')
      ),
      parameter = cms.vstring('pt', 'eta', 'phi', 'energy'),
      xAxis = cms.string('#PAR#'),
      yAxis = cms.string('efficiency'),
      legend = cms.string('efficiency'),
      labels = cms.vstring('pt', 'eta'),
      drawOptionSet = cms.string('efficiency')
    ),
    AgainstMuonEff = cms.PSet(
      plots = cms.PSet(
        dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/shrinkingConePFTauProducer_shrinkingConePFTauDiscriminationAgainstMuon/AgainstMuonEff#PAR#'), 
        processes = cms.vstring('test', 'reference')
      ),
      parameter = cms.vstring('pt', 'eta', 'phi', 'energy'),
      xAxis = cms.string('#PAR#'),
      yAxis = cms.string('efficiency'),
      legend = cms.string('efficiency'),
      labels = cms.vstring('pt', 'eta'),
      drawOptionSet = cms.string('efficiency')
    ),
    TauIdEffStepByStep = cms.PSet(
      plots = cms.VPSet(
        cms.PSet(
          dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/shrinkingConePFTauProducer_Matched/PFJetMatchingEff#PAR#'),
          process = cms.string('test'),
          drawOptionEntry = cms.string('eff_overlay01'),
          legendEntry = cms.string('PFJet Matching')
        ),    
        cms.PSet(
          dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/shrinkingConePFTauProducer_shrinkingConePFTauDiscriminationByLeadingTrackPtCut/LeadingTrackPtCutEff#PAR#'),
          process = cms.string('test'),
          drawOptionEntry = cms.string('eff_overlay02'),
          legendEntry = cms.string('Lead Track Finding')
        ),
        cms.PSet(
          dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/shrinkingConePFTauProducer_shrinkingConePFTauDiscriminationByTrackIsolation/TrackIsolationEff#PAR#'),
          process = cms.string('test'),
          drawOptionEntry = cms.string('eff_overlay03'),
          legendEntry = cms.string('Track Iso.')
        ),
        cms.PSet(
          dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/shrinkingConePFTauProducer_shrinkingConePFTauDiscriminationByECALIsolation/ECALIsolationEff#PAR#'),
          process = cms.string('test'),
          drawOptionEntry = cms.string('eff_overlay04'),
          legendEntry = cms.string('Track + Gamma Iso.')
        ),
        cms.PSet(
          dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/shrinkingConePFTauProducer_shrinkingConePFTauDiscriminationAgainstElectron/AgainstElectronEff#PAR#'),
          process = cms.string('test'),
          drawOptionEntry = cms.string('eff_overlay05'),
          legendEntry = cms.string('Electron Rejection')
        ),
        cms.PSet(
          dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/shrinkingConePFTauProducer_shrinkingConePFTauDiscriminationAgainstMuon/AgainstMuonEff#PAR#'),
          process = cms.string('test'),
          drawOptionEntry = cms.string('eff_overlay06'),
          legendEntry = cms.string('Muon Rejection')
        )
      ),
      parameter = cms.vstring('pt', 'eta', 'phi', 'energy'),
      title = cms.string('TauId step by step efficiencies'),
      xAxis = cms.string('#PAR#'),
      yAxis = cms.string('efficiency'),
      legend = cms.string('efficiency_overlay'),
      labels = cms.vstring('pt', 'eta')
    )
  ),
  outputFilePath = cms.string('./shrinkingConePFTauProducer/'),
)      

##################################################
#
#   The plotting of all the CaloTau ID efficiencies
#
##################################################

plotCaloTauEfficiencies = cms.EDAnalyzer("DQMHistPlotter",
  standardDrawingStuff,
  standardCompareTestAndReference,
  drawJobs = cms.PSet(                                     
    CaloJetMatchingEff = cms.PSet(
      plots = cms.PSet(
        dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/caloRecoTauProducer_Matched/CaloJetMatchingEff#PAR#'),
        processes = cms.vstring('test', 'reference')
      ),
      parameter = cms.vstring('pt', 'eta', 'phi', 'energy'),
      xAxis = cms.string('#PAR#'),
      yAxis = cms.string('efficiency'),
      legend = cms.string('efficiency'),
      labels = cms.vstring('pt', 'eta'),
      drawOptionSet = cms.string('efficiency')
    ),
    LeadingTrackPtCutEff = cms.PSet(
      plots = cms.PSet(
        dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/caloRecoTauProducer_caloRecoTauDiscriminationByLeadingTrackPtCut/LeadingTrackPtCutEff#PAR#'),
        processes = cms.vstring('test', 'reference')
      ),  
      parameter = cms.vstring('pt', 'eta', 'phi', 'energy'),
      xAxis = cms.string('#PAR#'),
      yAxis = cms.string('efficiency'),
      legend = cms.string('efficiency'),
      labels = cms.vstring('pt', 'eta'),
      drawOptionSet = cms.string('efficiency')
    ),
    IsolationEff = cms.PSet(
      plots = cms.PSet(
        dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/caloRecoTauProducer_caloRecoTauDiscriminationByIsolation/IsolationEff#PAR#'),
        processes = cms.vstring('test', 'reference')
      ),
      parameter = cms.vstring('pt', 'eta', 'phi', 'energy'),
      xAxis = cms.string('#PAR#'),
      yAxis = cms.string('efficiency'),
      legend = cms.string('efficiency'),
      labels = cms.vstring('pt', 'eta'),
      drawOptionSet = cms.string('efficiency')
    ),
    AgainstElectronEff = cms.PSet(
      plots = cms.PSet(
        dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/caloRecoTauProducer_caloRecoTauDiscriminationAgainstElectron/AgainstElectronEff#PAR#'), 
        processes = cms.vstring('test', 'reference')
      ),
      parameter = cms.vstring('pt', 'eta', 'phi', 'energy'),
      xAxis = cms.string('#PAR#'),
      yAxis = cms.string('efficiency'),
      legend = cms.string('efficiency'),
      labels = cms.vstring('pt', 'eta'),
      drawOptionSet = cms.string('efficiency')
    ),
    TauIdEffStepByStep = cms.PSet(
      plots = cms.VPSet(
        cms.PSet(
          dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/caloRecoTauProducer_Matched/CaloJetMatchingEff#PAR#'),
          process = cms.string('test'),
          drawOptionEntry = cms.string('eff_overlay01'),
          legendEntry = cms.string('CaloJet Matching')
        ),    
        cms.PSet(
          dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/caloRecoTauProducer_caloRecoTauDiscriminationByLeadingTrackPtCut/LeadingTrackPtCutEff#PAR#'),
          process = cms.string('test'),
          drawOptionEntry = cms.string('eff_overlay02'),
          legendEntry = cms.string('Lead Track Finding')
        ),
        cms.PSet(
          dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/caloRecoTauProducer_caloRecoTauDiscriminationByIsolation/IsolationEff#PAR#'),
          process = cms.string('test'),
          drawOptionEntry = cms.string('eff_overlay03'),
          legendEntry = cms.string('Track Isolation')
        ),
        cms.PSet(
          dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/caloRecoTauProducer_caloRecoTauDiscriminationAgainstElectron/AgainstElectronEff#PAR#'),
          process = cms.string('test'),
          drawOptionEntry = cms.string('eff_overlay04'),
          legendEntry = cms.string('Electron Rejection')
        )
      ),
      parameter = cms.vstring('pt', 'eta', 'phi', 'energy'),
      title = cms.string('TauId step by step efficiencies'),
      xAxis = cms.string('#PAR#'),
      yAxis = cms.string('efficiency'),
      legend = cms.string('efficiency_overlay'),
      labels = cms.vstring('pt', 'eta')
    )
  ),
  outputFilePath = cms.string('./caloRecoTauProducer/'),
)

##################################################
#
#   Plot Tanc performance
#
##################################################

plotTancValidation = cms.EDAnalyzer("DQMHistPlotter",
  standardDrawingStuff,
  standardCompareTestAndReference,
  drawJobs = cms.PSet(                                     
    PFJetMatchingEff = cms.PSet(
      plots = cms.PSet(
        dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/shrinkingConePFTauProducerTanc_Matched/PFJetMatchingEff#PAR#'),
        processes = cms.vstring('test', 'reference')
      ),
      parameter = cms.vstring('pt', 'eta', 'phi', 'energy'),
      xAxis = cms.string('#PAR#'),
      yAxis = cms.string('efficiency'),
      legend = cms.string('efficiency'),
      labels = cms.vstring('pt', 'eta'),
      drawOptionSet = cms.string('efficiency')
    ),
    LeadingTrackPtCutEff = cms.PSet(
      plots = cms.PSet(
        dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/shrinkingConePFTauProducerTanc_shrinkingConePFTauDiscriminationByLeadingPionPtCut/LeadingPionPtCutEff#PAR#'),
        processes = cms.vstring('test', 'reference')
      ),  
      parameter = cms.vstring('pt', 'eta', 'phi', 'energy'),
      xAxis = cms.string('#PAR#'),
      yAxis = cms.string('efficiency'),
      legend = cms.string('efficiency'),
      labels = cms.vstring('pt', 'eta'),
      drawOptionSet = cms.string('efficiency')
    ),
    TaNCfrOnePercentEff = cms.PSet(
      plots = cms.PSet(
        dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/shrinkingConePFTauProducerTanc_shrinkingConePFTauDiscriminationByTaNCfrOnePercent/TaNCfrOnePercentEff#PAR#'),
        processes = cms.vstring('test', 'reference')
      ),
      parameter = cms.vstring('pt', 'eta', 'phi', 'energy'),
      xAxis = cms.string('#PAR#'),
      yAxis = cms.string('efficiency'),
      legend = cms.string('efficiency'),
      labels = cms.vstring('pt', 'eta'),
      drawOptionSet = cms.string('efficiency')
    ),
    TaNCfrHalfPercentEff = cms.PSet(
      plots = cms.PSet(
        dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/shrinkingConePFTauProducerTanc_shrinkingConePFTauDiscriminationByTaNCfrHalfPercent/TaNCfrHalfPercentEff#PAR#'),
        processes = cms.vstring('test', 'reference')
      ),
      parameter = cms.vstring('pt', 'eta', 'phi', 'energy'),
      xAxis = cms.string('#PAR#'),
      yAxis = cms.string('efficiency'),
      legend = cms.string('efficiency'),
      labels = cms.vstring('pt', 'eta'),
      drawOptionSet = cms.string('efficiency')
    ),
    TaNCfrQuarterPercentEff = cms.PSet(
      plots = cms.PSet(
        dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/shrinkingConePFTauProducerTanc_shrinkingConePFTauDiscriminationByTaNCfrQuarterPercent/TaNCfrQuarterPercentEff#PAR#'),
        processes = cms.vstring('test', 'reference')
      ),
      parameter = cms.vstring('pt', 'eta', 'phi', 'energy'),
      xAxis = cms.string('#PAR#'),
      yAxis = cms.string('efficiency'),
      legend = cms.string('efficiency'),
      labels = cms.vstring('pt', 'eta'),
      drawOptionSet = cms.string('efficiency')
    ),
    TaNCfrTenthPercentEff = cms.PSet(
      plots = cms.PSet(
        dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/shrinkingConePFTauProducerTanc_shrinkingConePFTauDiscriminationByTaNCfrTenthPercent/TaNCfrTenthPercentEff#PAR#'),
        processes = cms.vstring('test', 'reference')
      ),
      parameter = cms.vstring('pt', 'eta', 'phi', 'energy'),
      xAxis = cms.string('#PAR#'),
      yAxis = cms.string('efficiency'),
      legend = cms.string('efficiency'),
      labels = cms.vstring('pt', 'eta'),
      drawOptionSet = cms.string('efficiency')
    ),
    AgainstElectronEff = cms.PSet(
      plots = cms.PSet(
        dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/shrinkingConePFTauProducerTanc_shrinkingConePFTauDiscriminationAgainstElectron/AgainstElectronEff#PAR#'), 
        processes = cms.vstring('test', 'reference')
      ),
      parameter = cms.vstring('pt', 'eta', 'phi', 'energy'),
      xAxis = cms.string('#PAR#'),
      yAxis = cms.string('efficiency'),
      legend = cms.string('efficiency'),
      labels = cms.vstring('pt', 'eta'),
      drawOptionSet = cms.string('efficiency')
    ),
    AgainstMuonEff = cms.PSet(
      plots = cms.PSet(
        dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/shrinkingConePFTauProducerTanc_shrinkingConePFTauDiscriminationAgainstMuon/AgainstMuonEff#PAR#'), 
        processes = cms.vstring('test', 'reference')
      ),
      parameter = cms.vstring('pt', 'eta', 'phi', 'energy'),
      xAxis = cms.string('#PAR#'),
      yAxis = cms.string('efficiency'),
      legend = cms.string('efficiency'),
      labels = cms.vstring('pt', 'eta'),
      drawOptionSet = cms.string('efficiency')
    ),
    TauIdEffStepByStep = cms.PSet(
      plots = cms.VPSet(
        cms.PSet(
          dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/shrinkingConePFTauProducerTanc_Matched/PFJetMatchingEff#PAR#'),
          process = cms.string('test'),
          drawOptionEntry = cms.string('eff_overlay01'),
          legendEntry = cms.string('PFJet Matching')
        ),    
        cms.PSet(
          dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/shrinkingConePFTauProducerTanc_shrinkingConePFTauDiscriminationByLeadingPionPtCut/LeadingPionPtCutEff#PAR#'),
          process = cms.string('test'),
          drawOptionEntry = cms.string('eff_overlay02'),
          legendEntry = cms.string('Lead Pion Finding')
        ),
        cms.PSet(
          dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/shrinkingConePFTauProducerTanc_shrinkingConePFTauDiscriminationByTaNCfrOnePercent/TaNCfrOnePercentEff#PAR#'),
          process = cms.string('test'),
          drawOptionEntry = cms.string('eff_overlay03'),
          legendEntry = cms.string('TaNC One Percent')
        ),
        cms.PSet(
          dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/shrinkingConePFTauProducerTanc_shrinkingConePFTauDiscriminationByTaNCfrHalfPercent/TaNCfrHalfPercentEff#PAR#'),
          process = cms.string('test'),
          drawOptionEntry = cms.string('eff_overlay03'),
          legendEntry = cms.string('TaNC Half Percent')
        ),
        cms.PSet(
          dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/shrinkingConePFTauProducerTanc_shrinkingConePFTauDiscriminationByTaNCfrQuarterPercent/TaNCfrQuarterPercentEff#PAR#'),
          process = cms.string('test'),
          drawOptionEntry = cms.string('eff_overlay03'),
          legendEntry = cms.string('TaNC Quarter Percent')
        ),
        cms.PSet(
          dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/shrinkingConePFTauProducerTanc_shrinkingConePFTauDiscriminationByTaNCfrTenthPercent/TaNCfrTenthPercentEff#PAR#'),
          process = cms.string('test'),
          drawOptionEntry = cms.string('eff_overlay03'),
          legendEntry = cms.string('TaNC Tenth Percent')
        ),
        cms.PSet(
          dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/shrinkingConePFTauProducerTanc_shrinkingConePFTauDiscriminationAgainstElectron/AgainstElectronEff#PAR#'),
          process = cms.string('test'),
          drawOptionEntry = cms.string('eff_overlay05'),
          legendEntry = cms.string('Electron Rejection')
        ),
        cms.PSet(
          dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/shrinkingConePFTauProducerTanc_shrinkingConePFTauDiscriminationAgainstMuon/AgainstMuonEff#PAR#'),
          process = cms.string('test'),
          drawOptionEntry = cms.string('eff_overlay06'),
          legendEntry = cms.string('Muon Rejection')
        )
      ),
      parameter = cms.vstring('pt', 'eta', 'phi', 'energy'),
      title = cms.string('TauId step by step efficiencies'),
      xAxis = cms.string('#PAR#'),
      yAxis = cms.string('efficiency'),
      legend = cms.string('efficiency_overlay'),
      labels = cms.vstring('pt', 'eta')
    )
  ),
  outputFilePath = cms.string('./shrinkingConePFTauProducerTanc/'),
)


##################################################
#
#   The plotting of all the PFTauHighEfficiencyUsingLeadingPion ID efficiencies
#
##################################################

plotPFTauHighEfficiencyEfficienciesLeadingPion = cms.EDAnalyzer("DQMHistPlotter",
  standardDrawingStuff,
  standardCompareTestAndReference,
  drawJobs = cms.PSet(                                     
    PFJetMatchingEff = cms.PSet(
      plots = cms.PSet(
        dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/shrinkingConePFTauProducerLeadingPion_Matched/PFJetMatchingEff#PAR#'),
        processes = cms.vstring('test', 'reference')
      ),
      parameter = cms.vstring('pt', 'eta', 'phi', 'energy'),
      xAxis = cms.string('#PAR#'),
      yAxis = cms.string('efficiency'),
      legend = cms.string('efficiency'),
      labels = cms.vstring('pt', 'eta'),
      drawOptionSet = cms.string('efficiency')
    ),
    LeadingTrackPtCutEff = cms.PSet(
      plots = cms.PSet(
        dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/shrinkingConePFTauProducerLeadingPion_shrinkingConePFTauDiscriminationByLeadingPionPtCut/LeadingPionPtCutEff#PAR#'),
        processes = cms.vstring('test', 'reference')
      ),  
      parameter = cms.vstring('pt', 'eta', 'phi', 'energy'),
      xAxis = cms.string('#PAR#'),
      yAxis = cms.string('efficiency'),
      legend = cms.string('efficiency'),
      labels = cms.vstring('pt', 'eta'),
      drawOptionSet = cms.string('efficiency')
    ),
    TrackIsolationEff = cms.PSet(
      plots = cms.PSet(
        dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/shrinkingConePFTauProducerLeadingPion_shrinkingConePFTauDiscriminationByTrackIsolationUsingLeadingPion/TrackIsolationEff#PAR#'),
        processes = cms.vstring('test', 'reference')
      ),
      parameter = cms.vstring('pt', 'eta', 'phi', 'energy'),
      xAxis = cms.string('#PAR#'),
      yAxis = cms.string('efficiency'),
      legend = cms.string('efficiency'),
      labels = cms.vstring('pt', 'eta'),
      drawOptionSet = cms.string('efficiency')
    ),
    ECALIsolationEff = cms.PSet(
      plots = cms.PSet(
        dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/shrinkingConePFTauProducerLeadingPion_shrinkingConePFTauDiscriminationByECALIsolationUsingLeadingPion/ECALIsolationEff#PAR#'),
        processes = cms.vstring('test', 'reference')
      ),
      parameter = cms.vstring('pt', 'eta', 'phi', 'energy'),
      xAxis = cms.string('#PAR#'),
      yAxis = cms.string('efficiency'),
      legend = cms.string('efficiency'),
      labels = cms.vstring('pt', 'eta'),
      drawOptionSet = cms.string('efficiency')
    ),
    AgainstElectronEff = cms.PSet(
      plots = cms.PSet(
        dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/shrinkingConePFTauProducerLeadingPion_shrinkingConePFTauDiscriminationAgainstElectron/AgainstElectronEff#PAR#'), 
        processes = cms.vstring('test', 'reference')
      ),
      parameter = cms.vstring('pt', 'eta', 'phi', 'energy'),
      xAxis = cms.string('#PAR#'),
      yAxis = cms.string('efficiency'),
      legend = cms.string('efficiency'),
      labels = cms.vstring('pt', 'eta'),
      drawOptionSet = cms.string('efficiency')
    ),
    AgainstMuonEff = cms.PSet(
      plots = cms.PSet(
        dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/shrinkingConePFTauProducerLeadingPion_shrinkingConePFTauDiscriminationAgainstMuon/AgainstMuonEff#PAR#'), 
        processes = cms.vstring('test', 'reference')
      ),
      parameter = cms.vstring('pt', 'eta', 'phi', 'energy'),
      xAxis = cms.string('#PAR#'),
      yAxis = cms.string('efficiency'),
      legend = cms.string('efficiency'),
      labels = cms.vstring('pt', 'eta'),
      drawOptionSet = cms.string('efficiency')
    ),
    TauIdEffStepByStep = cms.PSet(
      plots = cms.VPSet(
        cms.PSet(
          dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/shrinkingConePFTauProducerLeadingPion_Matched/PFJetMatchingEff#PAR#'),
          process = cms.string('test'),
          drawOptionEntry = cms.string('eff_overlay01'),
          legendEntry = cms.string('PFJet Matching')
        ),    
        cms.PSet(
          dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/shrinkingConePFTauProducerLeadingPion_shrinkingConePFTauDiscriminationByLeadingPionPtCut/LeadingPionPtCutEff#PAR#'),
          process = cms.string('test'),
          drawOptionEntry = cms.string('eff_overlay02'),
          legendEntry = cms.string('Lead Pion Finding')
        ),
        cms.PSet(
          dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/shrinkingConePFTauProducerLeadingPion_shrinkingConePFTauDiscriminationByTrackIsolationUsingLeadingPion/TrackIsolationEff#PAR#'),
          process = cms.string('test'),
          drawOptionEntry = cms.string('eff_overlay03'),
          legendEntry = cms.string('Track Iso. Using Lead. Pion')
        ),
        cms.PSet(
          dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/shrinkingConePFTauProducerLeadingPion_shrinkingConePFTauDiscriminationByECALIsolationUsingLeadingPion/ECALIsolationEff#PAR#'),
          process = cms.string('test'),
          drawOptionEntry = cms.string('eff_overlay04'),
          legendEntry = cms.string('Track + Gamma Iso. Using Lead. Pioon')
        ),
        cms.PSet(
          dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/shrinkingConePFTauProducerLeadingPion_shrinkingConePFTauDiscriminationAgainstElectron/AgainstElectronEff#PAR#'),
          process = cms.string('test'),
          drawOptionEntry = cms.string('eff_overlay05'),
          legendEntry = cms.string('Electron Rejection')
        ),
        cms.PSet(
          dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/shrinkingConePFTauProducerLeadingPion_shrinkingConePFTauDiscriminationAgainstMuon/AgainstMuonEff#PAR#'),
          process = cms.string('test'),
          drawOptionEntry = cms.string('eff_overlay06'),
          legendEntry = cms.string('Muon Rejection')
        )
      ),
      parameter = cms.vstring('pt', 'eta', 'phi', 'energy'),
      title = cms.string('TauId step by step efficiencies'),
      xAxis = cms.string('#PAR#'),
      yAxis = cms.string('efficiency'),
      legend = cms.string('efficiency_overlay'),
      labels = cms.vstring('pt', 'eta')
    )
  ),
  outputFilePath = cms.string('./shrinkingConePFTauProducerLeadingPion/'),
)
