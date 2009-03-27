
import FWCore.ParameterSet.Config as cms

from Validation.RecoTau.plotterTauTagValidation_cfi import *

##################################################
#
#   The plotting of all the PFTau ID efficiencies
#
##################################################
plotPFTauEfficiencies = cms.EDAnalyzer("DQMHistPlotter",
  processes = cms.PSet(
    test = cms.PSet(
      dqmDirectory = cms.string('test'),
      legendEntry = test,
      type = cms.string('smMC') # Data/smMC/bsmMC/smSumMC
    ),
    reference = cms.PSet(
      dqmDirectory = cms.string('reference'),
      legendEntry = reference,
      type = cms.string('smMC') # Data/smMC/bsmMC/smSumMC
    )
  ),
                                     
  xAxes = cms.PSet(
    pt = cms.PSet(
      xAxisTitle = cms.string('P_{T} / GeV'),
      xAxisTitleOffset = cms.double(0.9),
      xAxisTitleSize = cms.double(0.05)
    ),
    eta = cms.PSet(
      xAxisTitle = cms.string('#eta'),
      xAxisTitleOffset = cms.double(0.9),
      xAxisTitleSize = cms.double(0.05)
    ),
    phi = cms.PSet(
      xAxisTitle = cms.string('#phi'),
      xAxisTitleOffset = cms.double(0.9),
      xAxisTitleSize = cms.double(0.05)
    ),
    energy = cms.PSet(
      xAxisTitle = cms.string('E / GeV'),
      xAxisTitleOffset = cms.double(0.9),
      xAxisTitleSize = cms.double(0.05)
    )
  ),

  yAxes = cms.PSet(                         
    efficiency = cms.PSet(
      yScale = cms.string('linear'), # linear/log
      minY_linear = cms.double(0.),
      maxY_linear = cms.double(1.6),
      yAxisTitle = cms.string('#varepsilon'), 
      yAxisTitleOffset = cms.double(1.1),
      yAxisTitleSize = cms.double(0.05)
    )
  ),

  legends = cms.PSet(
    efficiency = cms.PSet(
      posX = cms.double(0.50),
      posY = cms.double(0.72),
      sizeX = cms.double(0.39),
      sizeY = cms.double(0.17),
      header = cms.string(''),
      option = cms.string('brNDC'),
      borderSize = cms.int32(0),
      fillColor = cms.int32(0)
    ),
    efficiency_overlay = cms.PSet(
      posX = cms.double(0.50),
      posY = cms.double(0.66),
      sizeX = cms.double(0.39),
      sizeY = cms.double(0.23),
      header = cms.string(''),
      option = cms.string('brNDC'),
      borderSize = cms.int32(0),
      fillColor = cms.int32(0)
    )
  ),

  labels = cms.PSet(
    pt = cms.PSet(
      posX = cms.double(0.19),
      posY = cms.double(0.77),
      sizeX = cms.double(0.12),
      sizeY = cms.double(0.04),
      option = cms.string('brNDC'),
      borderSize = cms.int32(0),
      fillColor = cms.int32(0),
      textColor = cms.int32(1),
      textSize = cms.double(0.04),
      textAlign = cms.int32(22),
      text = cms.vstring('P_{T} > 5 GeV')
    ),
    eta = cms.PSet(
      posX = cms.double(0.19),
      posY = cms.double(0.83),
      sizeX = cms.double(0.12),
      sizeY = cms.double(0.04),
      option = cms.string('brNDC'),
      borderSize = cms.int32(0),
      fillColor = cms.int32(0),
      textColor = cms.int32(1),
      textSize = cms.double(0.04),
      textAlign = cms.int32(22),
      text = cms.vstring('-2.5 < #eta < +2.5')
    )
  ),

  drawOptionSets = cms.PSet(
    efficiency = cms.PSet(
      test = cms.PSet(
        markerColor = cms.int32(4),
        markerSize = cms.double(1.),
        markerStyle = cms.int32(20),
        lineColor = cms.int32(1),
        lineStyle = cms.int32(1),
        lineWidth = cms.int32(1),
        drawOption = cms.string('ep'),
        drawOptionLegend = cms.string('p')
      ),
      reference = cms.PSet(
        lineColor = cms.int32(1),
        lineStyle = cms.int32(1),
        lineWidth = cms.int32(1),
        fillColor = cms.int32(41),
        drawOption = cms.string('eBand'),
        drawOptionLegend = cms.string('l')
      )
    )
  ),
                                     
  drawOptionEntries = cms.PSet(
    eff_overlay01 = cms.PSet(
      markerColor = cms.int32(1),
      markerSize = cms.double(1.),
      markerStyle = cms.int32(20),
      lineColor = cms.int32(1),
      lineStyle = cms.int32(1),
      lineWidth = cms.int32(2),
      drawOption = cms.string('ex0'),
      drawOptionLegend = cms.string('p')
    ),
    eff_overlay02 = cms.PSet(
      markerColor = cms.int32(2),
      markerSize = cms.double(1.),
      markerStyle = cms.int32(20),
      lineColor = cms.int32(2),
      lineStyle = cms.int32(1),
      lineWidth = cms.int32(2),
      drawOption = cms.string('ex0'),
      drawOptionLegend = cms.string('p')
    ),
    eff_overlay03 = cms.PSet(
      markerColor = cms.int32(3),
      markerSize = cms.double(1.),
      markerStyle = cms.int32(20),
      lineColor = cms.int32(3),
      lineStyle = cms.int32(1),
      lineWidth = cms.int32(2),
      drawOption = cms.string('ex0'),
      drawOptionLegend = cms.string('p')
    ),
    eff_overlay04 = cms.PSet(
      markerColor = cms.int32(4),
      markerSize = cms.double(1.),
      markerStyle = cms.int32(20),
      lineColor = cms.int32(4),
      lineStyle = cms.int32(1),
      lineWidth = cms.int32(2),
      drawOption = cms.string('ex0'),
      drawOptionLegend = cms.string('p')
    ),
    eff_overlay05 = cms.PSet(
      markerColor = cms.int32(6),
      markerSize = cms.double(1.),
      markerStyle = cms.int32(20),
      lineColor = cms.int32(6),
      lineStyle = cms.int32(1),
      lineWidth = cms.int32(2),
      drawOption = cms.string('ex0'),
      drawOptionLegend = cms.string('p')
    ),
    eff_overlay06 = cms.PSet(
      markerColor = cms.int32(5),
      markerSize = cms.double(1.),
      markerStyle = cms.int32(20),
      lineColor = cms.int32(5),
      lineStyle = cms.int32(1),
      lineWidth = cms.int32(2),
      drawOption = cms.string('ex0'),
      drawOptionLegend = cms.string('p')
    )
  ),

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

  canvasSizeX = cms.int32(640),
  canvasSizeY = cms.int32(640),                         

  outputFilePath = cms.string('./fixedConePFTauProducer/'),
#  outputFileName = cms.string('FIRSTTEST.ps'),
  indOutputFileName = cms.string('#PLOT#.png')    
)                    

##################################################
#
#   The plotting of all the PFTauHighEfficiencies ID efficiencies
#
##################################################

plotPFTauHighEfficiencyEfficiencies = cms.EDAnalyzer("DQMHistPlotter",
  processes = cms.PSet(
    test = cms.PSet(
      dqmDirectory = cms.string('test'),
      legendEntry = test,
      type = cms.string('smMC') # Data/smMC/bsmMC/smSumMC
    ),
    reference = cms.PSet(
      dqmDirectory = cms.string('reference'),
      legendEntry = reference,
      type = cms.string('smMC') # Data/smMC/bsmMC/smSumMC
    )
  ),
                                     
  xAxes = cms.PSet(
    pt = cms.PSet(
      xAxisTitle = cms.string('P_{T} / GeV'),
      xAxisTitleOffset = cms.double(0.9),
      xAxisTitleSize = cms.double(0.05)
    ),
    eta = cms.PSet(
      xAxisTitle = cms.string('#eta'),
      xAxisTitleOffset = cms.double(0.9),
      xAxisTitleSize = cms.double(0.05)
    ),
    phi = cms.PSet(
      xAxisTitle = cms.string('#phi'),
      xAxisTitleOffset = cms.double(0.9),
      xAxisTitleSize = cms.double(0.05)
    ),
    energy = cms.PSet(
      xAxisTitle = cms.string('E / GeV'),
      xAxisTitleOffset = cms.double(0.9),
      xAxisTitleSize = cms.double(0.05)
    )
  ),

  yAxes = cms.PSet(                         
    efficiency = cms.PSet(
      yScale = cms.string('linear'), # linear/log
      minY_linear = cms.double(0.),
      maxY_linear = cms.double(1.6),
      yAxisTitle = cms.string('#varepsilon'), 
      yAxisTitleOffset = cms.double(1.1),
      yAxisTitleSize = cms.double(0.05)
    )
  ),

  legends = cms.PSet(
    efficiency = cms.PSet(
      posX = cms.double(0.50),
      posY = cms.double(0.72),
      sizeX = cms.double(0.39),
      sizeY = cms.double(0.17),
      header = cms.string(''),
      option = cms.string('brNDC'),
      borderSize = cms.int32(0),
      fillColor = cms.int32(0)
    ),
    efficiency_overlay = cms.PSet(
      posX = cms.double(0.50),
      posY = cms.double(0.66),
      sizeX = cms.double(0.39),
      sizeY = cms.double(0.23),
      header = cms.string(''),
      option = cms.string('brNDC'),
      borderSize = cms.int32(0),
      fillColor = cms.int32(0)
    )
  ),

  labels = cms.PSet(
    pt = cms.PSet(
      posX = cms.double(0.19),
      posY = cms.double(0.77),
      sizeX = cms.double(0.12),
      sizeY = cms.double(0.04),
      option = cms.string('brNDC'),
      borderSize = cms.int32(0),
      fillColor = cms.int32(0),
      textColor = cms.int32(1),
      textSize = cms.double(0.04),
      textAlign = cms.int32(22),
      text = cms.vstring('P_{T} > 5 GeV')
    ),
    eta = cms.PSet(
      posX = cms.double(0.19),
      posY = cms.double(0.83),
      sizeX = cms.double(0.12),
      sizeY = cms.double(0.04),
      option = cms.string('brNDC'),
      borderSize = cms.int32(0),
      fillColor = cms.int32(0),
      textColor = cms.int32(1),
      textSize = cms.double(0.04),
      textAlign = cms.int32(22),
      text = cms.vstring('-2.5 < #eta < +2.5')
    )
  ),

  drawOptionSets = cms.PSet(
    efficiency = cms.PSet(
      test = cms.PSet(
        markerColor = cms.int32(4),
        markerSize = cms.double(1.),
        markerStyle = cms.int32(20),
        lineColor = cms.int32(1),
        lineStyle = cms.int32(1),
        lineWidth = cms.int32(1),
        drawOption = cms.string('ep'),
        drawOptionLegend = cms.string('p')
      ),
      reference = cms.PSet(
        lineColor = cms.int32(1),
        lineStyle = cms.int32(1),
        lineWidth = cms.int32(1),
        fillColor = cms.int32(41),
        drawOption = cms.string('eBand'),
        drawOptionLegend = cms.string('l')
      )
    )
  ),
                                     
  drawOptionEntries = cms.PSet(
    eff_overlay01 = cms.PSet(
      markerColor = cms.int32(1),
      markerSize = cms.double(1.),
      markerStyle = cms.int32(20),
      lineColor = cms.int32(1),
      lineStyle = cms.int32(1),
      lineWidth = cms.int32(2),
      drawOption = cms.string('ex0'),
      drawOptionLegend = cms.string('p')
    ),
    eff_overlay02 = cms.PSet(
      markerColor = cms.int32(2),
      markerSize = cms.double(1.),
      markerStyle = cms.int32(20),
      lineColor = cms.int32(2),
      lineStyle = cms.int32(1),
      lineWidth = cms.int32(2),
      drawOption = cms.string('ex0'),
      drawOptionLegend = cms.string('p')
    ),
    eff_overlay03 = cms.PSet(
      markerColor = cms.int32(3),
      markerSize = cms.double(1.),
      markerStyle = cms.int32(20),
      lineColor = cms.int32(3),
      lineStyle = cms.int32(1),
      lineWidth = cms.int32(2),
      drawOption = cms.string('ex0'),
      drawOptionLegend = cms.string('p')
    ),
    eff_overlay04 = cms.PSet(
      markerColor = cms.int32(4),
      markerSize = cms.double(1.),
      markerStyle = cms.int32(20),
      lineColor = cms.int32(4),
      lineStyle = cms.int32(1),
      lineWidth = cms.int32(2),
      drawOption = cms.string('ex0'),
      drawOptionLegend = cms.string('p')
    ),
    eff_overlay05 = cms.PSet(
      markerColor = cms.int32(6),
      markerSize = cms.double(1.),
      markerStyle = cms.int32(20),
      lineColor = cms.int32(6),
      lineStyle = cms.int32(1),
      lineWidth = cms.int32(2),
      drawOption = cms.string('ex0'),
      drawOptionLegend = cms.string('p')
    ),
    eff_overlay06 = cms.PSet(
      markerColor = cms.int32(5),
      markerSize = cms.double(1.),
      markerStyle = cms.int32(20),
      lineColor = cms.int32(5),
      lineStyle = cms.int32(1),
      lineWidth = cms.int32(2),
      drawOption = cms.string('ex0'),
      drawOptionLegend = cms.string('p')
    )
  ),

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

  canvasSizeX = cms.int32(640),
  canvasSizeY = cms.int32(640),                         

  outputFilePath = cms.string('./shrinkingConePFTauProducer/'),
#  outputFileName = cms.string('FIRSTTEST.ps'),
  indOutputFileName = cms.string('#PLOT#.png')    
)      

##################################################
#
#   The plotting of all the CaloTau ID efficiencies
#
##################################################

plotCaloTauEfficiencies = cms.EDAnalyzer("DQMHistPlotter",
  processes = cms.PSet(
    test = cms.PSet(
      dqmDirectory = cms.string('test'),
      legendEntry = test,
      type = cms.string('smMC') # Data/smMC/bsmMC/smSumMC
    ),
    reference = cms.PSet(
      dqmDirectory = cms.string('reference'),
      legendEntry = reference,
      type = cms.string('smMC') # Data/smMC/bsmMC/smSumMC
    )
  ),
                                     
  xAxes = cms.PSet(
    pt = cms.PSet(
      xAxisTitle = cms.string('P_{T} / GeV'),
      xAxisTitleOffset = cms.double(0.9),
      xAxisTitleSize = cms.double(0.05)
    ),
    eta = cms.PSet(
      xAxisTitle = cms.string('#eta'),
      xAxisTitleOffset = cms.double(0.9),
      xAxisTitleSize = cms.double(0.05)
    ),
    phi = cms.PSet(
      xAxisTitle = cms.string('#phi'),
      xAxisTitleOffset = cms.double(0.9),
      xAxisTitleSize = cms.double(0.05)
    ),
    energy = cms.PSet(
      xAxisTitle = cms.string('E / GeV'),
      xAxisTitleOffset = cms.double(0.9),
      xAxisTitleSize = cms.double(0.05)
    )
  ),

  yAxes = cms.PSet(                         
    efficiency = cms.PSet(
      yScale = cms.string('linear'), # linear/log
      minY_linear = cms.double(0.),
      maxY_linear = cms.double(1.6),
      yAxisTitle = cms.string('#varepsilon'), 
      yAxisTitleOffset = cms.double(1.1),
      yAxisTitleSize = cms.double(0.05)
    )
  ),

  legends = cms.PSet(
    efficiency = cms.PSet(
      posX = cms.double(0.50),
      posY = cms.double(0.72),
      sizeX = cms.double(0.39),
      sizeY = cms.double(0.17),
      header = cms.string(''),
      option = cms.string('brNDC'),
      borderSize = cms.int32(0),
      fillColor = cms.int32(0)
    ),
    efficiency_overlay = cms.PSet(
      posX = cms.double(0.50),
      posY = cms.double(0.66),
      sizeX = cms.double(0.39),
      sizeY = cms.double(0.23),
      header = cms.string(''),
      option = cms.string('brNDC'),
      borderSize = cms.int32(0),
      fillColor = cms.int32(0)
    )
  ),

  labels = cms.PSet(
    pt = cms.PSet(
      posX = cms.double(0.19),
      posY = cms.double(0.77),
      sizeX = cms.double(0.12),
      sizeY = cms.double(0.04),
      option = cms.string('brNDC'),
      borderSize = cms.int32(0),
      fillColor = cms.int32(0),
      textColor = cms.int32(1),
      textSize = cms.double(0.04),
      textAlign = cms.int32(22),
      text = cms.vstring('P_{T} > 5 GeV')
    ),
    eta = cms.PSet(
      posX = cms.double(0.19),
      posY = cms.double(0.83),
      sizeX = cms.double(0.12),
      sizeY = cms.double(0.04),
      option = cms.string('brNDC'),
      borderSize = cms.int32(0),
      fillColor = cms.int32(0),
      textColor = cms.int32(1),
      textSize = cms.double(0.04),
      textAlign = cms.int32(22),
      text = cms.vstring('-2.5 < #eta < +2.5')
    )
  ),

  drawOptionSets = cms.PSet(
    efficiency = cms.PSet(
      test = cms.PSet(
        markerColor = cms.int32(4),
        markerSize = cms.double(1.),
        markerStyle = cms.int32(20),
        lineColor = cms.int32(1),
        lineStyle = cms.int32(1),
        lineWidth = cms.int32(1),
        drawOption = cms.string('ep'),
        drawOptionLegend = cms.string('p')
      ),
      reference = cms.PSet(
        lineColor = cms.int32(1),
        lineStyle = cms.int32(1),
        lineWidth = cms.int32(1),
        fillColor = cms.int32(41),
        drawOption = cms.string('eBand'),
        drawOptionLegend = cms.string('l')
      )
    )
  ),
                                     
  drawOptionEntries = cms.PSet(
    eff_overlay01 = cms.PSet(
      markerColor = cms.int32(1),
      markerSize = cms.double(1.),
      markerStyle = cms.int32(20),
      lineColor = cms.int32(1),
      lineStyle = cms.int32(1),
      lineWidth = cms.int32(2),
      drawOption = cms.string('ex0'),
      drawOptionLegend = cms.string('p')
    ),
    eff_overlay02 = cms.PSet(
      markerColor = cms.int32(2),
      markerSize = cms.double(1.),
      markerStyle = cms.int32(20),
      lineColor = cms.int32(2),
      lineStyle = cms.int32(1),
      lineWidth = cms.int32(2),
      drawOption = cms.string('ex0'),
      drawOptionLegend = cms.string('p')
    ),
    eff_overlay03 = cms.PSet(
      markerColor = cms.int32(3),
      markerSize = cms.double(1.),
      markerStyle = cms.int32(20),
      lineColor = cms.int32(3),
      lineStyle = cms.int32(1),
      lineWidth = cms.int32(2),
      drawOption = cms.string('ex0'),
      drawOptionLegend = cms.string('p')
    ),
    eff_overlay04 = cms.PSet(
      markerColor = cms.int32(4),
      markerSize = cms.double(1.),
      markerStyle = cms.int32(20),
      lineColor = cms.int32(4),
      lineStyle = cms.int32(1),
      lineWidth = cms.int32(2),
      drawOption = cms.string('ex0'),
      drawOptionLegend = cms.string('p')
    ),
    eff_overlay05 = cms.PSet(
      markerColor = cms.int32(5),
      markerSize = cms.double(1.),
      markerStyle = cms.int32(20),
      lineColor = cms.int32(5),
      lineStyle = cms.int32(1),
      lineWidth = cms.int32(2),
      drawOption = cms.string('ex0'),
      drawOptionLegend = cms.string('p')
    )
  ),

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

  canvasSizeX = cms.int32(640),
  canvasSizeY = cms.int32(640),                         

  outputFilePath = cms.string('./caloRecoTauProducer/'),
#  outputFileName = cms.string('FIRSTTEST.ps'),
  indOutputFileName = cms.string('#PLOT#.png')    
)

##################################################
#
#   The plotting of all the PFTauHighEfficiencyUsingLeadingPion ID efficiencies
#
##################################################

plotPFTauHighEfficiencyEfficienciesLeadingPion = cms.EDAnalyzer("DQMHistPlotter",
  processes = cms.PSet(
    test = cms.PSet(
      dqmDirectory = cms.string('test'),
      legendEntry = test,
      type = cms.string('smMC') # Data/smMC/bsmMC/smSumMC
    ),
    reference = cms.PSet(
      dqmDirectory = cms.string('reference'),
      legendEntry = reference,
      type = cms.string('smMC') # Data/smMC/bsmMC/smSumMC
    )
  ),
                                     
  xAxes = cms.PSet(
    pt = cms.PSet(
      xAxisTitle = cms.string('P_{T} / GeV'),
      xAxisTitleOffset = cms.double(0.9),
      xAxisTitleSize = cms.double(0.05)
    ),
    eta = cms.PSet(
      xAxisTitle = cms.string('#eta'),
      xAxisTitleOffset = cms.double(0.9),
      xAxisTitleSize = cms.double(0.05)
    ),
    phi = cms.PSet(
      xAxisTitle = cms.string('#phi'),
      xAxisTitleOffset = cms.double(0.9),
      xAxisTitleSize = cms.double(0.05)
    ),
    energy = cms.PSet(
      xAxisTitle = cms.string('E / GeV'),
      xAxisTitleOffset = cms.double(0.9),
      xAxisTitleSize = cms.double(0.05)
    )
  ),

  yAxes = cms.PSet(                         
    efficiency = cms.PSet(
      yScale = cms.string('linear'), # linear/log
      minY_linear = cms.double(0.),
      maxY_linear = cms.double(1.6),
      yAxisTitle = cms.string('#varepsilon'), 
      yAxisTitleOffset = cms.double(1.1),
      yAxisTitleSize = cms.double(0.05)
    )
  ),

  legends = cms.PSet(
    efficiency = cms.PSet(
      posX = cms.double(0.50),
      posY = cms.double(0.72),
      sizeX = cms.double(0.39),
      sizeY = cms.double(0.17),
      header = cms.string(''),
      option = cms.string('brNDC'),
      borderSize = cms.int32(0),
      fillColor = cms.int32(0)
    ),
    efficiency_overlay = cms.PSet(
      posX = cms.double(0.50),
      posY = cms.double(0.66),
      sizeX = cms.double(0.39),
      sizeY = cms.double(0.23),
      header = cms.string(''),
      option = cms.string('brNDC'),
      borderSize = cms.int32(0),
      fillColor = cms.int32(0)
    )
  ),

  labels = cms.PSet(
    pt = cms.PSet(
      posX = cms.double(0.19),
      posY = cms.double(0.77),
      sizeX = cms.double(0.12),
      sizeY = cms.double(0.04),
      option = cms.string('brNDC'),
      borderSize = cms.int32(0),
      fillColor = cms.int32(0),
      textColor = cms.int32(1),
      textSize = cms.double(0.04),
      textAlign = cms.int32(22),
      text = cms.vstring('P_{T} > 5 GeV')
    ),
    eta = cms.PSet(
      posX = cms.double(0.19),
      posY = cms.double(0.83),
      sizeX = cms.double(0.12),
      sizeY = cms.double(0.04),
      option = cms.string('brNDC'),
      borderSize = cms.int32(0),
      fillColor = cms.int32(0),
      textColor = cms.int32(1),
      textSize = cms.double(0.04),
      textAlign = cms.int32(22),
      text = cms.vstring('-2.5 < #eta < +2.5')
    )
  ),

  drawOptionSets = cms.PSet(
    efficiency = cms.PSet(
      test = cms.PSet(
        markerColor = cms.int32(4),
        markerSize = cms.double(1.),
        markerStyle = cms.int32(20),
        lineColor = cms.int32(1),
        lineStyle = cms.int32(1),
        lineWidth = cms.int32(1),
        drawOption = cms.string('ep'),
        drawOptionLegend = cms.string('p')
      ),
      reference = cms.PSet(
        lineColor = cms.int32(1),
        lineStyle = cms.int32(1),
        lineWidth = cms.int32(1),
        fillColor = cms.int32(41),
        drawOption = cms.string('eBand'),
        drawOptionLegend = cms.string('l')
      )
    )
  ),
                                     
  drawOptionEntries = cms.PSet(
    eff_overlay01 = cms.PSet(
      markerColor = cms.int32(1),
      markerSize = cms.double(1.),
      markerStyle = cms.int32(20),
      lineColor = cms.int32(1),
      lineStyle = cms.int32(1),
      lineWidth = cms.int32(2),
      drawOption = cms.string('ex0'),
      drawOptionLegend = cms.string('p')
    ),
    eff_overlay02 = cms.PSet(
      markerColor = cms.int32(2),
      markerSize = cms.double(1.),
      markerStyle = cms.int32(20),
      lineColor = cms.int32(2),
      lineStyle = cms.int32(1),
      lineWidth = cms.int32(2),
      drawOption = cms.string('ex0'),
      drawOptionLegend = cms.string('p')
    ),
    eff_overlay03 = cms.PSet(
      markerColor = cms.int32(3),
      markerSize = cms.double(1.),
      markerStyle = cms.int32(20),
      lineColor = cms.int32(3),
      lineStyle = cms.int32(1),
      lineWidth = cms.int32(2),
      drawOption = cms.string('ex0'),
      drawOptionLegend = cms.string('p')
    ),
    eff_overlay04 = cms.PSet(
      markerColor = cms.int32(4),
      markerSize = cms.double(1.),
      markerStyle = cms.int32(20),
      lineColor = cms.int32(4),
      lineStyle = cms.int32(1),
      lineWidth = cms.int32(2),
      drawOption = cms.string('ex0'),
      drawOptionLegend = cms.string('p')
    ),
    eff_overlay05 = cms.PSet(
      markerColor = cms.int32(6),
      markerSize = cms.double(1.),
      markerStyle = cms.int32(20),
      lineColor = cms.int32(6),
      lineStyle = cms.int32(1),
      lineWidth = cms.int32(2),
      drawOption = cms.string('ex0'),
      drawOptionLegend = cms.string('p')
    ),
    eff_overlay06 = cms.PSet(
      markerColor = cms.int32(5),
      markerSize = cms.double(1.),
      markerStyle = cms.int32(20),
      lineColor = cms.int32(5),
      lineStyle = cms.int32(1),
      lineWidth = cms.int32(2),
      drawOption = cms.string('ex0'),
      drawOptionLegend = cms.string('p')
    )
  ),

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

  canvasSizeX = cms.int32(640),
  canvasSizeY = cms.int32(640),                         

  outputFilePath = cms.string('./shrinkingConePFTauProducerLeadingPion/'),
#  outputFileName = cms.string('FIRSTTEST.ps'),
  indOutputFileName = cms.string('#PLOT#.png')    
)
