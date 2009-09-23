
import FWCore.ParameterSet.Config as cms

loadTau = cms.EDAnalyzer("DQMFileLoader",
  test = cms.PSet(
    inputFileNames = cms.vstring('/afs/cern.ch/user/v/vasquez/scratch0/CMSSW_2_2_1/src/Validation/RecoTau/test/FastSim_CMSSW_2_2_1_HadronicTauOneAndThreeProng_ALL_Florian.root'),
    scaleFactor = cms.double(1.),
    dqmDirectory_store = cms.string('test')
  ),
  reference = cms.PSet(
    inputFileNames = cms.vstring('/afs/cern.ch/user/v/vasquez/scratch0/CMSSW_2_2_1/src/Validation/RecoTau/test/CMSSW_2_2_1_HadronicTauOneAndThreeProng_ALL_NewDiscriminators.root'),
    scaleFactor = cms.double(1.),
    dqmDirectory_store = cms.string('reference')
  )
)

plotPFTauEfficiencies = cms.EDAnalyzer("DQMHistPlotter",
  processes = cms.PSet(
    test = cms.PSet(
      dqmDirectory = cms.string('test'),
      legendEntry = cms.string('CMSSW_2_2_1 FastSim by Florian'),
      type = cms.string('smMC') # Data/smMC/bsmMC/smSumMC
    ),
    reference = cms.PSet(
      dqmDirectory = cms.string('reference'),
      legendEntry = cms.string('CMSSW_2_2_1 Full'),
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
    ),
    eff_overlay06 = cms.PSet(
      markerColor = cms.int32(6),
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
        dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/fixedConePFTauProducer_fixedConePFTauDiscriminationByTrackIsolation/IsolationEff#PAR#'),
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
        dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/fixedConePFTauProducer_fixedConePFTauDiscriminationByECALIsolation/IsolationEff#PAR#'),
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
          dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/fixedConePFTauProducer_fixedConePFTauDiscriminationByTrackIsolation/IsolationEff#PAR#'),
          process = cms.string('test'),
          drawOptionEntry = cms.string('eff_overlay03'),
          legendEntry = cms.string('Track')
        ),
        cms.PSet(
          dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/fixedConePFTauProducer_fixedConePFTauDiscriminationByECALIsolation/IsolationEff#PAR#'),
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

  outputFilePath = cms.string('./fastSimfixedConePFTauProducerFlorian/HadronicTauOneAndThreeProng/'),
#  outputFileName = cms.string('FIRSTTEST.ps'),
  indOutputFileName = cms.string('#PLOT#.png')    
)                    

plotPFTauHighEfficiencyEfficiencies = cms.EDAnalyzer("DQMHistPlotter",
  processes = cms.PSet(
    test = cms.PSet(
      dqmDirectory = cms.string('test'),
      legendEntry = cms.string('CMSSW_2_2_1 FastSim by Florian'),
      type = cms.string('smMC') # Data/smMC/bsmMC/smSumMC
    ),
    reference = cms.PSet(
      dqmDirectory = cms.string('reference'),
      legendEntry = cms.string('CMSSW_2_2_1 Full'),
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
    ),
    eff_overlay06 = cms.PSet(
      markerColor = cms.int32(6),
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
        dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/shrinkingConePFTauProducer_shrinkingConePFTauDiscriminationByTrackIsolation/IsolationEff#PAR#'),
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
        dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/shrinkingConePFTauProducer_shrinkingConePFTauDiscriminationByECALIsolation/IsolationEff#PAR#'),
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
          dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/shrinkingConePFTauProducer_shrinkingConePFTauDiscriminationByTrackIsolation/IsolationEff#PAR#'),
          process = cms.string('test'),
          drawOptionEntry = cms.string('eff_overlay03'),
          legendEntry = cms.string('Track')
        ),
        cms.PSet(
          dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/shrinkingConePFTauProducer_shrinkingConePFTauDiscriminationByECALIsolation/IsolationEff#PAR#'),
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

  outputFilePath = cms.string('./fastSimshrinkingConePFTauProducerFlorian/HadronicTauOneAndThreeProng/'),
#  outputFileName = cms.string('FIRSTTEST.ps'),
  indOutputFileName = cms.string('#PLOT#.png')    
)      




plotCaloTauEfficiencies = cms.EDAnalyzer("DQMHistPlotter",
  processes = cms.PSet(
    test = cms.PSet(
      dqmDirectory = cms.string('test'),
      legendEntry = cms.string('CMSSW_2_2_1 FastSim by Florian'),
      type = cms.string('smMC') # Data/smMC/bsmMC/smSumMC
    ),
    reference = cms.PSet(
      dqmDirectory = cms.string('reference'),
      legendEntry = cms.string('CMSSW_2_2_1 Full'),
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
          legendEntry = cms.string('PFJet Matching')
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
          legendEntry = cms.string('Track')
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

  outputFilePath = cms.string('./fastSimcaloRecoTauProducerFlorian/HadronicTauOneAndThreeProng/'),
#  outputFileName = cms.string('FIRSTTEST.ps'),
  indOutputFileName = cms.string('#PLOT#.png')    
)                    
