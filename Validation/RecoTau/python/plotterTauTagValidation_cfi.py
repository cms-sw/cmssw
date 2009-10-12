# This test config file that needs to be modified to have two files being compared
#

import FWCore.ParameterSet.Config as cms

#########################################################
# Give the correct test and reference labels on the plots
#########################################################

test = cms.string('CMSSW_3_1_0_pre2')
reference = cms.string('CMSSW_3_1_0_pre1')

###########################
# Load Files to be compared
###########################

loadTau = cms.EDAnalyzer("DQMFileLoader",
  test = cms.PSet(
    inputFileNames = cms.vstring('/afs/cern.ch/user/v/vasquez/scratch0/removavableTest/CMSSW_3_1_0_pre2/src/Validation/RecoTau/test/CMSSW_3_1_0_pre2_tauGenJets.root'),
    scaleFactor = cms.double(1.),
    dqmDirectory_store = cms.string('test')
  ),
  reference = cms.PSet(
    inputFileNames = cms.vstring('/afs/cern.ch/user/v/vasquez/scratch0/CMSSW_3_1_0_pre1/src/Validation/RecoTau/test/CMSSW_3_1_0_pre1_tauGenJets.root'),
    scaleFactor = cms.double(1.),
    dqmDirectory_store = cms.string('reference')
  )
)

# standard drawing stuff
standardDrawingStuff = cms.PSet(
  canvasSizeX = cms.int32(640),
  canvasSizeY = cms.int32(640),                         
  indOutputFileName = cms.string('#PLOT#.png'),
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
)

standardCompareTestAndReference = cms.PSet(
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
)


