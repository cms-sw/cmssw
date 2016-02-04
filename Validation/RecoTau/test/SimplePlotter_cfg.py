import FWCore.ParameterSet.Config as cms

process = cms.Process("MakingPlots")

process.source = cms.Source("EmptySource")
process.loadTau = cms.EDAnalyzer("DQMFileLoader",
    test = cms.PSet(
        inputFileNames = cms.vstring('TauVal_CMSSW_3_3_3_QCD.root'),
        scaleFactor = cms.double(1.0),
        dqmDirectory_store = cms.string('DQMData')
    )
)


process.plotPFTauHighEfficiencyEfficienciesLeadingPion = cms.EDAnalyzer("DQMHistPlotter",
    processes = cms.PSet(
        test = cms.PSet(
            dqmDirectory = cms.string('DQMData'),
            type = cms.string('smMC'),
            legendEntry = cms.string('no test label')
        ),
        reference = cms.PSet(
            dqmDirectory = cms.string('reference'),
            type = cms.string('smMC'),
            legendEntry = cms.string('no ref label')
        )
    ),
    legends = cms.PSet(
        efficiency = cms.PSet(
            option = cms.string('brNDC'),
            sizeX = cms.double(0.39),
            sizeY = cms.double(0.17),
            borderSize = cms.int32(0),
            header = cms.string(''),
            fillColor = cms.int32(0),
            posX = cms.double(0.5),
            posY = cms.double(0.72)
        ),
        efficiency_overlay = cms.PSet(
            option = cms.string('brNDC'),
            sizeX = cms.double(0.39),
            sizeY = cms.double(0.23),
            borderSize = cms.int32(0),
            header = cms.string(''),
            fillColor = cms.int32(0),
            posX = cms.double(0.5),
            posY = cms.double(0.66)
        )
    ),
    labels = cms.PSet(
        eta = cms.PSet(
            textSize = cms.double(0.04),
            option = cms.string('brNDC'),
            text = cms.vstring('-2.5 < #eta < +2.5'),
            sizeX = cms.double(0.12),
            sizeY = cms.double(0.04),
            borderSize = cms.int32(0),
            fillColor = cms.int32(0),
            posX = cms.double(0.19),
            posY = cms.double(0.83),
            textColor = cms.int32(1),
            textAlign = cms.int32(22)
        ),
        pt = cms.PSet(
            textSize = cms.double(0.04),
            option = cms.string('brNDC'),
            text = cms.vstring('P_{T} > 10 GeV'),
            sizeX = cms.double(0.12),
            sizeY = cms.double(0.04),
            borderSize = cms.int32(0),
            fillColor = cms.int32(0),
            posX = cms.double(0.19),
            posY = cms.double(0.77),
            textColor = cms.int32(1),
            textAlign = cms.int32(22)
        )
    ),
    indOutputFileName = cms.string('#PLOT#.png'),
    drawOptionSets = cms.PSet(
        efficiency = cms.PSet(
            test = cms.PSet(
                drawOptionLegend = cms.string('p'),
                markerSize = cms.double(1.0),
                markerColor = cms.int32(4),
                lineColor = cms.int32(1),
                drawOption = cms.string('ep'),
                lineWidth = cms.int32(1),
                lineStyle = cms.int32(1),
                markerStyle = cms.int32(20)
            ),
            reference = cms.PSet(
                drawOptionLegend = cms.string('l'),
                fillColor = cms.int32(41),
                lineColor = cms.int32(1),
                drawOption = cms.string('eBand'),
                lineWidth = cms.int32(1),
                lineStyle = cms.int32(1)
            )
        )
    ),
    xAxes = cms.PSet(
        phi = cms.PSet(
            xAxisTitleSize = cms.double(0.05),
            xAxisTitle = cms.string('#phi'),
            xAxisTitleOffset = cms.double(0.9)
        ),
        eta = cms.PSet(
            xAxisTitleSize = cms.double(0.05),
            xAxisTitle = cms.string('#eta'),
            xAxisTitleOffset = cms.double(0.9)
        ),
        energy = cms.PSet(
            xAxisTitleSize = cms.double(0.05),
            xAxisTitle = cms.string('E / GeV'),
            xAxisTitleOffset = cms.double(0.9)
        ),
        pt = cms.PSet(
            xAxisTitleSize = cms.double(0.05),
            xAxisTitle = cms.string('P_{T} / GeV'),
            xAxisTitleOffset = cms.double(0.9)
        )
    ),
    yAxes = cms.PSet(
        efficiency = cms.PSet(
            yAxisTitleOffset = cms.double(1.1),
            maxY_log = cms.double(1.8),
            maxY_linear = cms.double(1.6),
            yScale = cms.string('linear'),
            minY_linear = cms.double(0.0),
            yAxisTitle = cms.string('#varepsilon'),
            minY_log = cms.double(0.001),
            yAxisTitleSize = cms.double(0.05)
        ),
        fakeRate = cms.PSet(
            yAxisTitleOffset = cms.double(1.1),
            maxY_log = cms.double(1.8),
            maxY_linear = cms.double(1.6),
            yScale = cms.string('log'),
            minY_linear = cms.double(0.0),
            yAxisTitle = cms.string('#varepsilon'),
            minY_log = cms.double(0.001),
            yAxisTitleSize = cms.double(0.05)
        )
    ),
    drawJobs = cms.PSet(
        TauIdEffStepByStep = cms.PSet(
            title = cms.string('TauId step by step efficiencies'),
            labels = cms.vstring('pt', 
                'eta'),
            yAxis = cms.string('fakeRate'),
            xAxis = cms.string('#PAR#'),
            plots = cms.VPSet(cms.PSet(
                process = cms.string('test'),
                dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/shrinkingConePFTauProducerLeadingPion_shrinkingConePFTauDiscriminationByLeadingPionPtCut/LeadingPionPtCutEff#PAR#'),
                drawOptionEntry = cms.string('eff_overlay01'),
                legendEntry = cms.string('Lead Pion P_{T} Cut')
            ), 
                              cms.PSet(
                process = cms.string('test'),
                dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/shrinkingConePFTauProducerLeadingPion_shrinkingConePFTauDiscriminationByLeadingTrackPtCut/LeadingTrackPtCutEff#PAR#'),
                drawOptionEntry = cms.string('eff_overlay02'),
                legendEntry = cms.string('Lead Track P_{T} Cut')
            ), 
                cms.PSet(
                    process = cms.string('test'),
                    dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/shrinkingConePFTauProducerLeadingPion_shrinkingConePFTauDiscriminationByTrackIsolation/TrackIsolationEff#PAR#'),
                    drawOptionEntry = cms.string('eff_overlay03'),
                    legendEntry = cms.string('Track Iso. Using Lead. Pion')
                ), 
                cms.PSet(
                    process = cms.string('test'),
                    dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/shrinkingConePFTauProducerLeadingPion_shrinkingConePFTauDiscriminationByECALIsolation/ECALIsolationEff#PAR#'),
                    drawOptionEntry = cms.string('eff_overlay04'),
                    legendEntry = cms.string('Track + Gamma Iso. Using Lead. Pion')
                )),
            parameter = cms.vstring('pt', 
                'eta', 
                'phi', 
                'energy'),
            legend = cms.string('efficiency_overlay')
        )
    ),
    canvasSizeX = cms.int32(640),
    canvasSizeY = cms.int32(640),
    outputFilePath = cms.string('./'),
    drawOptionEntries = cms.PSet(
        eff_overlay05 = cms.PSet(
            drawOptionLegend = cms.string('p'),
            markerSize = cms.double(1.0),
            markerColor = cms.int32(6),
            lineColor = cms.int32(6),
            drawOption = cms.string('ex0'),
            lineWidth = cms.int32(2),
            lineStyle = cms.int32(1),
            markerStyle = cms.int32(20)
        ),
        eff_overlay04 = cms.PSet(
            drawOptionLegend = cms.string('p'),
            markerSize = cms.double(1.0),
            markerColor = cms.int32(4),
            lineColor = cms.int32(4),
            drawOption = cms.string('ex0'),
            lineWidth = cms.int32(2),
            lineStyle = cms.int32(1),
            markerStyle = cms.int32(20)
        ),
        eff_overlay06 = cms.PSet(
            drawOptionLegend = cms.string('p'),
            markerSize = cms.double(1.0),
            markerColor = cms.int32(5),
            lineColor = cms.int32(5),
            drawOption = cms.string('ex0'),
            lineWidth = cms.int32(2),
            lineStyle = cms.int32(1),
            markerStyle = cms.int32(20)
        ),
        eff_overlay01 = cms.PSet(
            drawOptionLegend = cms.string('p'),
            markerSize = cms.double(1.0),
            markerColor = cms.int32(1),
            lineColor = cms.int32(1),
            drawOption = cms.string('ex0'),
            lineWidth = cms.int32(2),
            lineStyle = cms.int32(1),
            markerStyle = cms.int32(20)
        ),
        eff_overlay03 = cms.PSet(
            drawOptionLegend = cms.string('p'),
            markerSize = cms.double(1.0),
            markerColor = cms.int32(3),
            lineColor = cms.int32(3),
            drawOption = cms.string('ex0'),
            lineWidth = cms.int32(2),
            lineStyle = cms.int32(1),
            markerStyle = cms.int32(20)
        ),
        eff_overlay02 = cms.PSet(
            drawOptionLegend = cms.string('p'),
            markerSize = cms.double(1.0),
            markerColor = cms.int32(2),
            lineColor = cms.int32(2),
            drawOption = cms.string('ex0'),
            lineWidth = cms.int32(2),
            lineStyle = cms.int32(1),
            markerStyle = cms.int32(20)
        )
    )
)


process.plotTauValidation = cms.Sequence(process.plotPFTauHighEfficiencyEfficienciesLeadingPion)


process.loadAndPlotTauValidation = cms.Sequence(process.loadTau)


process.p = cms.Path(process.loadTau*process.plotTauValidation)


process.DQMStore = cms.Service("DQMStore")


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(0)
)



