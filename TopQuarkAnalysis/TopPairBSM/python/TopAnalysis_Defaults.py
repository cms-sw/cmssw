import FWCore.ParameterSet.Config as cms

TopAnalyzer = cms.EDAnalyzer("BooLowMAnalyzer",
                             debug = cms.bool( False ),
                             IsMCTop = cms.bool( True ),
                             genEventSource = cms.InputTag('genEvt'),
                             muonSource    = cms.InputTag('selectedLayer1Muons'),
                             electronSource = cms.InputTag('selectedLayer1Electrons'),
                             metSource      = cms.InputTag('layer1METs'),
                             jetSource      = cms.InputTag('selectedLayer1Jets'),
                             rootFilename = cms.string('TopAnalysis.root'),
                             PdfInfoTag = cms.untracked.InputTag("genEventPdfInfo"),
                             PdfSetName = cms.untracked.string("cteq66"), # Hard coded to use LHgrid
                             jetCuts = cms.PSet(
                                       MinJetPt        = cms.double( 30. ),
                                       MaxJetEta       = cms.double( 2.4),
                                       ApplyAsymmetricCuts = cms.bool(False),
                                       JES             = cms.double( 1. ),
                                       ApplyFlavorJEC = cms.bool(False)
                                       ),
                             muonCuts = cms.PSet(
                                       MinPt  = cms.double( 20. ),
                                       MaxEta = cms.double( 2.1 ),
                                       ),
                             muonIsolation = cms.PSet(
                                       RelIso = cms.double( 0.95 ),
                                       MaxVetoEm = cms.double( 4.0 ),
                                       MaxVetoHad = cms.double( 6.0 )
                                       ),
                             electronCuts = cms.PSet(
                                       MinPt  = cms.double( 20. ),
                                       MaxEta = cms.double( 2.4 ),
                                       RelIso = cms.double( 0.9 )
                                       ),
                             METCuts = cms. PSet(
                                       MinMET = cms.double( 0. ),
                                       Recalculate = cms.bool(False)
                                       ),
                             UsebTagging = cms.bool(False),
                             UseMtopConstraint = cms.bool(True),
                             writeAscii = cms.bool( False),
                             asciiFilename = cms.string('TopAnalysis.txt'),
                             processOnlyEvent = cms.int32( -1 ),
                             makeJetLegoPlots = cms.bool( False ),
                             )

