import FWCore.ParameterSet.Config as cms

TopAnalyzer = cms.EDAnalyzer("BoostedTopAnalyzer",
                             debug = cms.bool( False ),
                             leptonFlavor = cms.int32( 13 ),
                             genEventSource = cms.InputTag('genEvt'),
                             leptonSource   = cms.InputTag('selectedLayer1Muons'),
                             metSource      = cms.InputTag('selectedLayer1METs'),
                             jetSource      = cms.InputTag('selectedLayer1Jets'),
                             EvtSolution    = cms.InputTag('solutions::TEST'),
                             rootFilename = cms.string('TopAnalysis.root'),
                             jetCuts = cms.PSet(
                                       MinLeadingJetEt = cms.double( 65. ),
                                       MinJetEt        = cms.double( 2.1 ),
                                       MinJetEta       = cms.double( 2.4)
                                       ),
                             leptonCuts = cms.PSet(
                                       MinLeptonPt  = cms.double( 30. ),
                                       MinLeptonEta = cms.double( 2.1 ),
                                       TrackIso     = cms.double( 3.0 ),
                                       CaloIso      = cms.double( 1.0 )
                                       ),
                             METCuts = cms. PSet(
                                       MinMET = cms.double( 0. )
                                       ),
                             writeAscii = cms.bool( False),
                             asciiFilename = cms.string('TopAnalysis.txt'),
                             processOnlyEvent = cms.int32( -1 ),
                             makeJetLegoPlots = cms.bool( False ),
                             )

