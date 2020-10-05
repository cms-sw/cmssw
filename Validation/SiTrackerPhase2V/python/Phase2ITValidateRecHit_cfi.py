import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
rechitValidIT = DQMEDAnalyzer('Phase2ITValidateRecHit',
                            Verbosity = cms.bool(False),
                            TopFolderName = cms.string("TrackerPhase2ITRecHitV"),
                            ITPlotFillingFlag = cms.bool(False),
                            rechitsSrc = cms.InputTag("siPixelRecHits"),
                            InnerPixelDigiSource   = cms.InputTag("simSiPixelDigis","Pixel"),                          
                            InnerPixelDigiSimLinkSource = cms.InputTag("simSiPixelDigis", "Pixel"), 
                            PSimHitSource  = cms.VInputTag('g4SimHits:TrackerHitsPixelBarrelLowTof',
                                                           'g4SimHits:TrackerHitsPixelBarrelHighTof',
                                                           'g4SimHits:TrackerHitsPixelEndcapLowTof',
                                                           'g4SimHits:TrackerHitsPixelEndcapHighTof',
                                                           'g4SimHits:TrackerHitsTIBLowTof',
                                                           'g4SimHits:TrackerHitsTIBHighTof',
                                                           'g4SimHits:TrackerHitsTIDLowTof',
                                                           'g4SimHits:TrackerHitsTIDHighTof',
                                                           'g4SimHits:TrackerHitsTOBLowTof',
                                                           'g4SimHits:TrackerHitsTOBHighTof',
                                                           'g4SimHits:TrackerHitsTECLowTof',
                                                           'g4SimHits:TrackerHitsTECHighTof'),
                              simTracksSrc = cms.InputTag("g4SimHits"),
                              SimTrackMinPt = cms.double(2.),
                              SimTrackSource = cms.InputTag("g4SimHits"),
                              SimVertexSource = cms.InputTag("g4SimHits"),
                              usePhase2Tracker = cms.bool(True),#these are used by simHit assoc.
                              associatePixel = cms.bool(True),
                              associateRecoTracks = cms.bool(False),
                              associateStrip = cms.bool(False),
                              associateHitbySimTrack = cms.bool(True),
                              pixelSimLinkSrc = cms.InputTag("simSiPixelDigis","Pixel"),
                              ROUList  =  cms.vstring('g4SimHitsTrackerHitsPixelBarrelLowTof',
                                                      'g4SimHitsTrackerHitsPixelBarrelHighTof',
                                                      'g4SimHitsTrackerHitsPixelEndcapLowTof',
                                                      'g4SimHitsTrackerHitsPixelEndcapHighTof',
                                                  ),
                              DeltaX = cms.PSet(
                                  NxBins = cms.int32(100),
                                  xmin = cms.double(-0.2),
                                  xmax = cms.double(0.2),
                                  switch = cms.bool(True)
                              ),
                              DeltaY = cms.PSet(
                                  NxBins = cms.int32(100),
                                  xmin = cms.double(-0.2),
                                  xmax = cms.double(0.2),
                                  switch = cms.bool(True)
                              ),
                              PullX = cms.PSet(
                                  NxBins = cms.int32(100),
                                  xmin = cms.double(-4.),
                                  xmax = cms.double(4.),
                                  switch = cms.bool(True)
                              ),
                              PullY = cms.PSet(
                                  NxBins = cms.int32(100),
                                  xmin = cms.double(-4.),
                                  xmax = cms.double(4.),
                                  switch = cms.bool(True)
                              ),
                              DeltaX_eta = cms.PSet(
                                  NxBins = cms.int32(82),
                                  xmin = cms.double(-4.1),
                                  xmax = cms.double(4.1),
                                  ymin = cms.double(-0.02),
                                  ymax = cms.double(0.02),
                                  switch = cms.bool(True)
                              ),
                              DeltaY_eta = cms.PSet(
                                  NxBins = cms.int32(82),
                                  xmin = cms.double(-4.1),
                                  xmax = cms.double(4.1),
                                  ymin = cms.double(-0.02),
                                  ymax = cms.double(0.02),
                                  switch = cms.bool(True)
                              ),
                              PullX_eta = cms.PSet(
                                  NxBins = cms.int32(82),
                                  xmin = cms.double(-4.1),
                                  xmax = cms.double(4.1),
                                  ymin = cms.double(-4.),
                                  ymax = cms.double(4.),
                                  switch = cms.bool(True)
                              ),
                              PullY_eta = cms.PSet(
                                  NxBins = cms.int32(82),
                                  xmin = cms.double(-4.1),
                                  xmax = cms.double(4.1),
                                  ymin = cms.double(-4.),
                                  ymax = cms.double(4.),
                                  switch = cms.bool(True)
                              ),
                              nRecHits_primary = cms.PSet(
                                  NxBins = cms.int32(100),
                                  xmin = cms.double(0.),
                                  xmax = cms.double(0.),
                                  switch = cms.bool(True)
                              ),
                              DeltaX_primary = cms.PSet(
                                  NxBins = cms.int32(100),
                                  xmin = cms.double(-0.2),
                                  xmax = cms.double(0.2),
                                  switch = cms.bool(True)
                              ), 
                              DeltaY_primary = cms.PSet(
                                  NxBins = cms.int32(100),
                                  xmin = cms.double(-0.2),
                                  xmax = cms.double(0.2),
                                  switch = cms.bool(True)
                              ), 
                              PullX_primary = cms.PSet(
                                  NxBins = cms.int32(82),
                                  xmin = cms.double(-4.1),
                                  xmax = cms.double(4.1),
                                  ymin = cms.double(-4.),
                                  ymax = cms.double(4.),
                                  switch = cms.bool(True)
                              ),
                              PullY_primary = cms.PSet(
                                  NxBins = cms.int32(82),
                                  xmin = cms.double(-4.1),
                                  xmax = cms.double(4.1),
                                  ymin = cms.double(-4.),
                                  ymax = cms.double(4.),
                                  switch = cms.bool(True)
                              ) 
) 
