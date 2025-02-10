import FWCore.ParameterSet.Config as cms

offlinePrimaryVertices = cms.EDProducer(
    "PrimaryVertexProducer",

    verbose = cms.untracked.bool(False),
    TrackLabel = cms.InputTag("generalTracks"),
    beamSpotLabel = cms.InputTag("offlineBeamSpot"),

    TkFilterParameters = cms.PSet(
        algorithm=cms.string('filter'),
        maxNormalizedChi2 = cms.double(10.0),
        minPixelLayersWithHits=cms.int32(2),
        minSiliconLayersWithHits = cms.int32(5),
        maxD0Significance = cms.double(4.0),
        maxD0Error = cms.double(1.0),
        maxDzError = cms.double(1.0),
        minPt = cms.double(0.0),
        maxEta = cms.double(2.4),
        trackQuality = cms.string("any")
    ),

    TkClusParameters = cms.PSet(
        algorithm   = cms.string("DA_vect"),
        TkDAClusParameters = cms.PSet(
            coolingFactor = cms.double(0.6),  # moderate annealing speed
            zrange = cms.double(4.),          # consider only clusters within 4 sigma*sqrt(T) of a track
            delta_highT = cms.double(1.e-2),  # convergence requirement at high T
            delta_lowT = cms.double(1.e-3),   # convergence requirement at low T
            convergence_mode = cms.int32(0),  # 0 = two steps, 1 = dynamic with sqrt(T)
            Tmin = cms.double(2.0),           # end of vertex splitting
            Tpurge = cms.double(2.0),         # cleaning
            Tstop = cms.double(0.5),          # end of annealing
            vertexSize = cms.double(0.006),   # added in quadrature to track-z resolutions
            d0CutOff = cms.double(3.),        # downweight high IP tracks
            dzCutOff = cms.double(3.),        # outlier rejection after freeze-out (T<Tmin)
            zmerge = cms.double(1e-2),        # merge intermediat clusters separated by less than zmerge
            uniquetrkweight = cms.double(0.8),# require at least two tracks with this weight at T=Tpurge
            uniquetrkminp = cms.double(0.0),  # minimal a priori track weight for counting unique tracks
            runInBlocks = cms.bool(False),    # activate the DA running in blocks of z sorted tracks
            block_size = cms.uint32(10000),   # block size in tracks
            overlap_frac = cms.double(0.0)    # overlap between consecutive blocks (blocks_size*overlap_frac)
        )
    ),

    vertexCollections = cms.VPSet(
     [cms.PSet(label=cms.string(""),
               algorithm=cms.string("AdaptiveVertexFitter"),
               chi2cutoff = cms.double(2.5),
               minNdof=cms.double(0.0),
               useBeamConstraint = cms.bool(False),
               maxDistanceToBeam = cms.double(1.0)
               ),
      cms.PSet(label=cms.string("WithBS"),
               algorithm = cms.string('AdaptiveVertexFitter'),
               chi2cutoff = cms.double(2.5),
               minNdof=cms.double(2.0),
               useBeamConstraint = cms.bool(True),
               maxDistanceToBeam = cms.double(1.0),
               )
      ]
    ),

    isRecoveryIteration = cms.bool(False),
    recoveryVtxCollection = cms.InputTag("")


)

from Configuration.ProcessModifiers.vertexInBlocks_cff import vertexInBlocks
vertexInBlocks.toModify(offlinePrimaryVertices,
    TkClusParameters = dict(
        TkDAClusParameters = dict(
            runInBlocks = True,
            block_size = 128,
            overlap_frac = 0.5
        )
    )
)

from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
(phase2_tracker & vertexInBlocks).toModify(offlinePrimaryVertices,
     TkClusParameters = dict(
         TkDAClusParameters = dict(
             block_size = 512,
             overlap_frac = 0.5)
         )
)

from Configuration.Eras.Modifier_highBetaStar_cff import highBetaStar
highBetaStar.toModify(offlinePrimaryVertices,
    TkClusParameters = dict(
        TkDAClusParameters = dict(
            Tmin = 4.0,
            Tpurge = 1.0,
            Tstop = 1.0,
            vertexSize = 0.01,
            d0CutOff = 4.,
            dzCutOff = 5.,
            zmerge = 2.e-2,
            uniquetrkweight = 0.9
       )
    )
)

DA_vectParameters = cms.PSet(offlinePrimaryVertices.TkClusParameters.clone())

from Configuration.ProcessModifiers.weightedVertexing_cff import weightedVertexing
weightedVertexing.toModify(offlinePrimaryVertices,
                           vertexCollections = cms.VPSet(
                           [cms.PSet(label=cms.string(""),
                                     algorithm=cms.string("WeightedMeanFitter"),
                                     chi2cutoff = cms.double(2.5),
                                     minNdof=cms.double(0.0),
                                     useBeamConstraint = cms.bool(False),
                                     maxDistanceToBeam = cms.double(1.0)
                           ),
                           cms.PSet(label=cms.string("WithBS"),
                                     algorithm = cms.string('WeightedMeanFitter'),
                                     minNdof=cms.double(0.0),
                                     chi2cutoff = cms.double(2.5),
                                     useBeamConstraint = cms.bool(True),
                                     maxDistanceToBeam = cms.double(1.0)
                                     )
                           ]
                           ))
# This customization is needed in the trackingLowPU era to be able to
# produce vertices also in the cases in which the pixel detector is
# not included in data-taking, like it was the case for "Quiet Beam"
# collisions on 2016 with run 269207.

from Configuration.Eras.Modifier_trackingLowPU_cff import trackingLowPU
trackingLowPU.toModify(offlinePrimaryVertices,
                            TkFilterParameters = dict(minPixelLayersWithHits = 0))


# higher eta cut for the phase 2 tracker
from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
phase2_tracker.toModify(offlinePrimaryVertices,
                        TkFilterParameters = dict(maxEta = 4.0))

from Configuration.Eras.Modifier_pp_on_XeXe_2017_cff import pp_on_XeXe_2017
from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
(pp_on_XeXe_2017 | pp_on_AA).toModify(offlinePrimaryVertices,
               TkFilterParameters = dict(
                   algorithm="filterWithThreshold",
                   maxD0Significance = 2.0,
                   maxD0Error = 10.0,
                   maxDzError = 10.0,
                   minPixelLayersWithHits=3,
                   minPt = 0.7,
                   trackQuality = "highPurity",
                   numTracksThreshold = cms.int32(10),
                   maxNumTracksThreshold = cms.int32(1000),
                   minPtTight = cms.double(1.0)
               ),
               TkClusParameters = cms.PSet(
                 algorithm = cms.string("gap"),
                 TkGapClusParameters = cms.PSet(
                   zSeparation = cms.double(1.0)
                 )
               )
)

highBetaStar.toModify(offlinePrimaryVertices,
     TkFilterParameters = dict(
         maxNormalizedChi2 = 80.0,
         minPixelLayersWithHits = 1,
         minSiliconLayersWithHits = 3,
         maxD0Significance = 7.0,
         maxD0Error = 10.0,
         maxDzError = 10.0,
         maxEta = 2.5
     ),
     vertexCollections = {
         0: dict(chi2cutoff = 4.0, minNdof = -1.1),
         1: dict(chi2cutoff = 4.0, minNdof = -2.0),
     }
)

from Configuration.Eras.Modifier_run3_upc_cff import run3_upc
run3_upc.toModify(offlinePrimaryVertices,
    TkFilterParameters = dict(
        algorithm="filterWithThreshold",
        maxNormalizedChi2 = 80.0,
        minPixelLayersWithHits = 1,
        minSiliconLayersWithHits = 3,
        maxD0Significance = 4.0,
        maxD0Error = 10.0,
        maxDzError = 10.0,
        minPt = 0.0,
        maxEta = 3.0,
        trackQuality = "highPurity",
        numTracksThreshold = cms.int32(3),
        maxNumTracksThreshold = cms.int32(1000),
        minPtTight = cms.double(1.0)
    ),
    TkClusParameters = cms.PSet(
        algorithm = cms.string("gap"),
        TkGapClusParameters = cms.PSet(
            zSeparation = cms.double(6.0)
        )
    ),
    vertexCollections = {
        0: dict(chi2cutoff = 4.0, minNdof = -1.1),
        1: dict(chi2cutoff = 4.0, minNdof = -2.0),
    }
)
