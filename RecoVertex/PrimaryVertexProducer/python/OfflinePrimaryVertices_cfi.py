import FWCore.ParameterSet.Config as cms

from RecoVertex.PrimaryVertexProducer.primaryVertexProducer_cfi import primaryVertexProducer

offlinePrimaryVertices = primaryVertexProducer.clone()

DA_vectParameters = cms.PSet(primaryVertexProducer.TkClusParameters.clone())

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

from Configuration.Eras.Modifier_highBetaStar_2018_cff import highBetaStar_2018
highBetaStar_2018.toModify(offlinePrimaryVertices,
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
               TkClusParameters = dict(
                 algorithm = "gap"
               )
)
    
from Configuration.Eras.Modifier_highBetaStar_2018_cff import highBetaStar_2018
highBetaStar_2018.toModify(offlinePrimaryVertices,
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
