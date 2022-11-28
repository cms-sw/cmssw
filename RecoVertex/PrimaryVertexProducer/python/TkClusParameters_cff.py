import FWCore.ParameterSet.Config as cms

DA_vectParameters = cms.PSet(
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
)

from Configuration.ProcessModifiers.vertexInBlocks_cff import vertexInBlocks
vertexInBlocks.toModify(DA_vectParameters,
    TkDAClusParameters = dict(
    runInBlocks = True,
    block_size = 128,
    overlap_frac = 0.5
    )
)

from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
(phase2_tracker & vertexInBlocks).toModify(DA_vectParameters,
        TkDAClusParameters = dict(
        block_size = 512,
        overlap_frac = 0.5))

from Configuration.Eras.Modifier_highBetaStar_2018_cff import highBetaStar_2018
highBetaStar_2018.toModify(DA_vectParameters,
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

DA2D_vectParameters = cms.PSet(
    algorithm   = cms.string("DA2D_vect"),
    TkDAClusParameters = cms.PSet(
        coolingFactor = cms.double(0.6),  # moderate annealing speed
        zrange = cms.double(4.),          # consider only clusters within 4 sigma*sqrt(T) of a track
        delta_highT = cms.double(1.e-2),  # convergence requirement at high T
        delta_lowT = cms.double(1.e-3),   # convergence requirement at low T
        convergence_mode = cms.int32(0),  # 0 = two steps, 1 = dynamic with sqrt(T)
        Tmin = cms.double(4.0),           # end of vertex splitting
        Tpurge = cms.double(4.0),         # cleaning 
        Tstop = cms.double(2.0),          # end of annealing 
        vertexSize = cms.double(0.006),   # added in quadrature to track-z resolutions
        vertexSizeTime = cms.double(0.008),
        d0CutOff = cms.double(3.),        # downweight high IP tracks 
        dzCutOff = cms.double(3.),        # outlier rejection after freeze-out (T<Tmin)
        dtCutOff = cms.double(4.),        # outlier rejection after freeze-out (T<Tmin)
        t0Max = cms.double(1.0),          # outlier rejection for use of timing information
        zmerge = cms.double(1e-2),        # merge intermediat clusters separated by less than zmerge and tmerge
        tmerge = cms.double(1e-1),        # merge intermediat clusters separated by less than zmerge and tmerge
        uniquetrkweight = cms.double(0.8),# require at least two tracks with this weight at T=Tpurge
        uniquetrkminp = cms.double(0.0)   # minimal a priori track weight for counting unique tracks
        )
)
