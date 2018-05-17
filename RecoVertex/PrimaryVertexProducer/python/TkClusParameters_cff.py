import FWCore.ParameterSet.Config as cms

DA_vectParameters = cms.PSet(
    algorithm   = cms.string("DA_vect"),
    TkDAClusParameters = cms.PSet(
        coolingFactor = cms.double(0.6),  # moderate annealing speed
        Tmin = cms.double(2.0),           # end of vertex splitting
        Tpurge = cms.double(2.0),         # cleaning 
        Tstop = cms.double(0.5),          # end of annealing 
        vertexSize = cms.double(0.006),   # added in quadrature to track-z resolutions        
        d0CutOff = cms.double(3.),        # downweight high IP tracks 
        dzCutOff = cms.double(3.),        # outlier rejection after freeze-out (T<Tmin)       
        zmerge = cms.double(1e-2),        # merge intermediat clusters separated by less than zmerge
        uniquetrkweight = cms.double(0.8) # require at least two tracks with this weight at T=Tpurge
        )
)

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
        Tmin = cms.double(4.0),           # end of vertex splitting
        Tpurge = cms.double(4.0),         # cleaning 
        Tstop = cms.double(2.0),          # end of annealing 
        vertexSize = cms.double(0.006),   # added in quadrature to track-z resolutions
        vertexSizeTime = cms.double(0.008),
        d0CutOff = cms.double(3.),        # downweight high IP tracks 
        dzCutOff = cms.double(3.),        # outlier rejection after freeze-out (T<Tmin)
        dtCutOff = cms.double(4.),        # outlier rejection after freeze-out (T<Tmin)
        zmerge = cms.double(1e-2),        # merge intermediat clusters separated by less than zmerge and tmerge
        tmerge = cms.double(1e-1),        # merge intermediat clusters separated by less than zmerge and tmerge
        uniquetrkweight = cms.double(0.8) # require at least two tracks with this weight at T=Tpurge
        )
)
