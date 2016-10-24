import FWCore.ParameterSet.Config as cms

DA2DParameters = cms.PSet(
    algorithm   = cms.string("DA2D"),
    TkDAClusParameters = cms.PSet(
        coolingFactor = cms.double(0.6),  #  moderate annealing speed
        Tmin = cms.double(4.),            #  end of annealing
        vertexSize = cms.double(0.01),    #  ~ resolution / sqrt(Tmin)
        d0CutOff = cms.double(3.),        # downweight high IP tracks 
        dzCutOff = cms.double(4.)         # outlier rejection after freeze-out (T<Tmin)
        )
)

DA_vectParameters = cms.PSet(
    algorithm   = cms.string("DA_vect"),
    TkDAClusParameters = cms.PSet(
        coolingFactor = cms.double(0.6),  #  moderate annealing speed
        Tmin = cms.double(4.),            #  end of annealing
        vertexSize = cms.double(0.01),    #  ~ resolution / sqrt(Tmin)
        d0CutOff = cms.double(3.),        # downweight high IP tracks 
        dzCutOff = cms.double(4.)         # outlier rejection after freeze-out (T<Tmin)
        )
)
