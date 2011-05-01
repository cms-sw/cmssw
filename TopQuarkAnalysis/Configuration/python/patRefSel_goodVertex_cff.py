import FWCore.ParameterSet.Config as cms

from DPGAnalysis.Skims.goodvertexSkim_cff import primaryVertexFilter
goodVertex = primaryVertexFilter.clone( vertexCollection = cms.InputTag( 'offlinePrimaryVertices' )
                                      , minimumNDOF      =  4
                                      , maxAbsZ          = 24.
                                      , maxd0            =  2.
                                      )
