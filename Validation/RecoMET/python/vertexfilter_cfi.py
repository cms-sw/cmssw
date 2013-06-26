
import FWCore.ParameterSet.Config as cms

vertexfilter = cms.EDFilter(
							"NvertexFilter",
                      minNvtx = cms.double(0),   
                      maxNvtx = cms.double(9999)
 )

