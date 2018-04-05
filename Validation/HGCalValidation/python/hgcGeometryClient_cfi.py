import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

hgcalGeometryClient = DQMEDHarvester("HGCalGeometryClient", 
                                     DirectoryName = cms.string("Geometry"),
                                     )
