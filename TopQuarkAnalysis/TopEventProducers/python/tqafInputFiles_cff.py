import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.tools.cmsswVersionTools import pickRelValInputFiles
relValTTbar = pickRelValInputFiles(cmsswVersion  = 'CMSSW_5_0_0',
                                   relVal        = 'RelValProdTTbar',
                                   globalTag     = 'START50_V8',
                                   dataTier      = 'AODSIM',
                                   maxVersions   = 1
                                   )

