import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.tools.cmsswVersionTools import pickRelValInputFiles
relValTTbar = pickRelValInputFiles(cmsswVersion  = 'CMSSW_4_4_0',
                                   relVal        = 'RelValTTbar',
                                   globalTag     = 'START44_V5'
                                   )

