import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.tools.cmsswVersionTools import pickRelValInputFiles
relValTTbar = pickRelValInputFiles(cmsswVersion  = 'CMSSW_4_2_0_pre8',
                                   relVal        = 'RelValTTbar',
                                   globalTag     = 'MC_42_V7'
                                   )

