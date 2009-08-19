import FWCore.ParameterSet.Config as cms

import Validation.Generator.BasicGenTest_cfi

basicGenTestC=Validation.Generator.BasicGenTest_cfi.basicGenTest.clone()

basicGenTest_seq = cms.Sequence(basicGenTestC)
