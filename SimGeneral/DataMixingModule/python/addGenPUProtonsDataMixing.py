import FWCore.ParameterSet.Config as cms

def customiseGenPUProtonsDataMixing(process):

    process.mixData.GenPUProtonsInputTags = cms.VInputTag("genPUProtonsBx0", "genPUProtonsBxm1", "genPUProtonsBxp1")
    return process
