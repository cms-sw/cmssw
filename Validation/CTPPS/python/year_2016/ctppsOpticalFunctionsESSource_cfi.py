import FWCore.ParameterSet.Config as cms
from CondFormats.CTPPSReadoutObjects.ctppsOpticalFunctionsESSource_cfi import ctppsOpticalFunctionsESSource as _tmp

ctppsOpticalFunctionsESSource = _tmp.clone(
    xangle1 = cms.double(185),
    xangle2 = cms.double(185),
    fileName1 = cms.FileInPath("RecoCTPPS/TotemRPLocal/data/optical_functions_2016.root"),
    fileName2 = cms.FileInPath("RecoCTPPS/TotemRPLocal/data/optical_functions_2016.root"),
    scoringPlanes = cms.VPSet(
        # z in cm
        cms.PSet( rpId = cms.uint32(0x76100000), dirName = cms.string("XRPH_C6L5_B2"), z = cms.double(-20382.6) ),  # RP 002
        cms.PSet( rpId = cms.uint32(0x76180000), dirName = cms.string("XRPH_D6L5_B2"), z = cms.double(-21255.1) ),  # RP 003
        cms.PSet( rpId = cms.uint32(0x77100000), dirName = cms.string("XRPH_C6R5_B1"), z = cms.double(+20382.6) ),  # RP 102
        cms.PSet( rpId = cms.uint32(0x77180000), dirName = cms.string("XRPH_D6R5_B1"), z = cms.double(+21255.1) ),  # RP 103
    )
)
