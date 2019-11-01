#run with: cmsRun hgcsiNoiseMapTester_cfg.py doseMap=SimCalorimetry/HGCalSimProducers/data/doseParams_3000fb.txt

import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing
from Configuration.StandardSequences.Eras import eras

options = VarParsing()
options.register ("doseMap", "",  VarParsing.multiplicity.singleton, VarParsing.varType.string)
options.parseArguments()

process = cms.Process("demo",eras.Phase2C8)

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load('Configuration.Geometry.GeometryExtended2026D41Reco_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic', '')

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )
process.source = cms.Source("EmptySource")

from SimCalorimetry.HGCalSimProducers.hgcalDigitizer_cfi import HGCAL_ileakParam_toUse, HGCAL_cceParams_toUse
process.plotter_eol = cms.EDAnalyzer("HGCSiNoiseMapAnalyzer",
                                     doseMap            = cms.string( options.doseMap ),
                                     doseMapAlgo        = cms.uint32(0),
                                     ileakParam         = HGCAL_ileakParam_toUse,
                                     cceParams          = HGCAL_cceParams_toUse,
                                     aimMIPtoADC        = cms.int32(10),
                                     ignoreGainSettings = cms.bool(False)
                                 )

process.plotter_eol_nogain = process.plotter_eol.clone( ignoreGainSettings = cms.bool(True) )

process.plotter_start = process.plotter_eol.clone( doseMapAlgo=cms.uint32(3) )


process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string("dosemap_output.root")
)

process.p = cms.Path(process.plotter_eol
                     *process.plotter_eol_nogain
                     *process.plotter_start)
