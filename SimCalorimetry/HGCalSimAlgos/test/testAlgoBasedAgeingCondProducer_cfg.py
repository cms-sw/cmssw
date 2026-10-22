import FWCore.ParameterSet.Config as cms
process = cms.Process("TEST")

from FWCore.ParameterSet.VarParsing import VarParsing
options = VarParsing('python')
options.register('verbosity',0,mytype=VarParsing.varType.int,
                 info='Tester verbosity: 0 = Base+FED prints / 1 = +Module prints / 2 = +Cell prints')
options.register('modules','Geometry/HGCalMapping/data/ModuleMaps/modulelocator_test.txt',mytype=VarParsing.varType.string,
                 info="Path to module mapper. Absolute, or relative to CMSSW src directory")
options.register('sicells','Geometry/HGCalMapping/data/CellMaps/WaferCellMapTraces.txt',mytype=VarParsing.varType.string,
                 info="Path to Si cell mapper. Absolute, or relative to CMSSW src directory")
options.register('sipmcells','Geometry/HGCalMapping/data/CellMaps/channels_sipmontile.hgcal.txt',mytype=VarParsing.varType.string,
                 info="Path to SiPM-on-tile cell mapper. Absolute, or relative to CMSSW src directory")
options.register('offsetfile','Geometry/HGCalMapping/data/CellMaps/calibration_to_surrounding_offsetMap.txt',mytype=VarParsing.varType.string,
                 info="Path to calibration-to-surrounding cell offset file. Absolute, or relative to CMSSW src directory")
options.parseArguments()

process.source = cms.Source('EmptySource')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

# electronics mapping
from Geometry.HGCalMapping.hgcalmapping_cff import customise_hgcalmapper
kwargs = { k: getattr(options,k) for k in ['modules','sicells','sipmcells','offsetfile'] if getattr(options,k)!='' }
process = customise_hgcalmapper(process, **kwargs)

# ageing conditions
from SimCalorimetry.HGCalSimAlgos.hgcal_ageing_byalgo_cfi import customise_hgcal_ageing_byalgo_startup
process = customise_hgcal_ageing_byalgo_startup(process)

# Geometry
process.load('Configuration.Geometry.GeometryExtendedRun4D122Reco_cff')

# tester
process.load('SimCalorimetry.HGCalSimAlgos.hgCalAgeingConditionTester_cfi')

# test path
process.p = cms.Path(process.hgCalAgeingConditionTester)
