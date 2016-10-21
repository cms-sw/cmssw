import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing

options = VarParsing()
options.register ("pixels"      , 27370,  VarParsing.multiplicity.singleton, VarParsing.varType.int)
options.register ("npeMin"      , 1000,   VarParsing.multiplicity.singleton, VarParsing.varType.int)
options.register ("npeMax"      , 57000,  VarParsing.multiplicity.singleton, VarParsing.varType.int)
options.register ("npeStep"     , 50,     VarParsing.multiplicity.singleton, VarParsing.varType.int)
options.register ("nReps"       , 5,      VarParsing.multiplicity.singleton, VarParsing.varType.int)
options.register ("nBins"       , 1200,   VarParsing.multiplicity.singleton, VarParsing.varType.int)
options.register ("binMin"      , 0,      VarParsing.multiplicity.singleton, VarParsing.varType.int)
options.register ("binMax"      , 60000,  VarParsing.multiplicity.singleton, VarParsing.varType.int)
options.register ("tau"         , 5.,     VarParsing.multiplicity.singleton, VarParsing.varType.float)
options.register ("dt"          , 0.5,    VarParsing.multiplicity.singleton, VarParsing.varType.float)
options.register ("nPreciseBins", 500,    VarParsing.multiplicity.singleton, VarParsing.varType.int)
options.register ("fitname"     , "pol2", VarParsing.multiplicity.singleton, VarParsing.varType.string)
options.parseArguments()

process = cms.Process("demo")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )
process.source = cms.Source("EmptySource")

process.load("Configuration.StandardSequences.Services_cff")
process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    ana = cms.PSet(
        initialSeed = cms.untracked.uint32(12345),
        engineName = cms.untracked.string('HepJamesRandom')
    )
)

process.ana = cms.EDAnalyzer("SiPMNonlinearityAnalyzer",
    pixels       = cms.uint32(options.pixels      ),
    npeMin       = cms.uint32(options.npeMin      ),
    npeMax       = cms.uint32(options.npeMax      ),
    npeStep      = cms.uint32(options.npeStep     ),
    nReps        = cms.uint32(options.nReps       ),
    nBins        = cms.uint32(options.nBins       ),
    binMin       = cms.uint32(options.binMin      ),
    binMax       = cms.uint32(options.binMax      ),
    tau          = cms.double(options.tau         ),
    dt           = cms.double(options.dt          ),
    nPreciseBins = cms.uint32(options.nPreciseBins),
    fitname      = cms.string(options.fitname     ),
)

process.TFileService = cms.Service("TFileService",
    fileName = cms.string("nonlin_pixels"+str(options.pixels)+"_npeMin"+str(options.npeMin)+"_npeMax"+str(options.npeMax)+"_npeStep"+str(options.npeStep)+"_nReps"+str(options.nReps)+"_tau"+str(options.tau)+"_"+options.fitname+".root")
)

process.p1 = cms.Path(process.ana)
