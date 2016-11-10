# In order to produce everything that you need in one go, use the command:
#
# for t in {'BeamPipe','Tracker','PixBar','PixFwdMinus','PixFwdPlus','TIB','TOB','TIDB','TIDF','TEC','TkStrct','InnerServices'}; do cmsRun runP_Tracker_cfg.py geom=run2 label=$t >& /dev/null &; done


import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing

process = cms.Process("PROD")

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

# The default geometry is PhaseI. If the run2 geoemtry is needed, the
# appropriate flag has to be passed at command line, e.g.:
# cmsRun runP_GenericComponent.py geom="XYZ"

# The default component to be monitored is the Tracker. If other components
# need to be studied, they must be supplied, one at a time, at the command
# line, e.g.:
# cmsRun runP_GenericComponent.py comp="XYZ"

_LABELS2COMPS = {'BeamPipe': 'BEAM',
                 'Tracker': 'Tracker',
                 'PixBar':  'PixelBarrel',
                 'PixFwd':  ['PixelForwardZplus', 'PixelForwardZminus'],
                 'PixFwdMinus': 'PixelForwardZminus',
                 'PixFwdPlus':  'PixelForwardZplus',
                 'TIB':         'TIB',
                 'TOB':         'TOB',
                 'TIDB':        'TIDB',
                 'TIDF':        'TIDF',
                 'TEC':         'TEC',
                 'InnerServices': ['TIBTIDServicesF', 'TIBTIDServicesB'],
                 'TkStrct': ['TrackerOuterCylinder', 'TrackerBulkhead'],
                 'Phase1PixelBarrel': 'Phase1PixelBarrel',
                 'Phase2OTBarrel': 'Phase2OTBarrel',
                 'Phase2PixelEndcap': 'Phase2PixelEndcap',
                 'Phase2OTForward': 'Phase2OTForward'}

_ALLOWED_LABELS = _LABELS2COMPS.keys()

options = VarParsing('analysis')
options.register('geom',        #name
                 'phaseI',      #default value
                 VarParsing.multiplicity.singleton,   # kind of options
                 VarParsing.varType.string,           # type of option
                 "Select the geometry to be studied"  # help message
                )

options.register('label',         #name
                 'Tracker',              #default value
                 VarParsing.multiplicity.singleton,   # kind of options
                 VarParsing.varType.string,           # type of option
                 "Select the label to be used to create output files. Default to tracker. If multiple components are selected, it defaults to the join of all components, with '_' as separator."  # help message
                )

options.setDefault('inputFiles', ['file:single_neutrino_random.root'])

options.parseArguments()
# Option validation

if options.label not in _ALLOWED_LABELS:
    print "\n*** Error, '%s' not registered as a valid components to monitor." % options.label
    print "Allowed components:", _ALLOWED_LABELS
    print
    raise RuntimeError("Unknown label")

_components = _LABELS2COMPS[options.label]
#
#Geometry
#
if options.geom == 'phaseI':
  process.load("Configuration.Geometry.GeometryExtended2017_cff")
elif options.geom == 'run2':
  process.load("Configuration.Geometry.GeometryExtended2016_cff")
elif options.geom == 'phaseIID4':
  process.load("Configuration.Geometry.GeometryExtended2023D4_cff")
else:
  print("Unknown geometry %s" % options.geom)
  sys.exit(1)

#
#Magnetic Field
#
process.load("Configuration.StandardSequences.MagneticField_38T_cff")

# Output of events, etc...
#
# Explicit note : since some histos/tree might be dumped directly,
#                 better NOT use PoolOutputModule !
# Detector simulation (Geant4-based)
#
process.load("SimG4Core.Application.g4SimHits_cfi")

process.load("IOMC.RandomEngine.IOMC_cff")
process.RandomNumberGeneratorService.g4SimHits.initialSeed = 9876

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(options.inputFiles)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.p1 = cms.Path(process.g4SimHits)
process.g4SimHits.StackingAction.TrackNeutrino = cms.bool(True)
process.g4SimHits.UseMagneticField = False
process.g4SimHits.Physics.type = 'SimG4Core/Physics/DummyPhysics'
process.g4SimHits.Physics.DummyEMPhysics = True
process.g4SimHits.Physics.CutsPerRegion = False
process.g4SimHits.Watchers = cms.VPSet(cms.PSet(
    type = cms.string('MaterialBudgetAction'),
    MaterialBudgetAction = cms.PSet(
        HistosFile = cms.string('matbdg_%s.root' % options.label),
        AllStepsToTree = cms.bool(True),
        HistogramList = cms.string('Tracker'),
        SelectedVolumes = cms.vstring(_components),
        TreeFile = cms.string('None'), ## is NOT requested

        StopAfterProcess = cms.string('None'),
#        TextFile = cms.string("matbdg_Tracker.txt")
        TextFile = cms.string('None')
    )
))
