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

_ALLOWED_COMPS = ['BEAM', 'Tracker', 'PixelBarrel', 
                  'PixelForwardZMinus', 'PixelForwardZPlus',
                  'TIB', 'TOB', 'TIDB', 'TIDF',
                  'TEC', 'TIBTIDServicesF', 'TIBTIDServicesB',
                  'TrackerOuterCylinder', 'TrackerBulkhead', 'ECAL']

options = VarParsing('analysis')
options.register('geom',        #name
                 'phaseI',      #default value
                 VarParsing.multiplicity.singleton,   # kind of options
                 VarParsing.varType.string,           # type of option
                 "Select the geometry to be studied"  # help message
                )
options.register('components',         #name
                 '',             #default value
                 VarParsing.multiplicity.list,        # kind of options
                 VarParsing.varType.string,           # type of option
                 "Select the geometry component to be studied"  # help message
                )

options.register('label',         #name
                 '',              #default value
                 VarParsing.multiplicity.singleton,   # kind of options
                 VarParsing.varType.string,           # type of option
                 "Select the label to be used to create output files. Default to tracker. If multiple components are selected, it defaults to the join of all components, with '_' as separator."  # help message
                )

options.setDefault('inputFiles', ['file:single_neutrino_random.root'])

options.parseArguments()
# Option validation

for comp in options.components:
  if comp not in _ALLOWED_COMPS:
    print "Error, '%s' not registered as a valid components to monitor." % comp
    print "Allowed components:", _ALLOWED_COMPS
    raise RuntimeError("Unknown components")

if options.label == '':
  options.label = '_'.join(options.components)

#
#Geometry
#
if options.geom == 'phaseI':
  process.load("Configuration.Geometry.GeometryExtended2017_cff")
elif options.geom == 'run2':
  process.load("Configuration.Geometry.GeometryExtended2016_cff")
#process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
#process.load("Geometry.HcalCommonData.hcalParameters_cfi")
#process.load("Geometry.HcalCommonData.hcalDDDSimConstants_cfi")

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
        SelectedVolumes = cms.vstring(options.components),
        TreeFile = cms.string('None'), ## is NOT requested

        StopAfterProcess = cms.string('None'),
#        TextFile = cms.string("matbdg_Tracker.txt")
        TextFile = cms.string('None')
    )
))
