# In order to produce what you need (or in a loop)
#time cmsRun runP_FromVertexUpToInFrontOfMuonStations_cfg.py geom=Extended2023D41 label=BeamPipe
#time cmsRun runP_FromVertexUpToInFrontOfMuonStations_cfg.py geom=Extended2023D41 label=Tracker
#time cmsRun runP_FromVertexUpToInFrontOfMuonStations_cfg.py geom=Extended2023D41 label=ECAL
#time cmsRun runP_FromVertexUpToInFrontOfMuonStations_cfg.py geom=Extended2023D41 label=HCal
##EndcapTimingLayer + Thermal Screen (Barrel Timing Layer is included in the Tracker)
#time cmsRun runP_FromVertexUpToInFrontOfMuonStations_cfg.py geom=Extended2023D41 label='EndcapTimingLayer + Thermal Screen'
##Neutron Moderator + Thermal Screen
#time cmsRun runP_FromVertexUpToInFrontOfMuonStations_cfg.py geom=Extended2023D41 label='Neutron Moderator + Thermal Screen'
##HGCal + HGCal Service + Thermal Screen
#time cmsRun runP_FromVertexUpToInFrontOfMuonStations_cfg.py geom=Extended2023D41 label='HGCal + HGCal Service + Thermal Screen'
##Solenoid Magnet
#time cmsRun runP_FromVertexUpToInFrontOfMuonStations_cfg.py geom=Extended2023D41 label='Solenoid Magnet'
##Muon Wheels and Cables
#time cmsRun runP_FromVertexUpToInFrontOfMuonStations_cfg.py geom=Extended2023D41 label='Muon Wheels and Cables'
##Finally, all together 
#time cmsRun runP_FromVertexUpToInFrontOfMuonStations_cfg.py geom=Extended2023D41 label=FromVertexToBackOfHGCal

import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing
import sys, re

process = cms.Process("PROD")

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

# The default geometry is Extended2023D41. If a different geoemtry
# is needed, the appropriate flag has to be passed at command line,
# e.g.: cmsRun runP_FromVertexUpToInFrontOfMuonStations_cfg.py geom="XYZ"

# The default component to be monitored is the HGCal. If other
# components need to be studied, they must be supplied, one at a time,
# at the command line, e.g.: cmsRun runP_FromVertexUpToInFrontOfMuonStations_cfg.py
# label="XYZ"

from Validation.Geometry.plot_hgcal_utils import _LABELS2COMPS

_ALLOWED_LABELS = _LABELS2COMPS.keys()

options = VarParsing('analysis')
options.register('geom',             #name
                 'Extended2023D41',      #default value
                 VarParsing.multiplicity.singleton,   # kind of options
                 VarParsing.varType.string,           # type of option
                 "Select the geometry to be studied"  # help message
                )

options.register('label',         #name
                 'HGCal',              #default value
                 VarParsing.multiplicity.singleton,   # kind of options
                 VarParsing.varType.string,           # type of option
                 "Select the label to be used to create output files. Default to HGCal. If multiple components are selected, it defaults to the join of all components, with '_' as separator."  # help message
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

# Load geometry either from the Database of from files
process.load("Configuration.Geometry.Geometry%s_cff" % options.geom)

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
        HistogramList = cms.string('HGCal'),
        SelectedVolumes = cms.vstring(_components),
        TreeFile = cms.string('None'), ## is NOT requested

        StopAfterProcess = cms.string('None'),
        #        TextFile = cms.string("matbdg_HGCal.txt")
        TextFile = cms.string('None'), 
        #Setting ranges for histos
        #Make z 2mm per bin. Be careful this could lead to memory crashes if too low. 
        minZ = cms.double(-7000.),
        maxZ = cms.double(7000.),
        nintZ = cms.int32(7000), 
        # Make r 1cm per bin
        rMin = cms.double(-50.), 
        rMax = cms.double(8000.),
        nrbin = cms.int32(805),
        # eta
        etaMin = cms.double(-5.), 
        etaMax = cms.double(5.),
        netabin = cms.int32(250),
        # phi
        phiMin = cms.double(-3.2), 
        phiMax = cms.double(3.2),
        nphibin = cms.int32(180),
        # R for profile histos
        RMin =  cms.double(0.), 
        RMax =  cms.double(8000.), 
        nRbin = cms.int32(800)

     )
))
