from FWCore.ParameterSet.VarParsing import VarParsing
import sys

options = VarParsing('analysis')

ALLOWED_GEOMETRIES = ['run2', 'Phase1', 'Phase2']

options.register('geometry',
                 'run2',
                 VarParsing.multiplicity.singleton,
                 VarParsing.varType.string,
                 """ Specify which geometry scenario to use. The different geometry
                 will automatically load also the correct XML grouping
                 files for the material description. Currently supported values are %s""" % " ".join(ALLOWED_GEOMETRIES) 
                 )

options.parseArguments()

if not options.geometry in ALLOWED_GEOMETRIES:
    print("\n**** ERROR ****\nUnknown geometry %s. Quitting.\n**** ERROR ****\n" % options.geometry)
    sys.exit(1)
