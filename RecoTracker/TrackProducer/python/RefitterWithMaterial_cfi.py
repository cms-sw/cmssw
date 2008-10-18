import FWCore.ParameterSet.Config as cms

import sys
print "WARNING: RecoTracker/TrackProducer/python/RefitterWithMaterial_cff.py and \nRecoTracker/TrackProducer/python/RefitterWithMaterial_cfi.py are obsolete and will be removed soon.";
print "Please use RecoTracker/TrackProducer/python/TrackRefitters_cff.py \nand RecoTracker/TrackProducer/python/TrackRefitter_cfi.py";
print "if still using old configuration files."
#print " Now exiting with -1 error. \nplease remove inclusion of this file and try again.";
#print "starting with the next release including this file will cause an exception";
#sys.exit(-1);


from RecoTracker.TrackProducer.TrackRefitter_cfi import *

