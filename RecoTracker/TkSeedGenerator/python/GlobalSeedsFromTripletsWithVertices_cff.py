import FWCore.ParameterSet.Config as cms

print
print "in file GlobalSeedsFromTripletsWithVertices_cff.py"
print "The name of this file is mis-leading. Please use GlobalSeedsFromTriplets_cff.py instead"
print "if you meant to use global triplt with vertex constraint"
print

import RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff
globalSeedsFromTripletsWithVertices = RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff.globalSeedsFromTriplets.clone()
