#include "DataFormats/Portable/interface/PortableHostCollectionReadRules.h"
#include "RecoTracker/PixelSeeding/interface/TripletDumpHost.h"

// ROOT read rules for the per-built-triplet host SoA, as for any single-layout PortableHostCollection.
SET_PORTABLEHOSTCOLLECTION_READ_RULES(PortableHostCollection<caStructures::TripletDumpSoA>);
