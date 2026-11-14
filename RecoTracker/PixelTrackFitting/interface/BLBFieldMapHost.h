#ifndef RecoTracker_PixelTrackFitting_interface_BLBFieldMapHost_h
#define RecoTracker_PixelTrackFitting_interface_BLBFieldMapHost_h

#include "DataFormats/Portable/interface/PortableHostObject.h"
#include "RecoTracker/PixelTrackFitting/interface/BLBFieldMap.h"

// The (Bz, Br) lattice of the BrokenLine fit in host memory, the EventSetup product of BLBFieldMapESProducerAlpaka.
using BLBFieldMapHost = PortableHostObject<blBFieldMap::Map>;

#endif  // RecoTracker_PixelTrackFitting_interface_BLBFieldMapHost_h
