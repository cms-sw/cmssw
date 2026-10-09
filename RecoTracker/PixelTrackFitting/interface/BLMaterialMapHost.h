#ifndef RecoTracker_PixelTrackFitting_interface_BLMaterialMapHost_h
#define RecoTracker_PixelTrackFitting_interface_BLMaterialMapHost_h

#include "DataFormats/Portable/interface/PortableHostObject.h"
#include "RecoTracker/PixelTrackFitting/interface/BLMaterialMap.h"

// The material map of the BrokenLine fit in host memory, the EventSetup product of BLMaterialMapESProducerAlpaka.
using BLMaterialMapHost = PortableHostObject<blMaterialMap::Map>;

#endif  // RecoTracker_PixelTrackFitting_interface_BLMaterialMapHost_h
