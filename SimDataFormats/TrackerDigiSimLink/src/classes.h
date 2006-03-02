#ifndef TRACKERDIGISIMLINK_CLASSES_H
#define TRACKERDIGISIMLINK_CLASSES_H
#include "SimDataFormats/TrackerDigiSimLink/interface/PixelDigiSimLinkCollection.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/PixelDigiSimLink.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/StripDigiSimLinkCollection.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/StripDigiSimLink.h"
#include "FWCore/EDProduct/interface/Wrapper.h"
#include <vector>

namespace {
  namespace {
    edm::Wrapper<PixelDigiSimLinkCollection> PixelDigiSimLinkCollectionWrapper;
    edm::Wrapper<StripDigiSimLinkCollection> StripDigiSimLinkCollectionWrapper;
  }
}

#endif // TRACKERDIGISIMLINK_CLASSES_H

