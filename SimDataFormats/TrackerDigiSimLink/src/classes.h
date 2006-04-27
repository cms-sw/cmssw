#ifndef TRACKERDIGISIMLINK_CLASSES_H
#define TRACKERDIGISIMLINK_CLASSES_H

#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include <vector>

#include "SimDataFormats/TrackerDigiSimLink/interface/PixelDigiSimLinkCollection.h"

namespace {
  namespace {
    edm::Wrapper<PixelDigiSimLinkCollection> PixelDigiSimLinkCollectionWrapper;
  }
}

#include "SimDataFormats/TrackerDigiSimLink/interface/StripDigiSimLink.h"

namespace {
  namespace {
    edm::Wrapper<StripDigiSimLink> StripDigiSimLinkWrapper;
    edm::Wrapper< edm::DetSet<StripDigiSimLink> > StripDigiSimLinkDetSetWrapper;
    edm::Wrapper< edm::DetSetVector<StripDigiSimLink> > StripDigiSimLinkDetSetVectorWrapper;
  }
}

#endif // TRACKERDIGISIMLINK_CLASSES_H

