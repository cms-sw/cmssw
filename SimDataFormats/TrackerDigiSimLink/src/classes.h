#ifndef TRACKERDIGISIMLINK_CLASSES_H
#define TRACKERDIGISIMLINK_CLASSES_H

#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include <vector>

#include "SimDataFormats/TrackerDigiSimLink/interface/PixelDigiSimLink.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/StripDigiSimLink.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/StripCompactDigiSimLinks.h"

namespace {
  struct dictionary {
   edm::Wrapper<PixelDigiSimLink> PixelLink1;
    edm::Wrapper< std::vector<PixelDigiSimLink>  > PixelLink2;
    edm::Wrapper< edm::DetSet<PixelDigiSimLink> > PixelLink3;
    edm::Wrapper< std::vector<edm::DetSet<PixelDigiSimLink> > > PixelLink4;
    edm::Wrapper< edm::DetSetVector<PixelDigiSimLink> > PixelLink5;

    edm::Wrapper< std::vector<StripDigiSimLink>  > StripDigiSimLinkVector;
    edm::Wrapper< std::vector<edm::DetSet<StripDigiSimLink> > > StripDigiSimLinkVectorDetSet; 
    edm::Wrapper<StripDigiSimLink> StripDigiSimLinkWrapper;
    edm::Wrapper< edm::DetSet<StripDigiSimLink> > StripDigiSimLinkDetSetWrapper;
    edm::Wrapper< edm::DetSetVector<StripDigiSimLink> > StripDigiSimLinkDetSetVectorWrapper;

    edm::Wrapper<StripCompactDigiSimLinks> StripCompactDigiSimLinksWrapper_;
  };
}

#endif // TRACKERDIGISIMLINK_CLASSES_H

