#ifndef GEMDIGISIMLINK_CLASSES_H
#define GEMDIGISIMLINK_CLASSES_H

#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"
#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"
#include <vector>
#include <map>

#include "SimDataFormats/GEMDigiSimLink/interface/GEMDigiSimLink.h"

namespace SimDataFormats_GEMDigiSimLink
{
  struct dictionary
  {
    edm::Wrapper<GEMDigiSimLink> GEMDigiSimLinkWrapper;
    edm::Wrapper<std::vector<GEMDigiSimLink> > GEMDigiSimLinkVector;

    edm::Wrapper<edm::DetSet<GEMDigiSimLink> > GEMDigiSimLinkDetSetWrapper;
    edm::Wrapper<std::vector<edm::DetSet<GEMDigiSimLink> > > GEMDigiSimLinkVectorDetSet;
    edm::Wrapper<edm::DetSetVector<GEMDigiSimLink> > GEMDigiSimLinkDetSetVectorWrapper;
  };
}

#endif // GEMDIGISIMLINK_CLASSES_H

