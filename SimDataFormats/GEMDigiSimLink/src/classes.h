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
#include "SimDataFormats/GEMDigiSimLink/interface/ME0DigiSimLink.h"

namespace SimDataFormats_GEMDigiSimLink
{
  struct dictionary
  {
    edm::Wrapper<GEMDigiSimLink> GEMDigiSimLinkWrapper;
    edm::Wrapper<std::vector<GEMDigiSimLink> > GEMDigiSimLinkVector;

    edm::Wrapper<edm::DetSet<GEMDigiSimLink> > GEMDigiSimLinkDetSetWrapper;
    edm::Wrapper<std::vector<edm::DetSet<GEMDigiSimLink> > > GEMDigiSimLinkVectorDetSet;
    edm::Wrapper<edm::DetSetVector<GEMDigiSimLink> > GEMDigiSimLinkDetSetVectorWrapper;

    edm::Wrapper<ME0DigiSimLink> ME0DigiSimLinkWrapper;
    edm::Wrapper<std::vector<ME0DigiSimLink> > ME0DigiSimLinkVector;

    edm::Wrapper<edm::DetSet<ME0DigiSimLink> > ME0DigiSimLinkDetSetWrapper;
    edm::Wrapper<std::vector<edm::DetSet<ME0DigiSimLink> > > ME0DigiSimLinkVectorDetSet;
    edm::Wrapper<edm::DetSetVector<ME0DigiSimLink> > ME0DigiSimLinkDetSetVectorWrapper;
  };
}

#endif // GEMDIGISIMLINK_CLASSES_H

