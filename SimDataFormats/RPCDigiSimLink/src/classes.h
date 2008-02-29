#ifndef RPCDIGISIMLINK_CLASSES_H
#define RPCDIGISIMLINK_CLASSES_H

#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"
#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"
#include <vector>
#include <map>


#include "SimDataFormats/RPCDigiSimLink/interface/RPCDigiSimLink.h"

namespace {
  namespace {
    edm::Wrapper< std::vector<RPCDigiSimLink>  > RPCDigiSimLinkVector;
    edm::Wrapper< std::vector<edm::DetSet<RPCDigiSimLink> > > RPCDigiSimLinkVectorDetSet; 
    edm::Wrapper<RPCDigiSimLink> RPCDigiSimLinkWrapper;
    edm::Wrapper< edm::DetSet<RPCDigiSimLink> > RPCDigiSimLinkDetSetWrapper;
    edm::Wrapper< edm::DetSetVector<RPCDigiSimLink> > RPCDigiSimLinkDetSetVectorWrapper;
  }
}

#endif // RPCDIGISIMLINK_CLASSES_H


