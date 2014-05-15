#ifndef SimGeneral_MixingModule_PileUpEventPrincipal_h
#define SimGeneral_MixingModule_PileUpEventPrincipal_h

#include <set>
#include <string>

#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/ProductKindOfType.h"
#include "FWCore/Utilities/interface/TypeID.h"
#include "DataFormats/Common/interface/BasicHandle.h"
#include "DataFormats/Common/interface/ConvertHandle.h"
#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"

class PileUpEventPrincipal {
public:

  PileUpEventPrincipal(edm::EventPrincipal const& ep, int bcr) :
    principal_(ep), bunchCrossing_(bcr) {}

  edm::EventPrincipal const& principal() {
    return principal_;
  }

  edm::EventPrincipal const& principal() const {
    return principal_;
  }

  int bunchCrossing() const {
    return bunchCrossing_;
  }


  template<typename T>
  bool
    getByLabel(edm::InputTag const& tag, edm::Handle<T>& result) const {
    typedef typename T::value_type ItemType;
    typedef typename T::iterator iterator;
    edm::BasicHandle bh = principal_.getByLabel(edm::PRODUCT_TYPE, edm::TypeID(typeid(T)), tag);
    convert_handle(bh, result);
    return result.isValid();
  }

private: 
  edm::EventPrincipal const& principal_;  
  int bunchCrossing_;
};

#endif

