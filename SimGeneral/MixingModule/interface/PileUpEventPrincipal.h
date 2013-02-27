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
class PCaloHit;
class PSimHit;
class SimTrack;
class SimVertex;

class PileUpEventPrincipal {
public:

  PileUpEventPrincipal(edm::EventPrincipal const& ep, int bcr, int bsp, int eventId, int vtxOffset) :
    principal_(ep), bunchCrossing_(bcr), bunchCrossingXbunchSpace_(bcr*bsp), id_(bcr, eventId), vertexOffset_(vtxOffset), labels_() {}

  bool
  addLabel(edm::TypeID const& type, std::string const& label) const {
    return true;
    //return labels_.insert(std::make_pair(type, label)).second;
  }

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
  void
  adjust(T& item) const {
  }

  template<typename T>
  bool
    getByLabel(edm::InputTag const& tag, edm::Handle<T>& result) const {
    typedef typename T::value_type ItemType;
    typedef typename T::iterator iterator;
    edm::BasicHandle bh = principal_.getByLabel(edm::PRODUCT_TYPE, edm::TypeID(typeid(T)), tag);
    convert_handle(bh, result);
    if(result.isValid() && addLabel(edm::TypeID(typeid(T)), tag.label())) {
      T& product = const_cast<T&>(*result.product());
      for(iterator i = product.begin(), iEnd = product.end(); i != iEnd; ++i) {
        adjust<ItemType>(*i);
      }
    }
    return result.isValid();
  }

private: 
  edm::EventPrincipal const& principal_;  
  int bunchCrossing_;
  int bunchCrossingXbunchSpace_;
  EncodedEventId id_;
  int vertexOffset_;
  mutable std::set<std::pair<edm::TypeID, std::string> > labels_;
};

template<>
void PileUpEventPrincipal::adjust<PCaloHit>(PCaloHit& item) const;

template<>
void PileUpEventPrincipal::adjust<PSimHit>(PSimHit& item) const;

template<>
void PileUpEventPrincipal::adjust<SimTrack>(SimTrack& item) const;

template<>
void PileUpEventPrincipal::adjust<SimVertex>(SimVertex& item) const;
#endif
