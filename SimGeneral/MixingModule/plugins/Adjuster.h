#ifndef SimGeneral_MixingModule_Adjuster_h
#define SimGeneral_MixingModule_Adjuster_h

#include "DataFormats/Common/interface/Wrapper.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

#include "boost/shared_ptr.hpp"

#include <vector>

namespace edm {
  class ModuleCallingContext;

  class AdjusterBase {
  public:
    virtual ~AdjusterBase() {}
    virtual void doOffset(int bunchspace, int bcr, const edm::EventPrincipal&, ModuleCallingContext const*, unsigned int EventNr, int vertexOffset) = 0;
    virtual bool checkSignal(edm::Event const& event) = 0;
  };

  template<typename T>
  class Adjuster : public AdjusterBase {

  public:
    Adjuster(InputTag const& tag);

    virtual ~Adjuster() {}

    virtual void doOffset(int bunchspace, int bcr, const edm::EventPrincipal&, ModuleCallingContext const*, unsigned int EventNr, int vertexOffset);

    virtual bool checkSignal(edm::Event const& event) {
      bool got = false;
      edm::Handle<std::vector<T> > result_t;
      got = event.getByLabel(tag_, result_t);
      return got;
    }

   private:
    InputTag tag_;
  };

  //==============================================================================
  //                              implementations
  //==============================================================================

  namespace detail {
    void doTheOffset(int bunchspace, int bcr, std::vector<SimTrack>& product, unsigned int eventNr, int vertexOffset);
    void doTheOffset(int bunchspace, int bcr, std::vector<SimVertex>& product, unsigned int eventNr, int vertexOffset);
    void doTheOffset(int bunchspace, int bcr, std::vector<PCaloHit>& product, unsigned int eventNr, int vertexOffset);
    void doTheOffset(int bunchspace, int bcr, std::vector<PSimHit>& product, unsigned int eventNr, int vertexOffset);
  }

  template<typename T>
  void  Adjuster<T>::doOffset(int bunchspace, int bcr, const EventPrincipal &ep, ModuleCallingContext const* mcc, unsigned int eventNr, int vertexOffset) {
    boost::shared_ptr<Wrapper<std::vector<T> > const> shPtr = getProductByTag<std::vector<T> >(ep, tag_, mcc);
    if (shPtr) {
      std::vector<T>& product = const_cast<std::vector<T>&>(*shPtr->product());
      detail::doTheOffset(bunchspace, bcr, product, eventNr, vertexOffset);
    }
  }

  template<typename T>
  Adjuster<T>::Adjuster(InputTag const& tag) : tag_(tag) {
  }
}

#endif
