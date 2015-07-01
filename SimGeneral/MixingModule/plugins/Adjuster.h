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
#include <string>
#include <iostream>

namespace edm {
  class AdjusterBase {
  public:
    virtual ~AdjusterBase() {}
    virtual void doOffset(int bunchspace, int bcr, const edm::EventPrincipal&, unsigned int EventNr, int vertexOffset) = 0;
    virtual bool checkSignal(edm::Event const& event) = 0;
  };

  template<typename T>
  class Adjuster : public AdjusterBase {

  public:
    Adjuster(InputTag const& tag);

    virtual ~Adjuster() {}

    virtual void doOffset(int bunchspace, int bcr, const edm::EventPrincipal&, unsigned int EventNr, int vertexOffset);

    virtual bool checkSignal(edm::Event const& event) {
      bool got = false;
      edm::Handle<std::vector<T> > result_t;
      got = event.getByLabel(tag_, result_t);
      return got;
    }

   private:
    InputTag tag_;
    bool WrapT_;
  };

  //==============================================================================
  //                              implementations
  //==============================================================================

  namespace detail {
    void doTheOffset(int bunchspace, int bcr, std::vector<SimTrack>& product, unsigned int eventNr, int vertexOffset, bool wraptimes);
    void doTheOffset(int bunchspace, int bcr, std::vector<SimVertex>& product, unsigned int eventNr, int vertexOffset, bool wrapti\
mes);
    void doTheOffset(int bunchspace, int bcr, std::vector<PCaloHit>& product, unsigned int eventNr, int vertexOffset, bool wrapti\
mes);
    void doTheOffset(int bunchspace, int bcr, std::vector<PSimHit>& product, unsigned int eventNr, int vertexOffset, bool wrapti\
mes);
  }

  template<typename T>
    void  Adjuster<T>::doOffset(int bunchspace, int bcr, const EventPrincipal &ep, unsigned int eventNr, int vertexOffset) {
    boost::shared_ptr<Wrapper<std::vector<T> > const> shPtr = getProductByTag<std::vector<T> >(ep, tag_);
    if (shPtr) {
      std::vector<T>& product = const_cast<std::vector<T>&>(*shPtr->product());
      detail::doTheOffset(bunchspace, bcr, product, eventNr, vertexOffset, WrapT_);
    }
  }

  template<typename T>
  Adjuster<T>::Adjuster(InputTag const& tag) : tag_(tag) {
    std::string Musearch = tag_.instance();
    if(Musearch.find("Muon") == 0) WrapT_ = true; // wrap time for neutrons in Muon system subdetectors
  }
}

#endif
