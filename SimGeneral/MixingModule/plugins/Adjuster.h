#ifndef SimGeneral_MixingModule_Adjuster_h
#define SimGeneral_MixingModule_Adjuster_h

#include "DataFormats/Common/interface/Wrapper.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"

#include <memory>
#include <vector>
#include <string>
#include <iostream>

class FastTrackerRecHit;

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
    Adjuster(InputTag const& tag, edm::ConsumesCollector&& iC, bool wrap);

    virtual ~Adjuster() {}

    virtual void doOffset(int bunchspace, int bcr, const edm::EventPrincipal&, ModuleCallingContext const*, unsigned int EventNr, int vertexOffset);

    virtual bool checkSignal(edm::Event const& event) {
      bool got = false;
      edm::Handle<T> result_t;
      got = event.getByToken(token_, result_t);
      return got;
    }

   private:
    InputTag tag_;
    bool WrapT_ = false;
    EDGetTokenT<T> token_;
  };

  //==============================================================================
  //                              implementations
  //==============================================================================

  namespace detail {
    void doTheOffset(int bunchspace, int bcr, std::vector<SimTrack>& product, unsigned int eventNr, int vertexOffset, bool wraptimes);
    void doTheOffset(int bunchspace, int bcr, std::vector<SimVertex>& product, unsigned int eventNr, int vertexOffset, bool wraptimes);
    void doTheOffset(int bunchspace, int bcr, std::vector<PCaloHit>& product, unsigned int eventNr, int vertexOffset, bool wraptimes);
    void doTheOffset(int bunchspace, int bcr, std::vector<PSimHit>& product, unsigned int eventNr, int vertexOffset, bool wraptimes);
    void doTheOffset(int bunchspace, int bcr, TrackingRecHitCollection & product, unsigned int eventNr, int vertexOffset, bool wraptimes);
  }

  template<typename T>
  void  Adjuster<T>::doOffset(int bunchspace, int bcr, const EventPrincipal &ep, ModuleCallingContext const* mcc, unsigned int eventNr, int vertexOffset) {
    std::shared_ptr<Wrapper<T> const> shPtr = getProductByTag<T>(ep, tag_, mcc);
    if (shPtr) {
      T& product = const_cast<T&>(*shPtr->product());
      detail::doTheOffset(bunchspace, bcr, product, eventNr, vertexOffset, WrapT_);
    }
  }

  template<typename T>
    Adjuster<T>::Adjuster(InputTag const& tag, ConsumesCollector&& iC, bool wrapLongTimes) : tag_(tag), token_(iC.consumes<T>(tag)) {
    if(wrapLongTimes) {
      std::string Musearch = tag_.instance();
      if(Musearch.find("Muon") == 0) WrapT_ = true; // wrap time for neutrons in Muon system subdetectors
    }
  }
}

#endif
