#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/CrossingFrame/interface/PCrossingFrame.h"
#include "MixingWorker.h"

#include <memory>

namespace edm {
  template <>
  void MixingWorker<HepMCProduct>::addPileups(const EventPrincipal& ep, ModuleCallingContext const* mcc, unsigned int eventNr) {
    // HepMCProduct does not come as a vector....
    for(InputTag const& tag : allTags_) {
      std::shared_ptr<Wrapper<HepMCProduct> const> shPtr = getProductByTag<HepMCProduct>(ep, tag, mcc);
      if(shPtr) {
        LogDebug("MixingModule") << "HepMC pileup objects  added, eventNr " << eventNr << " Tag " << tag << std::endl;
        crFrame_->setPileupPtr(shPtr);
        crFrame_->addPileups(*shPtr->product());
        break;
      }
    }
  }

  template <>
  void MixingWorker<HepMCProduct>::addSignals(const Event &e) { 
    //HepMC - here the interface is different!!!
    bool got = false;
    Handle<HepMCProduct>  result_t;
    for(InputTag const& tag : allTags_) {
      got = e.getByLabel(tag, result_t);
      if (got) {
        LogDebug("MixingModule") << "adding HepMCProduct from signal event  with " << tag;
        crFrame_->addSignals(result_t.product(), e.id());  
        break;
      }
    }
    if(!got) {
      LogInfo("MixingModule") << "!!!!!!! Did not get any signal data for HepMCProduct with " << allTags_[0];
    }
  }
  
  template <>
  bool MixingWorker<HepMCProduct>::checkSignal(const Event &e) {   
          bool got = false;
	  Handle<HepMCProduct> result_t;
          for(InputTag const& tag : allTags_) {
	    got = e.getByLabel(tag, result_t);
            if(got) {
	      InputTag t = InputTag(tag.label(), tag.instance());
	      LogInfo("MixingModule") <<" Will create a CrossingFrame for HepMCProduct with "
                                      << " with InputTag= "<< t.encode();
              break;
            }
          }
	  return got;
  }
}//namespace edm
