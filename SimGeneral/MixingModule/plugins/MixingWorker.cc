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
    std::shared_ptr<Wrapper<HepMCProduct> const> shPtr = getProductByTag<HepMCProduct>(ep, tag_, mcc);
    if(!shPtr) {
       shPtr = getProductByTag<HepMCProduct>(ep, InputTag("generator"), mcc);
    }
    if (shPtr) {
      LogDebug("MixingModule") <<"HepMC pileup objects  added, eventNr "<<eventNr << " Tag " << tag_ << std::endl;
      crFrame_->setPileupPtr(shPtr);
      crFrame_->addPileups(*shPtr->product());
    }
  }

  template <>
  void MixingWorker<HepMCProduct>::addSignals(const Event &e) { 
    //HepMC - here the interface is different!!!
    Handle<HepMCProduct>  result_t;
    bool got = e.getByLabel(tag_,result_t);
    if (got) {
      LogDebug("MixingModule") <<" adding HepMCProduct from signal event  with "<<tag_;
      crFrame_->addSignals(result_t.product(),e.id());  
    } else {
      LogInfo("MixingModule") <<"!!!!!!! Did not get any signal data for HepMCProduct with "<<tag_;
    }
  }
  
  template <>
  bool MixingWorker<HepMCProduct>::checkSignal(const Event &e) {   
          bool got;
	  InputTag t;
	  
	  Handle<HepMCProduct> result_t;
	  got = e.getByLabel(tag_,result_t);
	  t = InputTag(tag_.label(),tag_.instance());
          if(!got) {
	     got = e.getByLabel(InputTag("generator","unsmeared"),result_t);
          }
	  
	  if (got) {
	       LogInfo("MixingModule") <<" Will create a CrossingFrame for HepMCProduct with "
	  			       << " with InputTag= "<< t.encode();
          }
				       
	  return got;
  }
      
}//namespace edm
