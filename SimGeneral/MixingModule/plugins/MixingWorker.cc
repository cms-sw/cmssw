#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/CrossingFrame/interface/PCrossingFrame.h"
#include "MixingWorker.h"

#include "boost/shared_ptr.hpp"

namespace edm {
  template <>
  void MixingWorker<PCaloHit>::addPileups(const int bcr, const EventPrincipal &ep, unsigned int eventNr,int vertexoffset) {

    boost::shared_ptr<Wrapper<std::vector<PCaloHit> > const> shPtr = edm::getProductByTag<std::vector<PCaloHit> >(ep, tag_);

    if (shPtr) {
      LogDebug("MixingModule") <<shPtr->product()->size()<<"  pileup objects  added, eventNr "<<eventNr;
      crFrame_->setPileupPtr(shPtr);
      crFrame_->addPileups(bcr,const_cast< std::vector<PCaloHit> * >(shPtr->product()),eventNr);
    }
  }

  template <>
  void MixingWorker<PSimHit>::addPileups(const int bcr, const EventPrincipal &ep,unsigned int eventNr,int vertexoffset) {
    //    changed for high/low treatment
    boost::shared_ptr<Wrapper<std::vector<PSimHit> > const> shPtr = getProductByTag<std::vector<PSimHit> >(ep, tag_);
    if (shPtr) {
      LogDebug("MixingModule") <<shPtr->product()->size()<<"  pileup objects  added, eventNr "<<eventNr;
      crFrame_->setPileupPtr(shPtr);
      crFrame_->addPileups(bcr,const_cast< std::vector<PSimHit> * > (shPtr->product()),eventNr);
    }
  }

  template <>
  void  MixingWorker<SimTrack>::addPileups(const int bcr, const EventPrincipal &ep,unsigned int eventNr,int vertexoffset) { 
      // changed to transmit vertexoffset
      boost::shared_ptr<Wrapper<std::vector<SimTrack> > const> shPtr = getProductByTag<std::vector<SimTrack> >(ep, tag_);
      
      if (shPtr) {
	LogDebug("MixingModule") <<shPtr->product()->size()<<"  pileup objects  added, eventNr "<<eventNr;
        crFrame_->setPileupPtr(shPtr);
        crFrame_->addPileups(bcr,const_cast< std::vector<SimTrack> * > (shPtr->product()),eventNr,vertexoffset);
      }
  }

  template <>
  void MixingWorker<SimVertex>::addPileups(const int bcr, const EventPrincipal &ep,unsigned int eventNr,int vertexoffset) {
  
    // changed to take care of vertexoffset
    boost::shared_ptr<Wrapper<std::vector<SimVertex> > const> shPtr = getProductByTag<std::vector<SimVertex> >(ep, tag_);
        
    if (shPtr) {
      LogDebug("MixingModule") <<shPtr->product()->size()<<"  pileup objects  added, eventNr "<<eventNr;
      vertexoffset+=shPtr->product()->size();
      crFrame_->setPileupPtr(shPtr);
      crFrame_->addPileups(bcr,const_cast< std::vector<SimVertex> * > (shPtr->product()),eventNr);
    }
  }

  template <>
  void MixingWorker<HepMCProduct>::addPileups(const int bcr, const EventPrincipal& ep,unsigned int eventNr,int vertexoffset) {
    // HepMCProduct does not come as a vector....
    boost::shared_ptr<Wrapper<HepMCProduct> const> shPtr = getProductByTag<HepMCProduct>(ep, tag_);
    if (shPtr) {
      LogDebug("MixingModule") <<"HepMC pileup objects  added, eventNr "<<eventNr << " Tag " << tag_ << std::endl;
      crFrame_->setPileupPtr(shPtr);
      crFrame_->addPileups(bcr,const_cast<HepMCProduct*> (shPtr->product()),eventNr);
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
	  
	  if (got) {
	       LogInfo("MixingModule") <<" Will create a CrossingFrame for HepMCProduct with "
	  			       << " with InputTag= "<< t.encode();
          }
				       
	  return got;
  }
      
}//namespace edm
