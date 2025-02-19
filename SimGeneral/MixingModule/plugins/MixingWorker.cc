#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/CrossingFrame/interface/PCrossingFrame.h"
#include "MixingWorker.h"

#include "boost/shared_ptr.hpp"

namespace edm {
  template <>
  void MixingWorker<PSimHit>::addPileups(const int bcr, const EventPrincipal &ep,unsigned int eventNr,int vertexoffset)
  {
    if (!mixProdStep2_){ 
      //    default version changed for high/low treatment
      boost::shared_ptr<Wrapper<std::vector<PSimHit> > const> shPtr = getProductByTag<std::vector<PSimHit> >(ep, tag_);
        if (shPtr) {
	  LogDebug("MixingModule") <<shPtr->product()->size()<<"  pileup objects  added, eventNr "<<eventNr;
          crFrame_->setPileupPtr(shPtr);
	  crFrame_->addPileups(bcr,const_cast< std::vector<PSimHit> * > (shPtr->product()),eventNr);
        }
      } else{ // In case mixProdStep2_=true
        boost::shared_ptr<Wrapper<PCrossingFrame<PSimHit> > const> shPtr = getProductByTag<PCrossingFrame<PSimHit> >(ep, tag_);
        if (shPtr) { 
          crFrame_->setPileupPtr(shPtr);
      	  secSourceCF_ = const_cast<PCrossingFrame<PSimHit> * >(shPtr->product());
	  LogDebug("MixingModule") << "Add PCrossingFrame<PSimHit>,  eventNr " << secSourceCF_->getEventID();
	  copyPCrossingFrame(secSourceCF_);	  
	} 
        else 
	  LogDebug("MixingModule") << "Could not get the PCrossingFrame<PSimHit>!";
    }//else mixProd2    
  }


  template <>
  void  MixingWorker<SimTrack>::addPileups(const int bcr, const EventPrincipal &ep,unsigned int eventNr,int vertexoffset)
  { 
    if (!mixProdStep2_){ 
      // default version changed to transmit vertexoffset
      boost::shared_ptr<Wrapper<std::vector<SimTrack> > const> shPtr = getProductByTag<std::vector<SimTrack> >(ep, tag_);
      
      if (shPtr) {
	LogDebug("MixingModule") <<shPtr->product()->size()<<"  pileup objects  added, eventNr "<<eventNr;
        crFrame_->setPileupPtr(shPtr);
        crFrame_->addPileups(bcr,const_cast< std::vector<SimTrack> * > (shPtr->product()),eventNr,vertexoffset);
      }
    }
    else
    { // In case mixProdStep2_=true	
	boost::shared_ptr<Wrapper<PCrossingFrame<SimTrack> > const> shPtr = getProductByTag<PCrossingFrame<SimTrack> >(ep, tag_);

	if (shPtr){
	  crFrame_->setPileupPtr(shPtr);
	  secSourceCF_ = const_cast<PCrossingFrame<SimTrack> * >(shPtr->product());
	  LogDebug("MixingModule") << "Add PCrossingFrame<SimTrack>,  eventNr " << secSourceCF_->getEventID();
		
	  // Get PCrossingFrame data members values from the mixed secondary sources file
	  copyPCrossingFrame(secSourceCF_);
	}
	else 
	  LogDebug("MixingModule") << "Could not get the PCrossingFrame<SimTrack>!";
    }
  }


  template <>
  void MixingWorker<SimVertex>::addPileups(const int bcr, const EventPrincipal &ep,unsigned int eventNr,int vertexoffset)
  {
  
    if (!mixProdStep2_){ 
      // default version changed to take care of vertexoffset
      boost::shared_ptr<Wrapper<std::vector<SimVertex> > const> shPtr = getProductByTag<std::vector<SimVertex> >(ep, tag_);
        
      if (shPtr) {
	LogDebug("MixingModule") <<shPtr->product()->size()<<"  pileup objects  added, eventNr "<<eventNr;
        vertexoffset+=shPtr->product()->size();
        crFrame_->setPileupPtr(shPtr);
        crFrame_->addPileups(bcr,const_cast< std::vector<SimVertex> * > (shPtr->product()),eventNr);
      }
    }
    else {
      
      boost::shared_ptr<Wrapper<PCrossingFrame<SimVertex> > const> shPtr = getProductByTag<PCrossingFrame<SimVertex> >(ep, tag_);
      
      if (shPtr){
      	crFrame_->setPileupPtr(shPtr);      	
        secSourceCF_ = const_cast<PCrossingFrame<SimVertex> * >(shPtr->product());
	LogDebug("MixingModule") << "Add PCrossingFrame<SimVertex>,  eventNr " << secSourceCF_->getEventID();    
	
	copyPCrossingFrame(secSourceCF_);      
      }
      else
        LogDebug("MixingModule") << "Could not get the PCrossingFrame<SimVertex>!";		
    }
  }

  template <>
  void MixingWorker<HepMCProduct>::addPileups(const int bcr, const EventPrincipal& ep,unsigned int eventNr,int vertexoffset)
  {
    if (!mixProdStep2_){ 
      // default version
      // HepMCProduct does not come as a vector....
      boost::shared_ptr<Wrapper<HepMCProduct> const> shPtr = getProductByTag<HepMCProduct>(ep, tag_);
        if (shPtr) {
	  LogDebug("MixingModule") <<"HepMC pileup objects  added, eventNr "<<eventNr;
	    crFrame_->setPileupPtr(shPtr);
            crFrame_->addPileups(bcr,const_cast<HepMCProduct*> (shPtr->product()),eventNr);
        }
    }
    else {
      // Mixproduction version: step2
      boost::shared_ptr<Wrapper<PCrossingFrame<HepMCProduct> > const> shPtr = getProductByTag<PCrossingFrame<HepMCProduct> >(ep, tag_);

      if (shPtr){	
        crFrame_->setPileupPtr(shPtr);
        secSourceCF_ = const_cast<PCrossingFrame<HepMCProduct> * >(shPtr->product());
	LogDebug("MixingModule") << "Add PCrossingFrame<HepMCProduct>,  eventNr " << secSourceCF_->getEventID();
	
	copyPCrossingFrame(secSourceCF_);
      }
      else
        LogDebug("MixingModule") << "Could not get the PCrossingFrame<HepMCProduct>!";
    }
  }

  template <>
  void MixingWorker<HepMCProduct>::addSignals(const Event &e)
  { 
    if (mixProdStep2_){
      //HepMC - here the interface is different!!!
      Handle<HepMCProduct>  result_t;
      bool got = e.getByLabel(tagSignal_,result_t);
      if (got) {
        LogDebug("MixingModule") <<" adding HepMCProduct from signal event  with "<<tagSignal_;
        crFrame_->addSignals(result_t.product(),e.id());  
      }
      else LogInfo("MixingModule") <<"!!!!!!! Did not get any signal data for HepMCProduct with "<<tagSignal_;
    }
    else{
      //HepMC - here the interface is different!!!
      Handle<HepMCProduct>  result_t;
      bool got = e.getByLabel(tag_,result_t);
      if (got) {
        LogDebug("MixingModule") <<" adding HepMCProduct from signal event  with "<<tag_;
        crFrame_->addSignals(result_t.product(),e.id());  
      }
      else LogInfo("MixingModule") <<"!!!!!!! Did not get any signal data for HepMCProduct with "<<tag_;

    }
      
  }
  
  template <>
  bool MixingWorker<HepMCProduct>::checkSignal(const Event &e)
      {   
          bool got;
	  InputTag t;
	  
	  Handle<HepMCProduct>  result_t;
	  if (mixProdStep2_){
	     got = e.getByLabel(tagSignal_,result_t);
	     t = InputTag(tagSignal_.label(),tagSignal_.instance());   
	  }
	  else{	     
	     got = e.getByLabel(tag_,result_t);
	     t = InputTag(tag_.label(),tag_.instance());
	  }
	  
	  if (got)
	       LogInfo("MixingModule") <<" Will create a CrossingFrame for HepMCProduct with "
	  			       << " with InputTag= "<< t.encode();
				       
	  return got;
  }
      
}//namespace edm
