#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "MixingWorker.h"

template <> const int  edm::MixingWorker<PSimHit>::lowTrackTof = -36; 
template <> const int  edm::MixingWorker<PSimHit>::highTrackTof = 36; 
template <> const int  edm::MixingWorker<PSimHit>::limHighLowTof = 36; 

namespace edm {
  template <>
  void MixingWorker<PSimHit>::addPileups(const int bcr, EventPrincipal *ep,unsigned int eventNr,int vertexoffset)
  {
    //    default version changed for high/low treatment

    if (!isTracker_) {
      boost::shared_ptr<Wrapper<std::vector<PSimHit> > const> shPtr = 
	getProductByTag<std::vector<PSimHit> >(*ep, tag_);
      if (shPtr) {
	LogDebug("MixingModule") <<shPtr->product()->size()<<"  pileup objects  added, eventNr "<<eventNr;

	crFrame_->addPileups(bcr,const_cast< std::vector<PSimHit> * > (shPtr->product()),eventNr);
      }
    } else {
      boost::shared_ptr<Wrapper<std::vector<PSimHit> > const> shPtrHigh, shPtrLow;
      if(trackerHigh_) {
	shPtrHigh=getProductByTag<std::vector<PSimHit> >(*ep, tag_);
	crFrame_->setPileupPtr(shPtrHigh);
	shPtrLow=edm::getProductByTag<std::vector<PSimHit> >(*ep, opp_);
      }
      else {
	shPtrHigh=getProductByTag<std::vector<PSimHit> >(*ep, opp_);
	shPtrLow=getProductByTag<std::vector<PSimHit> >(*ep, tag_);
	crFrame_->setPileupPtr(shPtrLow);
      }

      // add HighTof simhits to high and low signals
      float tof = bcr*crFrame_->getBunchSpace();
      if (shPtrHigh) {
	if ( !checktof_ || ((limHighLowTof +tof ) <= highTrackTof)) { 
	  crFrame_->addPileups(bcr,const_cast< std::vector<PSimHit> * > (shPtrHigh->product()),eventNr,0,checktof_,trackerHigh_);
	  LogDebug("MixingModule") <<"For bcr "<<bcr<<", "<<subdet_<<", evNr "<<eventNr<<", "<<shPtrHigh->product()->size()<<" Hits added from high";
	}
      }

      // add LowTof simhits to high and low signals
      if (shPtrLow) {
	if (  !checktof_ || ((tof+limHighLowTof) >= lowTrackTof && tof <= highTrackTof)) {
	  crFrame_->addPileups(bcr,const_cast< std::vector<PSimHit> * > (shPtrLow->product()),eventNr,0,checktof_,trackerHigh_);
	  LogDebug("MixingModule") <<"For bcr "<<bcr<<", "<<subdet_<<", evNr "<<eventNr<<", "<<shPtrLow->product()->size()<<" Hits added from low";
	}
      }
    }
  }

  template <>
  void  MixingWorker<SimTrack>::addPileups(const int bcr, EventPrincipal *ep,unsigned int eventNr,int vertexoffset)
  {
    // default version changed to transmit vertexoffset
    boost::shared_ptr<Wrapper<std::vector<SimTrack> > const> shPtr =
      getProductByTag<std::vector<SimTrack> >(*ep, tag_);
    if (shPtr) {
      LogDebug("MixingModule") <<shPtr->product()->size()<<"  pileup objects  added, eventNr "<<eventNr;
      crFrame_->setPileupPtr(shPtr);
      crFrame_->addPileups(bcr,const_cast< std::vector<SimTrack> * > (shPtr->product()),eventNr,vertexoffset);
    }
  }

  template <>
  //  void MixingWorker<SimVertex>::addPileups(const int bcr, Event* e,unsigned int eventNr,int vertexoffset)
  void MixingWorker<SimVertex>::addPileups(const int bcr, EventPrincipal * ep,unsigned int eventNr,int vertexoffset)
  {
    // default version changed to take care of vertexoffset
    boost::shared_ptr<Wrapper<std::vector<SimVertex> > const> shPtr = 
      getProductByTag<std::vector<SimVertex> >(*ep, tag_);
    if (shPtr) {
      LogDebug("MixingModule") <<shPtr->product()->size()<<"  pileup objects  added, eventNr "<<eventNr;
      vertexoffset+=shPtr->product()->size();
      crFrame_->setPileupPtr(shPtr);
      crFrame_->addPileups(bcr,const_cast< std::vector<SimVertex> * > (shPtr->product()),eventNr);
    }
  }

  template <>
  void MixingWorker<HepMCProduct>::addPileups(const int bcr, EventPrincipal *ep,unsigned int eventNr,int vertexoffset)
  {
    // HepMCProduct does not come as a vector....
    boost::shared_ptr<Wrapper<HepMCProduct> const> shPtr = 
      getProductByTag<HepMCProduct>(*ep, tag_);
       if (shPtr) {
           LogDebug("MixingModule") <<"HepMC pileup objects  added, eventNr "<<eventNr;
	   crFrame_->setPileupPtr(shPtr);
           crFrame_->addPileups(bcr,const_cast<HepMCProduct*> (shPtr->product()),eventNr);
       }
  }

  template <>
  void MixingWorker<HepMCProduct>::addSignals(const Event &e)
  {
    //HepMC - here the interface is different!!!
    Handle<HepMCProduct>  result_t;
    bool got = e.getByLabel(tag_,result_t);
    if (got) {
      LogDebug("MixingModule") <<" adding HepMCProduct from signal event  with "<<tag_;
      crFrame_->addSignals(result_t.product(),e.id());  
    }
    else	  LogInfo("MixingModule") <<"!!!!!!! Did not get any signal data for HepMCProduct with "<<tag_;
  }

  template <>
  void MixingWorker<PSimHit>::setTof(){
    crFrame_->setTof();
  }

}//namespace edm
