#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "MixingWorker.h"

using namespace edm;
template <> const int  edm::MixingWorker<PSimHit>::lowTrackTof = -36; 
template <> const int  edm::MixingWorker<PSimHit>::highTrackTof = 36; 
template <> const int  edm::MixingWorker<PSimHit>::limHighLowTof = 36; 

template <>
void MixingWorker<PSimHit>::addPileups(const int bcr, edm::Event* e,unsigned int eventNr,int & vertexoffset)
{
  // default version changed for high/low treatment
  if (!isTracker_) {
    Handle<std::vector<PSimHit> >  result_t;
    bool got =e->getByLabel(tag_,result_t);//FIXME: getMany
    if (got) {
      LogDebug("MixingModule") <<"For "<<subdet_<<", "<<result_t.product()->size()<<"  pileup objects  added, eventNr "<<eventNr;
      crFrame_->addPileups(bcr,result_t.product(),eventNr,vertexoffset);
    }
  }else {
    // Tracker treatment
    // do not read branches if clearly outside of tof bounds (and verification is asked for, default)
    // we have to add hits from this subdetector + opposite one

    Handle<std::vector<PSimHit> >  simhitshigh,simhitslow;
    bool gotHigh,gotLow;

    if(trackerHigh_) {
      gotHigh=e->getByLabel(tag_,simhitshigh);
      gotLow=e->getByLabel(opp_,simhitslow);
    }
    else {
      gotHigh=e->getByLabel(opp_,simhitshigh);
      gotLow=e->getByLabel(tag_,simhitslow);
    }

    // add HighTof simhits to high and low signals
    float tof = bcr*crFrame_->getBunchSpace();
    if (gotHigh) {
      if ( !checktof_ || ((limHighLowTof +tof ) <= highTrackTof)) { 
	crFrame_->addPileups(bcr,simhitshigh.product(),eventNr,0,checktof_,trackerHigh_);
	LogDebug("MixingModule") <<"For bcr "<<bcr<<", "<<subdet_<<", evNr "<<eventNr<<", "<<simhitshigh->size()<<" Hits added from high";
      }
    }

    // add LowTof simhits to high and low signals
    if (gotLow) {
      if (  !checktof_ || ((tof+limHighLowTof) >= lowTrackTof && tof <= highTrackTof)) {
	crFrame_->addPileups(bcr,simhitslow.product(),eventNr,0,checktof_,trackerHigh_);
	LogDebug("MixingModule") <<"For bcr "<<bcr<<", "<<subdet_<<", evNr "<<eventNr<<", "<<simhitslow->size()<<" Hits added from low";
      }
    }
  }

}


template <>
void MixingWorker<SimTrack>::addPileups(const int bcr, edm::Event* e,unsigned int eventNr,int & vertexoffset)
{
  // default version changed to transmit vertexoffset
  Handle<std::vector<SimTrack> >  result_t;
  bool got =   e->getByLabel(tag_,result_t);
  if (got) {
    LogDebug("MixingModule") <<result_t.product()->size()<<"  pileup tracks  added, eventNr "<<eventNr;
    crFrame_->addPileups(bcr,result_t.product(),eventNr,vertexoffset);
  }
}

template <>
void MixingWorker<SimVertex>::addPileups(const int bcr, Event* e,unsigned int eventNr,int & vertexoffset)
{
  // default version changed to take care of vertexoffset
  Handle<std::vector<SimVertex> >  result;
  bool got = e->getByLabel(tag_,result);
  if (got) {
    LogDebug("MixingModule") <<result.product()->size()<<"  pileup vertices  added, eventNr "<<eventNr;
    crFrame_->addPileups(bcr,result.product(),eventNr);
    vertexoffset+=result.product()->size();
  }

}


template <>
void MixingWorker<HepMCProduct>::addPileups(const int bcr, Event* e,unsigned int eventNr,int & vertexoffset)
{
  //HepMC - we are creating a dummy vector, to have the same interfaces as for the other objects
  Handle<HepMCProduct>  result_mc;
  bool got = e->getByLabel(tag_,result_mc);
  if (got){
    LogDebug("MixingModule") <<"  HepMCProduct added";
    std::vector<HepMCProduct> vec;
    vec.push_back(*(result_mc.product()));
    crFrame_->addPileups(bcr,&vec,eventNr);
  }


}
