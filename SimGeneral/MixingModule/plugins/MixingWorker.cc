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
    std::vector<Handle<std::vector<PSimHit> > > result_t;
    e->getMany((*sel_),result_t);//FIXME: getMany
    int str=result_t.size();
    if (str>1) LogWarning("MixingModule") <<"Too many simhit containers, should be only one!";
    if (str>0) {
      LogDebug("MixingModule") <<"For "<<subdet_<<", "<<result_t[0].product()->size()<<"  pileup objects  added, eventNr "<<eventNr;
      crFrame_->addPileups(bcr,result_t[0].product(),eventNr,vertexoffset);
    }
  }else {
    // Tracker treatment
      // do not read branches if clearly outside of tof bounds (and verification is asked for, default)
     // we have to add hits from this subdetector + opposite one

     Handle<std::vector<PSimHit> >  simhitshigh,simhitslow;
     bool gotHigh,gotLow;

     if(trackerHigh_) {
       gotHigh=e->get((*sel_),simhitshigh);
       gotLow=e->get((*opp_),simhitslow);
     }
     else {
       gotHigh=e->get((*opp_),simhitshigh);
       gotLow=e->get((*sel_),simhitslow);
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
  std::vector<Handle<std::vector<SimTrack> > > result_t;
  e->getMany((*sel_),result_t);//FIXME: getMany
  int str=result_t.size();
  if (str>1) LogWarning("MixingModule") <<"Too many track containers, should be only one!";
  if (str>0) {
    LogDebug("MixingModule") <<result_t[0].product()->size()<<"  pileup tracks  added, eventNr "<<eventNr;
    crFrame_->addPileups(bcr,result_t[0].product(),eventNr,vertexoffset);
  }
}

template <>
void MixingWorker<SimVertex>::addPileups(const int bcr, Event* e,unsigned int eventNr,int & vertexoffset)
{
  // default version changed to take care of vertexoffset
  std::vector<Handle<std::vector<SimVertex> > > result;
  e->getMany((*sel_),result);//FIXME: getMany
  int str=result.size();
  if (str>1) LogWarning("MixingModule") <<"Too many vertex containers, should be only one!";
  if (str>0) {
    LogDebug("MixingModule") <<result[0].product()->size()<<"  pileup vertices  added, eventNr "<<eventNr;
    crFrame_->addPileups(bcr,result[0].product(),eventNr);
  }
  vertexoffset+=result[0].product()->size();
}


template <>
void MixingWorker<HepMCProduct>::addPileups(const int bcr, Event* e,unsigned int eventNr,int & vertexoffset)
{
    //HepMC - we are creating a dummy vector, to have the same interfaces as for the other objects
    std::vector<Handle<HepMCProduct> > result_mc;
    e->getMany((*sel_),result_mc);//FIXME: getMany
    int smc=result_mc.size();
    if (smc>1) LogWarning("MixingModule") <<"Too many HepMCProducts, should be only one!"; 
    if (smc>0) {
      LogDebug("MixingModule") <<"  HepMCProduct added";
      std::vector<HepMCProduct> vec;
      vec.push_back(*(result_mc[0].product()));
      crFrame_->addPileups(bcr,&vec,eventNr);
    }

}
