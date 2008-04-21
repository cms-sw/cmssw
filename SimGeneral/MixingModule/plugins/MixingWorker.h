#ifndef MixingWorker_h
#define MixingWorker_h

/** \class MixingWorker
 *
 * MixingWorker is an auxiliary class for the MixingModule
 *
 * \author Ursula Berthon, LLR Palaiseau
 *
 * \version   1st Version JMarch 2008

 *
 ************************************************************/

#include "FWCore/Framework/interface/Event.h"

#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "DataFormats/Common/interface/Handle.h"

#include <vector>
#include <string>
#include <typeinfo>
#include "MixingWorkerBase.h"
#include "MixingModule.h"

namespace edm
{
  template <class T> 
    class MixingWorker: public MixingWorkerBase 
    {
    public:

      /** standard constructor*/
      explicit MixingWorker() {;}

      /*Normal constructor*/ 
      MixingWorker(int minBunch,int maxBunch, int bunchSpace,std::string subdet,int maxNbSources,Selector *sel, bool isTracker=false):
	MixingWorkerBase(minBunch,maxBunch,bunchSpace,subdet,maxNbSources, sel,isTracker)
	{

          trackerHigh_=false;
          if (isTracker) {
          int res = subdet.find("HighTof");
	    if (res>0) {
	      trackerHigh_=true;
	    } 
	  }
	}

      /**Default destructor*/
      virtual ~MixingWorker() {;}

    public:

      virtual void put(edm::Event &e) {
	std::auto_ptr<CrossingFrame<T> > pOut(crFrame_);
	e.put(pOut,subdet_);
      }

      virtual void createnewEDProduct(){
	crFrame_=new CrossingFrame<T>(minBunch_,maxBunch_,bunchSpace_,subdet_,maxNbSources_);
      }

      virtual void addSignals(const edm::Event &e){
	// default version
	std::vector<edm::Handle<std::vector<T> > > result_t;
	e.getMany((*sel_),result_t);
	int str=result_t.size();
	if (str>1) LogWarning("MixingModule") << " Found "<<str<<" collections in signal file, only first one will be stored!!!!!!";
	if (str>0) {
	  edm::BranchDescription desc =result_t[0].provenance()->product();
	  LogDebug("MixingModule") <<" adding " << result_t[0].product()->size()<<" signal objects";
	  crFrame_->addSignals(result_t[0].product(),e.id());
	}
      }

      virtual void addPileups(const int bcr, edm::Event* e,unsigned int eventNr,int &vertexoffset)
	{
	  // default version
	  // valid for CaloHits
	  std::vector<edm::Handle<std::vector<T> > > result_t;
	  e->getMany((*sel_),result_t);
	  int str=result_t.size();
	  if (str>1) LogWarning("MixingModule") <<"Too many containers, should be only one!";
	  if (str>0) {
	    LogDebug("MixingModule") <<result_t[0].product()->size()<<"  pileup objects  added, eventNr "<<eventNr;
	    crFrame_->addPileups(bcr,result_t[0].product(),eventNr);
	  }
	}
      virtual void setBcrOffset() {crFrame_->setBcrOffset();}
      virtual void setSourceOffset(const unsigned int s) {crFrame_->setSourceOffset(s);}

    private:
      CrossingFrame<T> * crFrame_;
      bool trackerHigh_;
      // limits for tof to be considered for trackers
      static const int lowTrackTof; //nsec
      static const int highTrackTof;
      static const int limHighLowTof;
    };
 }//edm

#endif
