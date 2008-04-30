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
      MixingWorker(int minBunch,int maxBunch, int bunchSpace,std::string subdet,std::string label, int maxNbSources,InputTag & tag, bool isTracker=false):
	MixingWorkerBase(minBunch,maxBunch,bunchSpace,subdet,label,maxNbSources,tag,isTracker)
	{

          trackerHigh_=false;
          if (isTracker) 
	      if (subdet.find("HighTof")!=std::string::npos) 		trackerHigh_=true;
	}

      /**Default destructor*/
      virtual ~MixingWorker() {;}

    public:

      virtual void put(edm::Event &e) {
        std::auto_ptr<CrossingFrame<T> > pOut(crFrame_);
	e.put(pOut,label_);
      }

      virtual void createnewEDProduct(){
        crFrame_=new CrossingFrame<T>(minBunch_,maxBunch_,bunchSpace_,subdet_,maxNbSources_);//FIXME: subdet not needed in CF
      }

      virtual void addSignals(const edm::Event &e){
	// default version
        edm::Handle<std::vector<T> >  result_t;
	bool got = e.getByLabel(tag_,result_t);
	if (got) {
	  LogDebug("MixingModule") <<" adding " << result_t.product()->size()<<" signal objects for "<<typeid(T).name()<<" with "<<tag_;
	  crFrame_->addSignals(result_t.product(),e.id());
	}
	//	else	  LogWarning("MixingModule") <<"!!!!!!! Did not get any signal data for "<<typeid(T).name()<<", with "<<tag_;
      }

      virtual void addPileups(const int bcr, edm::Event* e,unsigned int eventNr,int &vertexoffset)
	{
	  // default version
	  // valid for CaloHits
	  edm::Handle<std::vector<T> >  result_t;
	  bool got = e->getByLabel(tag_,result_t);
	  if (got) {
	    LogDebug("MixingModule") <<result_t.product()->size()<<"  pileup objects  added, eventNr "<<eventNr;
	    crFrame_->addPileups(bcr,result_t.product(),eventNr);
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
