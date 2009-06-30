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
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/Handle.h"

#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

#include <vector>
#include <string>
#include <typeinfo>
#include "MixingWorkerBase.h"

class SimTrack;
class SimVertex;
namespace edm
{
  template <class T> 
    class MixingWorker: public MixingWorkerBase 
    {
    public:

      /** standard constructor*/
      explicit MixingWorker() {;}

      /*Normal constructor*/ 
      MixingWorker(int minBunch,int maxBunch, int bunchSpace,std::string subdet,std::string label, int maxNbSources,InputTag& tag,bool checktof, bool isTracker=false):
	MixingWorkerBase(minBunch,maxBunch,bunchSpace,subdet,label,maxNbSources,tag,checktof,isTracker)
	{

          trackerHigh_=false;
          if (isTracker) 
	    if (subdet.find("HighTof")!=std::string::npos) 		trackerHigh_=true;
	}

      /**Default destructor*/
      virtual ~MixingWorker() {;}

    public:

      void setTof();

      virtual void put(edm::Event &e) {
        std::auto_ptr<CrossingFrame<T> > pOut(crFrame_);
	e.put(pOut,label_);
	LogDebug("MixingModule") <<" CF was put for type "<<typeid(T).name()<<" with "<<label_;
      }

      virtual void createnewEDProduct(){
        crFrame_=new CrossingFrame<T>(minBunch_,maxBunch_,bunchSpace_,subdet_,maxNbSources_);
      }
      virtual void setBcrOffset() {crFrame_->setBcrOffset();}
      virtual void setSourceOffset(const unsigned int s) {crFrame_->setSourceOffset(s);}


      virtual void addSignals(const edm::Event &e){
	// default version
        edm::Handle<std::vector<T> >  result_t;
	bool got = e.getByLabel(tag_,result_t);
	if (got) {
	  LogDebug("MixingModule") <<" adding " << result_t.product()->size()<<" signal objects for "<<typeid(T).name()<<" with "<<tag_;
	  crFrame_->addSignals(result_t.product(),e.id());
	}
	else	  LogInfo("MixingModule") <<"!!!!!!! Did not get any signal data for "<<typeid(T).name()<<", with "<<tag_;
      }

      virtual void addPileups(const int bcr, EventPrincipal *ep, unsigned int eventNr,int vertexoffset);

    private:
      CrossingFrame<T> * crFrame_;
      bool trackerHigh_;
      // limits for tof to be considered for trackers
      static const int lowTrackTof; //nsec
      static const int highTrackTof;
      static const int limHighLowTof;
    };

//=============== template specializations ====================================================================================
  template <class T>
    void MixingWorker<T>::addPileups(const int bcr, EventPrincipal *ep, unsigned int eventNr,int vertexoffset)
    {
      // default version
      // valid for CaloHits 
      boost::shared_ptr<Wrapper<std::vector<T> > const> shPtr =
	edm::getProductByTag<std::vector<T> >(*ep, tag_);

      if (shPtr) {
	LogDebug("MixingModule") <<shPtr->product()->size()<<"  pileup objects  added, eventNr "<<eventNr;
	crFrame_->addPileups(bcr,const_cast< std::vector<T> * >(shPtr->product()),eventNr);
      }
    }

  template <>
    void MixingWorker<PSimHit>::addPileups(const int bcr, EventPrincipal *ep, unsigned int eventNr,int vertexoffset);

  template <>
    void MixingWorker<SimTrack>::addPileups(const int bcr, EventPrincipal *ep, unsigned int eventNr,int vertexoffset);

  template <>
      void MixingWorker<SimVertex>::addPileups(const int bcr, EventPrincipal *ep, unsigned int eventNr,int vertexoffset);

template <>
      void MixingWorker<HepMCProduct>::addPileups(const int bcr, EventPrincipal *ep, unsigned int eventNr,int vertexoffset);

template <class T>
  void MixingWorker<T>::setTof() {;}

template <>
void MixingWorker<PSimHit>::setTof();

}//edm

#endif
