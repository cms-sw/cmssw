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
#include "SimDataFormats/CrossingFrame/interface/PCrossingFrame.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "FWCore/ParameterSet/interface/InputTag.h" 

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
      MixingWorker(int minBunch,int maxBunch, int bunchSpace,std::string subdet,std::string label, std::string labelCF,int maxNbSources, InputTag& tag, InputTag& tagCF, bool checktof, bool mixProdStep2, bool isTracker=false):
	MixingWorkerBase(minBunch,maxBunch,bunchSpace,subdet,label,labelCF,maxNbSources,tag,tagCF,checktof,mixProdStep2,isTracker)
	{
          
          trackerHigh_=false;
	  mixProdStep2_ = mixProdStep2;
          if (isTracker) 
	    if (subdet.find("HighTof")!=std::string::npos) 		trackerHigh_=true;
	}

      /**Default destructor*/
      virtual ~MixingWorker() {;}

    public:

      void setTof();

      virtual void put(edm::Event &e) {	
        std::auto_ptr<CrossingFrame<T> > pOut(crFrame_);
	if (!mixProdStep2_){
	  e.put(pOut,label_);
	  LogDebug("MixingModule") <<" CF was put for type "<<typeid(T).name()<<" with "<<label_;
	}
	else {
	  e.put(pOut,labelCF_);
	  LogDebug("MixingModule") <<" CF was put for type "<<typeid(T).name()<<" with "<<labelCF_;
	}
      }

      virtual void createnewEDProduct(const edm::Event &e){        
          bool got;
	  InputTag t;
	  if (mixProdStep2_){
	     edm::Handle<std::vector<T> >  result_t;
	     got = e.getByLabel(tagSignal_,result_t);
	     t = InputTag(tagSignal_.label(),tagSignal_.instance());    
	  }
	  else{
	     edm::Handle<std::vector<T> >  result_t;
	     got = e.getByLabel(tag_,result_t);
	     t = InputTag(tag_.label(),tag_.instance());
	  }
	  
	  // The CrossingFrame is created only if the signal data is present in the root file 
	  if (got){
	     crFrame_=new CrossingFrame<T>(minBunch_,maxBunch_,bunchSpace_,subdet_,maxNbSources_);
	  }
	  else LogInfo("MixingModule") <<" Will not create a CrossingFrame for "<< typeid(T).name() 
	  			       << " with InputTag = "<< t.encode()<<" because there is no signal data! " 
				       << "Please, check if the corresponding branch exists in "
				       << "the signal data file or if it is empty!"; 
      }
            
      virtual void setBcrOffset() {crFrame_->setBcrOffset();}
      virtual void setSourceOffset(const unsigned int s) {crFrame_->setSourceOffset(s);}


      virtual void addSignals(const edm::Event &e){
	if (mixProdStep2_){	  
          edm::Handle<std::vector<T> >  result_t;
	  bool got = e.getByLabel(tagSignal_,result_t);
	  if (got) {
	    LogDebug("MixingModule") <<" adding " << result_t.product()->size()<<" signal objects for "<<typeid(T).name()<<" with "<<tagSignal_;
	    crFrame_->addSignals(result_t.product(),e.id());
	  }
	  else	  LogInfo("MixingModule") <<"!!!!!!! Did not get any signal data for "<<typeid(T).name()<<", with "<<tagSignal_;
        }
	else{
	// Default version
	  edm::Handle<std::vector<T> >  result_t;
	  bool got = e.getByLabel(tag_,result_t);
	  if (got) {
	    LogDebug("MixingModule") <<" adding " << result_t.product()->size()<<" signal objects for "<<typeid(T).name()<<" with "<<tag_;
	    crFrame_->addSignals(result_t.product(),e.id());
	  }
	  else	  LogInfo("MixingModule") <<"!!!!!!! Did not get any signal data for "<<typeid(T).name()<<", with "<<tag_;

	}
	
      }

      virtual void addPileups(const int bcr, EventPrincipal *ep, unsigned int eventNr,int vertexoffset);
      // When using mixed secondary source 
      // Copy the data from the PCrossingFrame to the CrossingFrame
      virtual void copyPCrossingFrame(const PCrossingFrame<T> *PCF);
      
    private:
      CrossingFrame<T> * crFrame_;
      PCrossingFrame<T> * secSourceCF_;
      bool trackerHigh_;
      
      bool mixProdStep2_;
      // limits for tof to be considered for trackers
      static const int lowTrackTof; //nsec
      static const int highTrackTof;
      static const int limHighLowTof;
    };

//=============== template specializations ====================================================================================
  template <class T>
    void MixingWorker<T>::addPileups(const int bcr, EventPrincipal *ep, unsigned int eventNr,int vertexoffset)
    {
      if (!mixProdStep2_){
        // default version
        // valid for CaloHits 
        boost::shared_ptr<Wrapper<std::vector<T> > const> shPtr =
	edm::getProductByTag<std::vector<T> >(*ep, tag_);

        if (shPtr) {
	  LogDebug("MixingModule") <<shPtr->product()->size()<<"  pileup objects  added, eventNr "<<eventNr;
	  crFrame_->addPileups(bcr,const_cast< std::vector<T> * >(shPtr->product()),eventNr);
        }
	
      }
      else
      {
        boost::shared_ptr<Wrapper<PCrossingFrame<T> > const> shPtr = getProductByTag<PCrossingFrame<T> >(*ep, tag_);
     
        if (shPtr){     	      	
          secSourceCF_ = const_cast<PCrossingFrame<T> * >(shPtr->product());
	  LogDebug("MixingModule") << "Add PCrossingFrame<T>  eventNr " << secSourceCF_->getEventID();

	  copyPCrossingFrame(secSourceCF_);

        }
        else
          LogDebug("MixingModule") << "Could not get the PCrossingFrame<T>!";
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

template <class T>
    void MixingWorker<T>::copyPCrossingFrame(const PCrossingFrame<T> *PCF)
    { 
      crFrame_->setBunchRange(PCF->getBunchRange()); 
      crFrame_->setBunchSpace(PCF->getBunchSpace());
      crFrame_->setMaxNbSources(PCF->getMaxNbSources());
      crFrame_->setSubDet(PCF->getSubDet());
      crFrame_->setPileupOffsetsBcr(PCF->getPileupOffsetsBcr());
      crFrame_->setPileupOffsetsSource(PCF->getPileupOffsetsSource());
      crFrame_->setPileups(PCF->getPileups());
      
      // For playback option
      crFrame_->setPileupFileNr(PCF->getPileupFileNr());
      crFrame_->setIdFirstPileup(PCF->getIdFirstPileup());
    }
      
}//edm

#endif
