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

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "CondFormats/DataRecord/interface/MixingRcd.h"
#include "CondFormats/RunInfo/interface/MixingModuleConfig.h"

#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/PCrossingFrame.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "FWCore/Utilities/interface/InputTag.h" 

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
      explicit MixingWorker() :
        minBunch_(-5),
        maxBunch_(3),
        bunchSpace_(75),
        subdet_(std::string(" ")),
        label_(std::string(" ")),
        labelCF_(std::string(" ")),
        maxNbSources_(5) {
	    tag_=InputTag();
	    tagSignal_=InputTag();
	}

      /*Normal constructor*/ 
      MixingWorker(int minBunch,int maxBunch, int bunchSpace,
		   std::string subdet,std::string label,
		   std::string labelCF,int maxNbSources, InputTag& tag,
		   InputTag& tagCF):
	MixingWorkerBase(),
	minBunch_(minBunch),
	maxBunch_(maxBunch),
	bunchSpace_(bunchSpace),
	subdet_(subdet),
	label_(label),
	labelCF_(labelCF),
	maxNbSources_(maxNbSources),
	tag_(tag),
	tagSignal_(tagCF)
	{
	}

      /**Default destructor*/
      virtual ~MixingWorker() {;}

    public:

      virtual void reload(const edm::EventSetup & setup){
	//get the required parameters from DB.
	// watch the label/tag
	edm::ESHandle<MixingModuleConfig> config;
	setup.get<MixingRcd>().get(config);
	minBunch_=config->minBunch();
	maxBunch_=config->maxBunch();
	bunchSpace_=config->bunchSpace();
      }

      virtual bool checkSignal(const edm::Event &e){
          bool got;
	  InputTag t;
	  edm::Handle<std::vector<T> >  result_t;
	  got = e.getByLabel(tag_,result_t);
	  t = InputTag(tag_.label(),tag_.instance());
	  
	  if (got)
	       LogInfo("MixingModule") <<" Will create a CrossingFrame for "<< typeid(T).name() 
	  			       << " with InputTag= "<< t.encode();
				       
	  return got;
      }
      
      
      virtual void createnewEDProduct(){        
          crFrame_=new CrossingFrame<T>(minBunch_,maxBunch_,bunchSpace_,subdet_,maxNbSources_);
      }
           
      virtual void addSignals(const edm::Event &e){
	edm::Handle<std::vector<T> > result_t;
	bool got = e.getByLabel(tag_,result_t);
	if (got) {
	  LogDebug("MixingModule") <<" adding " << result_t.product()->size()<<" signal objects for "<<typeid(T).name()<<" with "<<tag_;
	  crFrame_->addSignals(result_t.product(),e.id());
	} else {
          LogInfo("MixingModule") <<"!!!!!!! Did not get any signal data for "<<typeid(T).name()<<", with "<<tag_;
        }
      }


      virtual void addPileups(const int bcr, const EventPrincipal &ep, unsigned int eventNr,int vertexoffset);

      virtual void setBcrOffset() {crFrame_->setBcrOffset();}
      virtual void setSourceOffset(const unsigned int s) {crFrame_->setSourceOffset(s);}

      void setTof();

      virtual void put(edm::Event &e) {	
        std::auto_ptr<CrossingFrame<T> > pOut(crFrame_);
	e.put(pOut,label_);
	LogDebug("MixingModule") <<" CF was put for type "<<typeid(T).name()<<" with "<<label_;
      }


      // When using mixed secondary source 
      // Copy the data from the PCrossingFrame to the CrossingFrame
      virtual void copyPCrossingFrame(const PCrossingFrame<T> *PCF);
      
    private:
      int minBunch_;
      int maxBunch_;
      int bunchSpace_;
      std::string const subdet_;
      std::string const label_;
      std::string const labelCF_;
      unsigned int const maxNbSources_;
      InputTag tag_;
      InputTag tagSignal_;

      CrossingFrame<T> * crFrame_;
      PCrossingFrame<T> * secSourceCF_;
    };

//=============== template specializations ====================================================================================
    
template <>
    void MixingWorker<PCaloHit>::addPileups(const int bcr, const EventPrincipal &ep, unsigned int eventNr,int vertexoffset);

template <>
    void MixingWorker<PSimHit>::addPileups(const int bcr, const EventPrincipal &ep, unsigned int eventNr,int vertexoffset);

template <>
    void MixingWorker<SimTrack>::addPileups(const int bcr, const EventPrincipal &ep, unsigned int eventNr,int vertexoffset);

template <>
    void MixingWorker<SimVertex>::addPileups(const int bcr, const EventPrincipal& ep, unsigned int eventNr,int vertexoffset);

template <>
    void MixingWorker<HepMCProduct>::addPileups(const int bcr, const EventPrincipal &ep, unsigned int eventNr,int vertexoffset);

template <class T>
    void MixingWorker<T>::setTof() {;}

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
