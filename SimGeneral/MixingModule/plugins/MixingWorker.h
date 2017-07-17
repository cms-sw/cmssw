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

#include <memory>
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
        maxNbSources_(5),
	tag_(),
	tagSignal_(),
        allTags_(),
        crFrame_(nullptr),
        secSourceCF_(nullptr)
        {
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
	tagSignal_(tagCF),
        allTags_(),
        crFrame_(nullptr),
        secSourceCF_(nullptr)
	{
	}

       /*constructor for HepMCproduct case*/  
      MixingWorker(int minBunch,int maxBunch, int bunchSpace,
		   std::string subdet,std::string label,
		   std::string labelCF,int maxNbSources, InputTag& tag,
		   InputTag& tagCF,
		   std::vector<InputTag> const& tags) : 
	MixingWorkerBase(),
	minBunch_(minBunch),
	maxBunch_(maxBunch),
	bunchSpace_(bunchSpace),
	subdet_(subdet),
	label_(label),
	labelCF_(labelCF),
	maxNbSources_(maxNbSources),
	tag_(tag),
	tagSignal_(tagCF),
        allTags_(tags),
        crFrame_(nullptr),
        secSourceCF_(nullptr)
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

      virtual void addPileups(const EventPrincipal &ep, ModuleCallingContext const*, unsigned int eventNr);

      virtual void setBcrOffset() {crFrame_->setBcrOffset();}
      virtual void setSourceOffset(const unsigned int s) {crFrame_->setSourceOffset(s);}

      void setTof();

      virtual void put(edm::Event &e) {	
        std::unique_ptr<CrossingFrame<T> > pOut(crFrame_);
	e.put(std::move(pOut),label_);
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
      std::vector<InputTag> allTags_; // for HepMCProduct

      CrossingFrame<T> * crFrame_;
      PCrossingFrame<T> * secSourceCF_;
    };

  template <typename T>
  void  MixingWorker<T>::addPileups(const EventPrincipal &ep, ModuleCallingContext const* mcc, unsigned int eventNr) {
    std::shared_ptr<Wrapper<std::vector<T> > const> shPtr = getProductByTag<std::vector<T> >(ep, tag_, mcc);
    if (shPtr) {
      LogDebug("MixingModule") << shPtr->product()->size() << "  pileup objects  added, eventNr " << eventNr;
      crFrame_->setPileupPtr(shPtr);
      crFrame_->addPileups(*shPtr->product());
    }
  }
//=============== template specializations ====================================================================================

template <>
    void MixingWorker<HepMCProduct>::addPileups(const EventPrincipal &ep, ModuleCallingContext const*, unsigned int eventNr);

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
