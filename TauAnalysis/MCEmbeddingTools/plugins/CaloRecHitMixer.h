
/** \class CaloRecHitMixer
 *
 * Merge collections of calorimeter recHits
 * for original Zmumu event and "embedded" simulated tau decay products
 * (detectors supported at the moment: EB/EE, HB/HE and HO)
 * 
 * \author Tomasz Maciej Frueboes;
 *         Christian Veelken, LLR
 *
 * \version $Revision: 1.9 $
 *
 * $Id: CaloRecHitMixer.h,v 1.9 2013/03/23 09:12:51 veelken Exp $
 *
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/Common/interface/SortedCollection.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"

#include <string>
#include <iostream>
#include <map>

template <typename T>
struct CaloRecHitMixer_mixedRecHitInfoType
{
  uint32_t rawDetId_;
  
  double energy1_;
  bool isRecHit1_;
  const T* recHit1_;
  
  double energy2_;
  bool isRecHit2_;
  const T* recHit2_;
  
  double energySum_;
  bool isRecHitSum_;
};

template <typename T>
class CaloRecHitMixer : public edm::EDProducer 
{
 public:
  explicit CaloRecHitMixer(const edm::ParameterSet&);
  ~CaloRecHitMixer();

 private:
  virtual void produce(edm::Event&, const edm::EventSetup&);

  std::string moduleLabel_;

  struct todoListEntryType
  {
    edm::InputTag srcRecHitCollection1_;
    bool killNegEnergyBeforeMixing1_;
    edm::InputTag srcRecHitCollection2_;
    bool killNegEnergyBeforeMixing2_;
    enum { kSubtractFromCollection1BeforeMixing, kSubtractFromCollection2BeforeMixing, kSubtractAfterMixing };
    int muonEnSutractionMode_;
    bool killNegEnergyAfterMixing_;
  };
  std::vector<todoListEntryType> todoList_; 

  typedef edm::SortedCollection<T> RecHitCollection;

  typedef std::map<uint32_t, CaloRecHitMixer_mixedRecHitInfoType<T> > detIdToMixedRecHitInfoMap;
  detIdToMixedRecHitInfoMap mixedRecHitInfos_;

  void updateRecHitInfos(const RecHitCollection&, int);
  T buildRecHit(const CaloRecHitMixer_mixedRecHitInfoType<T>&);

  edm::InputTag srcEnergyDepositMapMuPlus_;
  edm::InputTag srcEnergyDepositMapMuMinus_;
  enum { kAbsolute };
  int typeEnergyDepositMap_;

  typedef std::map<uint32_t, float> detIdToFloatMap;

  int verbosity_;
};

template <typename T>
CaloRecHitMixer<T>::CaloRecHitMixer(const edm::ParameterSet& cfg) 
  : moduleLabel_(cfg.getParameter<std::string>("@module_label")),    
    srcEnergyDepositMapMuPlus_(cfg.getParameter<edm::InputTag>("srcEnergyDepositMapMuPlus")),
    srcEnergyDepositMapMuMinus_(cfg.getParameter<edm::InputTag>("srcEnergyDepositMapMuMinus"))
{
  edm::VParameterSet todoList = cfg.getParameter<edm::VParameterSet>("todo");
  if ( todoList.size() == 0 ) {
    throw cms::Exception("Configuration") 
      << "Empty to-do list !!\n";
  }
  
  std::string typeEnergyDepositMap_string = cfg.getParameter<std::string>("typeEnergyDepositMap");
  if ( typeEnergyDepositMap_string == "absolute" ) typeEnergyDepositMap_ = kAbsolute;
  else throw cms::Exception("Configuration") 
    << "Invalid Configuration parameter 'typeEnergyDepositMap' = " << typeEnergyDepositMap_string << " !!\n";
 
  for ( edm::VParameterSet::const_iterator todoItem = todoList.begin();
	todoItem != todoList.end(); ++todoItem ) {
    todoListEntryType todoListEntry;
    todoListEntry.srcRecHitCollection1_ = todoItem->getParameter<edm::InputTag>("collection1");
    todoListEntry.killNegEnergyBeforeMixing1_ = todoItem->getParameter<bool>("killNegEnergyBeforeMixing1");
    todoListEntry.srcRecHitCollection2_ = todoItem->getParameter<edm::InputTag>("collection2");
    todoListEntry.killNegEnergyBeforeMixing2_ = todoItem->getParameter<bool>("killNegEnergyBeforeMixing2");
    std::string muonEnSutractionMode_string = todoItem->getParameter<std::string>("muonEnSutractionMode");
    if       ( muonEnSutractionMode_string == "subtractFromCollection1BeforeMixing" ) todoListEntry.muonEnSutractionMode_ = todoListEntryType::kSubtractFromCollection1BeforeMixing;
    else if  ( muonEnSutractionMode_string == "subtractFromCollection2BeforeMixing" ) todoListEntry.muonEnSutractionMode_ = todoListEntryType::kSubtractFromCollection2BeforeMixing;    
    else if  ( muonEnSutractionMode_string == "subtractAfterMixing"                 ) todoListEntry.muonEnSutractionMode_ = todoListEntryType::kSubtractAfterMixing;
    else throw cms::Exception("Configuration") 
      << "Invalid Configuration parameter 'muonEnSutractionMode' = " << muonEnSutractionMode_string << " !!\n";
    todoListEntry.killNegEnergyAfterMixing_ = todoItem->getParameter<bool>("killNegEnergyAfterMixing");
    
    std::string instanceLabel1 = todoListEntry.srcRecHitCollection1_.instance();
    std::string instanceLabel2 = todoListEntry.srcRecHitCollection2_.instance();
    if ( instanceLabel1 != instanceLabel2 ) {
      throw cms::Exception("Configuration") 
	<< "Mismatch in Instance labels for collection 1 = " << instanceLabel1 << " and 2 = " << instanceLabel2 << " !!\n";
    }
    
    todoList_.push_back(todoListEntry); 
    
    produces<RecHitCollection>(instanceLabel1);
    std::string instanceLabel_removedEnergyMuMinus = "removedEnergyMuMinus";
    if ( instanceLabel1 != "" ) instanceLabel_removedEnergyMuMinus.append("#").append(instanceLabel1);
    produces<double>(instanceLabel_removedEnergyMuMinus.data());
    std::string instanceLabel_removedEnergyMuPlus = "removedEnergyMuPlus";
    if ( instanceLabel1 != "" ) instanceLabel_removedEnergyMuPlus.append("#").append(instanceLabel1);
    produces<double>(instanceLabel_removedEnergyMuPlus.data());
  }  
  
  verbosity_ = ( cfg.exists("verbosity") ) ?
    cfg.getParameter<int>("verbosity") : 0;
}

template <typename T>
CaloRecHitMixer<T>::~CaloRecHitMixer()
{
// nothing to be done yet...  
}

namespace
{
  double getCorrection(uint32_t rawDetId, const std::map<uint32_t, float>& energyDepositMap)
  {
    double correction = 0.;
    std::map<uint32_t, float>::const_iterator energyDepositEntry = energyDepositMap.find(rawDetId);   
    if ( energyDepositEntry != energyDepositMap.end() ) {
      correction = energyDepositEntry->second;
    }
    return correction;
  }
}

template <typename T>
void CaloRecHitMixer<T>::produce(edm::Event& evt, const edm::EventSetup& es)
{
  if ( verbosity_ ) std::cout << "<CaloRecHitMixer::produce (" << moduleLabel_ << ")>:" << std::endl;
  
  edm::Handle<detIdToFloatMap> energyDepositMapMuPlus;
  evt.getByLabel(srcEnergyDepositMapMuPlus_, energyDepositMapMuPlus);
  edm::Handle<detIdToFloatMap> energyDepositMapMuMinus;
  evt.getByLabel(srcEnergyDepositMapMuMinus_, energyDepositMapMuMinus);

  for ( typename std::vector<todoListEntryType>::const_iterator todoItem = todoList_.begin();
	todoItem != todoList_.end(); ++todoItem ) {
    edm::Handle<RecHitCollection> recHitCollection1;
    evt.getByLabel(todoItem->srcRecHitCollection1_, recHitCollection1);
    edm::Handle<RecHitCollection> recHitCollection2;
    evt.getByLabel(todoItem->srcRecHitCollection2_, recHitCollection2);
    if ( verbosity_ ) {
      std::cout << "recHitCollection(input1 = " << todoItem->srcRecHitCollection1_.label() << ":" << todoItem->srcRecHitCollection1_.instance() << ":" << todoItem->srcRecHitCollection1_.process() << "):" 
		<< " #entries = " << recHitCollection1->size() << std::endl;
      std::cout << "recHitCollection(input2 = " << todoItem->srcRecHitCollection2_.label() << ":" << todoItem->srcRecHitCollection2_.instance() << ":" << todoItem->srcRecHitCollection2_.process() << "):" 
		<< " #entries = " << recHitCollection2->size() << std::endl;
    }

    mixedRecHitInfos_.clear();
    updateRecHitInfos(*recHitCollection1, 0);
    updateRecHitInfos(*recHitCollection2, 1);
    
    std::auto_ptr<RecHitCollection> recHitCollection_output(new RecHitCollection());
    std::auto_ptr<double> removedEnergyMuPlus(new double(0.));
    std::auto_ptr<double> removedEnergyMuMinus(new double(0.));

    double muPlusEnergySum  = 0.;
    double muMinusEnergySum = 0.;

    for ( typename detIdToMixedRecHitInfoMap::iterator mixedRecHitInfo = mixedRecHitInfos_.begin();
	  mixedRecHitInfo != mixedRecHitInfos_.end(); ++mixedRecHitInfo ) {
      uint32_t rawDetId = mixedRecHitInfo->second.rawDetId_;

      double muPlusEnergyDeposit  = getCorrection(rawDetId, *energyDepositMapMuPlus);
      double muMinusEnergyDeposit = getCorrection(rawDetId, *energyDepositMapMuMinus);
      double muonEnergyDeposit = muPlusEnergyDeposit + muMinusEnergyDeposit;

      muPlusEnergySum  += muPlusEnergyDeposit;
      muMinusEnergySum += muMinusEnergyDeposit;

      if ( muonEnergyDeposit > 0. ) {
	if ( verbosity_ ) std::cout << "removing muon energy: detId = " << rawDetId << ", subtracted = " << muonEnergyDeposit << std::endl;
	if ( mixedRecHitInfo->second.isRecHit1_ && todoItem->muonEnSutractionMode_ == todoListEntryType::kSubtractFromCollection1BeforeMixing ) {
	  if ( todoItem->killNegEnergyBeforeMixing1_ && mixedRecHitInfo->second.energy1_ < muonEnergyDeposit ) {
	    if ( verbosity_ ) std::cout << "--> killing recHit1: detId = " << rawDetId << ", energy = " << (mixedRecHitInfo->second.energy1_ - muonEnergyDeposit) << std::endl;
	    (*removedEnergyMuPlus) += ((mixedRecHitInfo->second.energy1_/muonEnergyDeposit)*muPlusEnergyDeposit);
	    (*removedEnergyMuMinus) += ((mixedRecHitInfo->second.energy1_/muonEnergyDeposit)*muMinusEnergyDeposit);
	    mixedRecHitInfo->second.energy1_ = 0.;
	    mixedRecHitInfo->second.isRecHit1_ = false;	  
	  } else {
	    (*removedEnergyMuPlus)  += muPlusEnergyDeposit;
	    (*removedEnergyMuMinus) += muMinusEnergyDeposit;
	    mixedRecHitInfo->second.energy1_ -= muonEnergyDeposit;	  
	  }
	}
	if ( mixedRecHitInfo->second.isRecHit2_ && todoItem->muonEnSutractionMode_ == todoListEntryType::kSubtractFromCollection2BeforeMixing ) {
	  if ( todoItem->killNegEnergyBeforeMixing2_ && mixedRecHitInfo->second.energy2_ < muonEnergyDeposit ) {
	    if ( verbosity_ ) std::cout << "--> killing recHit2: detId = " << rawDetId << ", energy = " << (mixedRecHitInfo->second.energy2_ - muonEnergyDeposit) << std::endl;
	    (*removedEnergyMuPlus)  += ((mixedRecHitInfo->second.energy2_/muonEnergyDeposit)*muPlusEnergyDeposit);
	    (*removedEnergyMuMinus) += ((mixedRecHitInfo->second.energy2_/muonEnergyDeposit)*muMinusEnergyDeposit);
	    mixedRecHitInfo->second.energy2_ = 0.;
	    mixedRecHitInfo->second.isRecHit2_ = false;
	  } else {
	    (*removedEnergyMuPlus)  += muPlusEnergyDeposit;
	    (*removedEnergyMuMinus) += muMinusEnergyDeposit;
	    mixedRecHitInfo->second.energy2_ -= muonEnergyDeposit;	  
	  }
	}
      }
      
      mixedRecHitInfo->second.energySum_ = 0.;
      mixedRecHitInfo->second.isRecHitSum_ = false;
      if ( mixedRecHitInfo->second.isRecHit1_ ) {
	mixedRecHitInfo->second.energySum_ += mixedRecHitInfo->second.energy1_;
	mixedRecHitInfo->second.isRecHitSum_ = true;
      }
      if ( mixedRecHitInfo->second.isRecHit2_ ) {
	mixedRecHitInfo->second.energySum_ += mixedRecHitInfo->second.energy2_;
	mixedRecHitInfo->second.isRecHitSum_ = true;
      }
      if ( muonEnergyDeposit > 0. ) {
	if ( mixedRecHitInfo->second.isRecHitSum_ && todoItem->muonEnSutractionMode_ == todoListEntryType::kSubtractAfterMixing ) {
	  if ( todoItem->killNegEnergyAfterMixing_ && mixedRecHitInfo->second.energySum_ < muonEnergyDeposit ) {
	    if ( verbosity_ ) std::cout << "--> killing recHitSum: detId = " << rawDetId << ", energy = " << (mixedRecHitInfo->second.energySum_ - muonEnergyDeposit) << std::endl;
	    (*removedEnergyMuPlus)  += ((mixedRecHitInfo->second.energySum_/muonEnergyDeposit)*muPlusEnergyDeposit);
	    (*removedEnergyMuMinus) += ((mixedRecHitInfo->second.energySum_/muonEnergyDeposit)*muMinusEnergyDeposit);
	    mixedRecHitInfo->second.energySum_ = 0.;
	    mixedRecHitInfo->second.isRecHitSum_ = false;
	  } else {
	    (*removedEnergyMuPlus)  += muPlusEnergyDeposit;
	    (*removedEnergyMuMinus) += muMinusEnergyDeposit;
	    mixedRecHitInfo->second.energySum_ -= muonEnergyDeposit;	  
	  }
	}
      }

      if ( mixedRecHitInfo->second.isRecHitSum_ ) {
	if ( verbosity_ ) {
	  if ( muonEnergyDeposit > 1.e-3 ) std::cout << "--> adding recHitSum (cleaned): detId = " << rawDetId << ", energy = " << mixedRecHitInfo->second.energySum_ << std::endl;
	  else std::cout << "--> adding recHitSum (uncleaned): detId = " << rawDetId << ", energy = " << mixedRecHitInfo->second.energySum_ << std::endl;
	}
	recHitCollection_output->push_back(buildRecHit(mixedRecHitInfo->second));
      }
    }

    if ( verbosity_ ) {
      std::cout << " mu+: sum(EnergyDeposits) = " << muPlusEnergySum << " (removed = " << (*removedEnergyMuPlus) << ")" << std::endl;
      std::cout << " mu-: sum(EnergyDeposits) = " << muMinusEnergySum << " (removed = " << (*removedEnergyMuMinus) << ")" << std::endl;
      std::cout << "recHitCollection(output = " << moduleLabel_ << ":" << todoItem->srcRecHitCollection1_.instance() << "): #entries = " << recHitCollection_output->size() << std::endl;
    }    

    std::string instanceLabel = todoItem->srcRecHitCollection1_.instance();
    evt.put(recHitCollection_output, instanceLabel);
    std::string instanceLabel_removedEnergyMuMinus = "removedEnergyMuMinus";
    if ( instanceLabel != "" ) instanceLabel_removedEnergyMuMinus.append("#").append(instanceLabel);
    evt.put(removedEnergyMuMinus, instanceLabel_removedEnergyMuMinus.data());
    std::string instanceLabel_removedEnergyMuPlus = "removedEnergyMuPlus";
    if ( instanceLabel != "" ) instanceLabel_removedEnergyMuPlus.append("#").append(instanceLabel);
    evt.put(removedEnergyMuPlus, instanceLabel_removedEnergyMuPlus.data());
  }
}

template <typename T>
void CaloRecHitMixer<T>::updateRecHitInfos(const RecHitCollection& recHitCollection, int idx)
{
  for ( typename RecHitCollection::const_iterator recHit = recHitCollection.begin();
	recHit != recHitCollection.end(); ++recHit ) {
    uint32_t rawDetId = recHit->detid().rawId();

    bool isNewRecHit = (mixedRecHitInfos_.find(rawDetId) == mixedRecHitInfos_.end());
    if ( isNewRecHit ) {
      CaloRecHitMixer_mixedRecHitInfoType<T> mixedRecHitInfo;
      mixedRecHitInfo.rawDetId_    = rawDetId;
      mixedRecHitInfo.energy1_     = 0.;
      mixedRecHitInfo.isRecHit1_   = false;
      mixedRecHitInfo.recHit1_     = 0;
      mixedRecHitInfo.energy2_     = 0.;
      mixedRecHitInfo.isRecHit2_   = false;
      mixedRecHitInfo.recHit2_     = 0;
      mixedRecHitInfo.energySum_   = 0.;
      mixedRecHitInfo.isRecHitSum_ = false;
      mixedRecHitInfos_.insert(std::pair<uint32_t, CaloRecHitMixer_mixedRecHitInfoType<T> >(rawDetId, mixedRecHitInfo));
    } 
    
    typename detIdToMixedRecHitInfoMap::iterator mixedRecHitInfo = mixedRecHitInfos_.find(rawDetId);
    assert(mixedRecHitInfo != mixedRecHitInfos_.end());

    if ( verbosity_ ) {
      if ( isNewRecHit ) std::cout << "creating new recHit: detId = " << rawDetId << ", energy = " << recHit->energy() << std::endl;
      else std::cout << "merging recHits: detId = " << rawDetId << ", total energy = " << (mixedRecHitInfo->second.energy1_ + mixedRecHitInfo->second.energy2_) 
		     << " (added = " << recHit->energy() << ")" << std::endl;
    }

    if ( idx == 0 ) {
      mixedRecHitInfo->second.energy1_   = recHit->energy();
      mixedRecHitInfo->second.isRecHit1_ = true;
      mixedRecHitInfo->second.recHit1_   = &(*recHit);
    } else if ( idx == 1 ) {
      mixedRecHitInfo->second.energy2_   = recHit->energy();
      mixedRecHitInfo->second.isRecHit2_ = true;
      mixedRecHitInfo->second.recHit2_   = &(*recHit);
    } else assert(0);
  }
}
