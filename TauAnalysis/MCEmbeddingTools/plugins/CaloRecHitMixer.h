
/** \class CaloRecHitMixer
 *
 * Merge collections of calorimeter recHits
 * for original Zmumu event and "embedded" simulated tau decay products
 * (detectors supported at the moment: EB/EE, HB/HE and HO)
 * 
 * \author Tomasz Maciej Frueboes;
 *         Christian Veelken, LLR
 *
 * \version $Revision: 1.5 $
 *
 * $Id: CaloRecHitMixer.h,v 1.5 2012/10/24 09:37:14 veelken Exp $
 *
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/Common/interface/SortedCollection.h"

#include <string>
#include <map>

template <typename T>
class CaloRecHitMixer : public edm::EDProducer 
{
 public:
  explicit CaloRecHitMixer(const edm::ParameterSet&);
  ~CaloRecHitMixer();

 private:
  virtual void produce(edm::Event&, const edm::EventSetup&);
 
  T cleanRH(const T&, double);
  T mergeRH(const T&, const T&);

  typedef edm::SortedCollection<T> RecHitCollection;

  std::string moduleLabel_;

  edm::VParameterSet todoList_; 

  edm::InputTag srcEnergyDepositMapMuPlus_;
  edm::InputTag srcEnergyDepositMapMuMinus_;
  enum { kAbsolute, kFraction };
  int typeEnergyDepositMap_;

  typedef std::map<uint32_t, float> detIdToFloatMap;

  int verbosity_;
};

template <typename T>
CaloRecHitMixer<T>::CaloRecHitMixer(const edm::ParameterSet& cfg) 
  : moduleLabel_(cfg.getParameter<std::string>("@module_label")),    
    todoList_(cfg.getParameter<edm::VParameterSet>("todo")),
    srcEnergyDepositMapMuPlus_(cfg.getParameter<edm::InputTag>("srcEnergyDepositMapMuPlus")),
    srcEnergyDepositMapMuMinus_(cfg.getParameter<edm::InputTag>("srcEnergyDepositMapMuMinus"))
{
  if ( todoList_.size() == 0 ) {
    throw cms::Exception("Configuration") 
      << "Empty to-do list !!\n";
  }
  
  std::string typeEnergyDepositMap_string = cfg.getParameter<std::string>("typeEnergyDepositMap");
  if       ( typeEnergyDepositMap_string == "absolute" ) typeEnergyDepositMap_ = kAbsolute;
  else if  ( typeEnergyDepositMap_string == "fraction" ) typeEnergyDepositMap_ = kFraction;
  else throw cms::Exception("Configuration") 
    << "Invalid Configiration parameter 'typeEnergyDepositMap' = " << typeEnergyDepositMap_string << " !!\n";
  
  for ( edm::VParameterSet::const_iterator todoItem = todoList_.begin();
	todoItem != todoList_.end(); ++todoItem ) {
    edm::InputTag srcCollection1 = todoItem->getParameter<edm::InputTag>("collection1");
    edm::InputTag srcCollection2 = todoItem->getParameter<edm::InputTag>("collection2");    
    
    std::string instanceLabel1 = srcCollection1.instance();
    std::string instanceLabel2 = srcCollection2.instance();
    if ( instanceLabel1 != instanceLabel2 ) {
      throw cms::Exception("Configuration") 
	<< "Mismatch in Instance labels for collection 1 = " << instanceLabel1 << " and 2 = " << instanceLabel2 << " !!\n";
    }

    if ( instanceLabel1 == "" ) produces<RecHitCollection>();
    else produces<RecHitCollection>(instanceLabel1);
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
  edm::Handle<detIdToFloatMap> energyDepositMapMuPlus;
  evt.getByLabel(srcEnergyDepositMapMuPlus_, energyDepositMapMuPlus);
  edm::Handle<detIdToFloatMap> energyDepositMapMuMinus;
  evt.getByLabel(srcEnergyDepositMapMuMinus_, energyDepositMapMuMinus);

  for ( edm::VParameterSet::const_iterator todoItem = todoList_.begin();
	todoItem != todoList_.end(); ++todoItem ) {
    edm::InputTag srcCollection1 = todoItem->getParameter<edm::InputTag>("collection1");
    edm::InputTag srcCollection2 = todoItem->getParameter<edm::InputTag>("collection2");

    typedef edm::Handle<RecHitCollection> RecHitCollectionHandle;
    std::vector<RecHitCollectionHandle> recHitCollections;

    RecHitCollectionHandle recHitCollection1;
    evt.getByLabel(srcCollection1, recHitCollection1);
    recHitCollections.push_back(recHitCollection1);

    RecHitCollectionHandle recHitCollection2;
    evt.getByLabel(srcCollection2, recHitCollection2);
    recHitCollections.push_back(recHitCollection2);
    
    typedef std::map<uint32_t, T> DetToRecHitMap;
    DetToRecHitMap detToRecHitMap;

    for ( typename std::vector<RecHitCollectionHandle>::const_iterator recHitCollection = recHitCollections.begin();
	  recHitCollection != recHitCollections.end(); ++recHitCollection ) {
      for ( typename RecHitCollection::const_iterator recHit = (*recHitCollection)->begin();
	    recHit != (*recHitCollection)->end(); ++recHit ) {
	uint32_t rawDetId = recHit->detid().rawId();

	bool isInMap = (detToRecHitMap.find(rawDetId) != detToRecHitMap.end());
	if ( !isInMap ) {
	  detToRecHitMap[rawDetId] = (*recHit);
	} else {
	  detToRecHitMap[rawDetId] = mergeRH(detToRecHitMap[rawDetId], *recHit); 
	}
      }
    }

    double muPlusEnergySum  = 0.;
    double muMinusEnergySum = 0.;

    std::auto_ptr<RecHitCollection> recHitCollection_output(new RecHitCollection());
    for ( typename DetToRecHitMap::const_iterator recHit = detToRecHitMap.begin();
	  recHit != detToRecHitMap.end(); ++recHit ) {
      uint32_t rawDetId = recHit->second.detid().rawId();

      double muPlusEnergyDeposit  = getCorrection(rawDetId, *energyDepositMapMuPlus);
      double muMinusEnergyDeposit = getCorrection(rawDetId, *energyDepositMapMuMinus);
      if ( typeEnergyDepositMap_ == kFraction ) {
	muPlusEnergyDeposit  *= recHit->second.energy();
	muMinusEnergyDeposit *= recHit->second.energy();
      }
      muPlusEnergySum  += muPlusEnergyDeposit;
      muMinusEnergySum += muMinusEnergyDeposit;
      
      double muonEnergyDeposit = muPlusEnergyDeposit + muMinusEnergyDeposit;

      if ( muonEnergyDeposit > 1.e-3 ) {
	T cleanedRecHit = cleanRH(recHit->second, muonEnergyDeposit);
	if ( cleanedRecHit.energy() > 0. ) recHitCollection_output->push_back(cleanedRecHit);
      } else {
	recHitCollection_output->push_back(recHit->second);
      }
    }

    std::string instanceLabel = srcCollection1.instance();
    evt.put(recHitCollection_output, instanceLabel); 

    if ( verbosity_ ) {
      std::cout << "<CaloRecHitMixer::produce (" << moduleLabel_ << ":" << instanceLabel << ")>:" << std::endl;
      std::cout << " mu+: sum(EnergyDeposits) = " << muPlusEnergySum << std::endl;
      std::cout << " mu-: sum(EnergyDeposits) = " << muMinusEnergySum << std::endl;
    }    
  }
}

