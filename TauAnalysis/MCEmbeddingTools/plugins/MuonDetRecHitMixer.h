
/** \class MuonDetRecHitMixer
 *
 * Merge collections of hits in muon detectors (CSC, DT and RPC)
 * for original Zmumu event and "embedded" simulated tau decay products
 * 
 * \author Christian Veelken, LLR
 *
 * \version $Revision: 1.2 $
 *
 * $Id: MuonDetRecHitMixer.h,v 1.2 2012/12/13 09:52:06 veelken Exp $
 *
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/Common/interface/RangeMap.h"
#include "DataFormats/Common/interface/OwnVector.h"

#include "TauAnalysis/MCEmbeddingTools/interface/embeddingAuxFunctions.h"

#include <string>
#include <vector>
#include <map>

template <typename T1, typename T2>
class MuonDetRecHitMixer : public edm::EDProducer 
{
 public:
  explicit MuonDetRecHitMixer(const edm::ParameterSet&);
  ~MuonDetRecHitMixer();

 private:
  virtual void produce(edm::Event&, const edm::EventSetup&);

  typedef edm::RangeMap<T1, edm::OwnVector<T2> > RecHitCollection;

  typedef std::map<uint32_t, int> detIdToIntMap;

  void addRecHits(std::map<T1, std::vector<T2> >&, const RecHitCollection&, bool, const detIdToIntMap&, const detIdToIntMap&, int&);

  void printHitMapRH(const edm::EventSetup&, const RecHitCollection&);

  uint32_t getRawDetId(const T2&);

  std::string moduleLabel_;

  struct TodoListEntry
  {
    edm::InputTag srcCollection1_;
    bool cleanCollection1_;
    edm::InputTag srcCollection2_;
    bool cleanCollection2_;
  };
  std::vector<TodoListEntry> todoList_; 

  edm::InputTag srcHitMapMuPlus_;
  edm::InputTag srcHitMapMuMinus_;

  int verbosity_;
};

template <typename T1, typename T2>
MuonDetRecHitMixer<T1,T2>::MuonDetRecHitMixer(const edm::ParameterSet& cfg) 
  : moduleLabel_(cfg.getParameter<std::string>("@module_label")),        
    srcHitMapMuPlus_(cfg.getParameter<edm::InputTag>("srcHitMapMuPlus")),
    srcHitMapMuMinus_(cfg.getParameter<edm::InputTag>("srcHitMapMuMinus"))
{
  edm::VParameterSet cfgTodoList = cfg.getParameter<edm::VParameterSet>("todo");
  for ( edm::VParameterSet::const_iterator cfgTodoItem = cfgTodoList.begin();
	cfgTodoItem != cfgTodoList.end(); ++cfgTodoItem ) {
    TodoListEntry todoItem;
    todoItem.srcCollection1_ = cfgTodoItem->getParameter<edm::InputTag>("collection1");
    todoItem.cleanCollection1_ = cfgTodoItem->getParameter<bool>("cleanCollection1");
    todoItem.srcCollection2_ = cfgTodoItem->getParameter<edm::InputTag>("collection2");
    todoItem.cleanCollection2_ = cfgTodoItem->getParameter<bool>("cleanCollection2");

    std::string instanceLabel1 = todoItem.srcCollection1_.instance();
    std::string instanceLabel2 = todoItem.srcCollection2_.instance();
    if ( instanceLabel1 != instanceLabel2 ) {
      throw cms::Exception("Configuration") 
	<< "Mismatch in Instance labels for collection 1 = " << instanceLabel1 << " and 2 = " << instanceLabel2 << " !!\n";
    }
    
    if ( instanceLabel1 == "" ) produces<RecHitCollection>();
    else produces<RecHitCollection>(instanceLabel1);
    
    todoList_.push_back(todoItem);
  }

  if ( todoList_.size() == 0 ) {
    throw cms::Exception("Configuration") 
      << "Empty to-do list !!\n";
  }

  verbosity_ = ( cfg.exists("verbosity") ) ?
    cfg.getParameter<int>("verbosity") : 0;
}

template <typename T1, typename T2>
MuonDetRecHitMixer<T1,T2>::~MuonDetRecHitMixer()
{
// nothing to be done yet...  
}

template <typename T1, typename T2>
void MuonDetRecHitMixer<T1,T2>::produce(edm::Event& evt, const edm::EventSetup& es)
{
  edm::Handle<detIdToIntMap> hitMapMuPlus;
  evt.getByLabel(srcHitMapMuPlus_, hitMapMuPlus);
  edm::Handle<detIdToIntMap> hitMapMuMinus;
  evt.getByLabel(srcHitMapMuMinus_, hitMapMuMinus);

  for ( typename std::vector<TodoListEntry>::const_iterator todoItem = todoList_.begin();
	todoItem != todoList_.end(); ++todoItem ) {

    typedef edm::Handle<RecHitCollection> RecHitCollectionHandle;
    RecHitCollectionHandle recHitCollection1;
    evt.getByLabel(todoItem->srcCollection1_, recHitCollection1);
    RecHitCollectionHandle recHitCollection2;
    evt.getByLabel(todoItem->srcCollection2_, recHitCollection2);
    
    std::map<T1, std::vector<T2> > recHits_output;
    
    int numHits_cleaned = 0;
    addRecHits(recHits_output, *recHitCollection1, todoItem->cleanCollection1_, *hitMapMuPlus, *hitMapMuMinus, numHits_cleaned);
    addRecHits(recHits_output, *recHitCollection2, todoItem->cleanCollection2_, *hitMapMuPlus, *hitMapMuMinus, numHits_cleaned);

    std::auto_ptr<RecHitCollection> recHitCollection_output(new RecHitCollection());
    for ( typename std::map<T1, std::vector<T2> >::const_iterator recHit = recHits_output.begin();
	  recHit != recHits_output.end(); ++recHit ) {
      recHitCollection_output->put(recHit->first, recHit->second.begin(), recHit->second.end());
    }
    std::string instanceLabel = todoItem->srcCollection1_.instance(); 
    if ( verbosity_ ) {
      std::cout << "<MuonDetRecHitMixer::produce>:" << std::endl;   
      std::cout << " #Hits(input1 = " << todoItem->srcCollection1_.label() << ":" << todoItem->srcCollection1_.instance() << ") = " << recHitCollection1->size() << std::endl;
      if ( verbosity_ >= 2 ) printHitMapRH(es, *recHitCollection1);
      std::cout << " #Hits(input2 = " << todoItem->srcCollection2_.label() << ":" << todoItem->srcCollection2_.instance() << ") = " << recHitCollection2->size() << std::endl;
      if ( verbosity_ >= 2 ) printHitMapRH(es, *recHitCollection2);
      std::cout << " #Hits(output = " << moduleLabel_ << ":" << instanceLabel << ") = " << recHitCollection_output->size() << std::endl;
      if ( verbosity_ >= 2 ) printHitMapRH(es, *recHitCollection_output);
      std::cout << " #Hits = " << numHits_cleaned << " removed during cleaning." << std::endl;   
    }    
    evt.put(recHitCollection_output, instanceLabel);
  }
}

namespace
{
  bool matchHitMapRH(uint32_t rawDetId, const std::map<uint32_t, int>& hitMap)
  {
    for ( std::map<uint32_t, int>::const_iterator rh = hitMap.begin();
	  rh != hitMap.end(); ++rh ) {
      if ( matchMuonDetId(rawDetId, rh->first) && rh->second > 0 ) return true;
    }
    return false;
  }
}

template <typename T1, typename T2>
void MuonDetRecHitMixer<T1,T2>::addRecHits(std::map<T1, std::vector<T2> >& recHits_output, const RecHitCollection& recHitCollection_input, 
					   bool applyCleaning, const detIdToIntMap& hitMapMuPlus, const detIdToIntMap& hitMapMuMinus, int& numHits_cleaned)
{
  for ( typename RecHitCollection::const_iterator recHit = recHitCollection_input.begin();
	recHit != recHitCollection_input.end(); ++recHit ) {
    uint32_t rawDetId = getRawDetId(*recHit);

    bool isToBeCleaned = false;
    if ( applyCleaning ) {      
      isToBeCleaned |= matchHitMapRH(rawDetId, hitMapMuPlus);
      isToBeCleaned |= matchHitMapRH(rawDetId, hitMapMuMinus);
    }
      
    if ( !isToBeCleaned ) {
      T1 detId(rawDetId);
      recHits_output[detId].push_back(*recHit);
    } else {
      ++numHits_cleaned;
    }
  }
}

template <typename T1, typename T2>
void MuonDetRecHitMixer<T1,T2>::printHitMapRH(const edm::EventSetup& es, const RecHitCollection& recHits)
{
  std::cout << "detIds:";
  for ( typename RecHitCollection::const_iterator rh = recHits.begin();
	rh != recHits.end(); ++rh ) {
    printMuonDetId(es, getRawDetId(*rh));
  }
}
