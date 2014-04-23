#ifndef HGCalSimProducers_HGCDigitizer_h
#define HGcalSimProducers_HGCDigitizer_h

#include "DataFormats/DetId/interface/DetId.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"

#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"

#include <vector>
#include <map>

class PCaloHit;
class PileUpEventPrincipal;

class HGCDigitizer
{
public:
  
  explicit HGCDigitizer(const edm::ParameterSet& ps);
  virtual ~HGCDigitizer();


  /**
     @short handle SimHit accumulation
   */
  void accumulate(edm::Event const& e, edm::EventSetup const& c);
  void accumulate(PileUpEventPrincipal const& e, edm::EventSetup const& c);
  void accumulate(edm::Handle<edm::PCaloHitContainer> const &hits, int bxCrossing);

  /**
     @short actions at the start/end of event
   */
  void initializeEvent(edm::Event const& e, edm::EventSetup const& c);
  void finalizeEvent(edm::Event& e, edm::EventSetup const& c);


  /**
   */
  bool producesEEDigis()      { return (hitCollection_.find("EE")!=std::string::npos);      } 
  bool producesHEfrontDigis() { return (hitCollection_.find("HEfront")!=std::string::npos); } 
  bool producesHEbackDigis()  { return (hitCollection_.find("HEback")!=std::string::npos);  } 

  /**
      @short actions at the start/end of run
   */
  void beginRun(const edm::EventSetup & es);
  void endRun();

private :

  std::string hitCollection_;

  //flag for trivial digitization
  bool doTrivialDigis_;

  //handle sim hits
  int maxSimHitsAccTime_;
  typedef std::vector<double> CaloSimHitData;
  typedef std::map<uint32_t, CaloSimHitData> CaloSimHitDataAccumulator;
  CaloSimHitDataAccumulator simHitAccumulator_;  
  void resetSimHitDataAccumulator();
  
};

#endif


 
