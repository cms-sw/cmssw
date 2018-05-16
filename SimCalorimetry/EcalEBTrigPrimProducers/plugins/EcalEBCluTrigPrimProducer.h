#ifndef EcalEBCluTrigPrimProducer_h
#define EcalEBCluTrigPrimProducer_h
  
/** \class EcalEBCluTrigPrimProducer
 *  For Phase II 
 *
 ************************************************************/

#include <memory>
 
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
 
#include "DataFormats/Common/interface/Handle.h"
 
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
  
class EcalEBTrigPrimBaseAlgo;



 
class EcalEBCluTrigPrimProducer : public edm::stream::EDProducer<>
{
 public:
  
  explicit EcalEBCluTrigPrimProducer(const edm::ParameterSet& conf);
  
  virtual ~EcalEBCluTrigPrimProducer();
  

  void beginRun(const edm::Run& run, const edm::EventSetup& es) override;
  void endRun(const edm::Run&, const edm::EventSetup&) override;
  void produce(edm::Event& e, const edm::EventSetup& c) override;

  private:
  std::unique_ptr<EcalEBTrigPrimBaseAlgo> algo_;
  bool barrelOnly_;
  bool tcpFormat_;
  bool debug_;
  bool famos_;
  int  nSamples_;
  int binOfMaximum_;
  int dEta_;
  int dPhi_;
  double hitNoiseCut_;
  double etCutOnSeed_;
  int  nEvent_;
  
  
  edm::EDGetTokenT<EBDigiCollection> tokenEBdigi_;


  bool fillBinOfMaximumFromHistory_;

  unsigned long long getRecords(edm::EventSetup const& setup);
  unsigned long long cacheID_;

};
  
#endif
 


