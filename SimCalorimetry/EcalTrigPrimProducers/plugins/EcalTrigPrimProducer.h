#ifndef EcaltrigprimProducer_h
#define EcaltrigprimProducer_h
  
/** \class EcalTrigPrimProducer
 *
 * EcalTrigPrimProducer produces a EcalTrigPrimDigiCollection
 * Simulation as close as possible to hardware
 * Main algorithm is EcalTrigPrimFunctionalAlgo which is now
 * templated to take EBdataFrames/EEDataFrames as input
 *
 * \author Ursula Berthon, Stephanie Baffioni, Pascal Paganini,   LLR Palaiseau
 *
 * \version   1st Version may 2006
 * \version   2nd Version jul 2006
 * \version   3rd Version nov 2006
 * \version   4th Version apr 2007   full endcap
 *
 ************************************************************/

#include <memory>
 
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
 
#include "DataFormats/Common/interface/Handle.h"
 
#include "FWCore/ParameterSet/interface/ParameterSet.h"
  
class EcalTrigPrimFunctionalAlgo;
 
class EcalTrigPrimProducer : public edm::stream::EDProducer<>
{
 public:
  
  explicit EcalTrigPrimProducer(const edm::ParameterSet& conf);
  
  virtual ~EcalTrigPrimProducer();
  
  void beginRun(const edm::Run& run, const edm::EventSetup& es) override;
  void endRun(const edm::Run&, const edm::EventSetup&) override;
  void produce(edm::Event& e, const edm::EventSetup& c) override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  
 private:
  std::unique_ptr<EcalTrigPrimFunctionalAlgo> algo_;
  bool barrelOnly_;
  bool tcpFormat_;
  bool debug_;
  bool famos_;
  std::string label_;
  std::string instanceNameEB_;
  std::string instanceNameEE_;

  int binOfMaximum_;
  bool fillBinOfMaximumFromHistory_;

  //method to get EventSetupRecords
  unsigned long long getRecords(edm::EventSetup const& setup);
  unsigned long long cacheID_;
};
  
#endif
 


