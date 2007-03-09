#ifndef EcaltrigprimProducer_h
#define EcaltrigprimProducer_h
  
/** \class EcalTrigPrimProducer
 *
 * EcalTrigPrimProducer produces a EcalTrigPrimDigiCollection
 * The barrel code does a detailed simulation
 * The code for the endcap is simulated in a rough way, due to missing strip geometry
 *
 *
 * \author Ursula Berthon, Stephanie Baffioni, Pascal Paganini,   LLR Palaiseau
 *
 * \version   1st Version may 2006
 * \version   2nd Version jul 2006
 * \version   3rd Version nov 2006

 *
 ************************************************************/

 
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
 
#include "DataFormats/Common/interface/EDProduct.h"
 
#include "FWCore/ParameterSet/interface/ParameterSet.h"
  
class TFile;
class TTree;
class EcalTrigPrimFunctionalAlgo;
class DBInterface; 
 
class EcalTrigPrimProducer : public edm::EDProducer
{
 public:
  
  explicit EcalTrigPrimProducer(const edm::ParameterSet& conf);
  
  virtual ~EcalTrigPrimProducer();
  
  void beginJob(edm::EventSetup const& setup);

  virtual void produce(edm::Event& e, const edm::EventSetup& c);
  
 private:
  EcalTrigPrimFunctionalAlgo *algo_;
  DBInterface * db_ ;
  TFile *histfile_;
  TTree *valTree_;
  bool valid_;
  bool barrelOnly_;
  bool tcpFormat_;
  bool debug_;
  std::string label_;
  std::string instanceNameEB_;
  std::string instanceNameEE_;
  std::string databaseFileNameEB_;
  std::string databaseFileNameEE_;
  double ebDccAdcToGeV_,eeDccAdcToGeV_;
  //  int fgvbMinEnergy_;
  //  double ttfThreshLow_;
  //  double ttfThreshHigh_;
  int binOfMaximum_;
  //  enum {nrSamples_= 5}; //nr samples to write, should not be changed, if not problems in EcalTriggerPrimitiveDigi class
  static const int nrSamples_; //nr samples to write, should not be changed, if not problems in EcalTriggerPrimitiveDigi class
};
  
#endif
 


