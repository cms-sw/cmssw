#ifndef EcaltrigprimProducer_h
#define EcaltrigprimProducer_h
  
/** \class EcalTrigPrimProducer
 *
 * EcalTrigPrimProducer produces a EcalTrigPrimDigiCollection
 * The barrel code does a detailed simulation
 * The code for the endcap is simulated in a rough way, due to missing strip geometry
 *
 *
 * \author Ursula Berthon, Stephanie Baffioni,  LLR Palaiseau
 *
 * \version   1st Version may 2006
 * \version   2nd Version jul 2006

 *
 ************************************************************/

 
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
 
#include "DataFormats/Common/interface/EDProduct.h"
 
#include "FWCore/ParameterSet/interface/ParameterSet.h"
  
class TFile;
class TTree;
class EcalTrigPrimFunctionalAlgo;
 
class EcalTrigPrimProducer : public edm::EDProducer
{
 public:
  
  explicit EcalTrigPrimProducer(const edm::ParameterSet& conf);
  
  virtual ~EcalTrigPrimProducer();
  
  void beginJob(edm::EventSetup const& setup);

  virtual void produce(edm::Event& e, const edm::EventSetup& c);
  
 private:
  EcalTrigPrimFunctionalAlgo *algo_;
  TFile *histfile_;
  TTree *valTree_;
  bool valid_;
  std::string label_;
  int fgvbMinEnergy_;
  int binOfMaximum_;
  enum {nrSamples_= 5}; //nr samples to write, should not be changed, if not problems in EcalTriggerPrimitiveDigi class
};
  
#endif
 


