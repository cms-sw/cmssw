// -*- C++ -*-
//
// Package:    SimCalorimetry/EcalEBTrigPrimProducers
// Class:      EcalEBTrigPrimPhase2SpikeTaggerESProducer
//
/**\class EcalEBTrigPrimPhase2SpikeTaggerESProducer

 Description: Produces the configuration parameters for the TP spike tagger algorithms

*/

// system include files
#include <iostream>
#include <memory>
#include <string>

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/DataRecord/interface/EcalEBPhase2TPGSpikeTaggerParamsRcd.h"
#include "SimCalorimetry/EcalEBTrigPrimAlgos/interface/EcalEBPhase2TPGSpikeTaggerParamsHelper.h"

class EcalEBTrigPrimPhase2SpikeTaggerESProducer : public edm::ESProducer {
public:
  EcalEBTrigPrimPhase2SpikeTaggerESProducer(const edm::ParameterSet&);

  using ReturnType = std::unique_ptr<EcalEBPhase2TPGSpikeTaggerParams>;

  ReturnType produce(const EcalEBPhase2TPGSpikeTaggerParamsRcd&);
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  EcalEBPhase2TPGSpikeTaggerParams params_;
};

EcalEBTrigPrimPhase2SpikeTaggerESProducer::EcalEBTrigPrimPhase2SpikeTaggerESProducer(const edm::ParameterSet& iConfig) {
  setWhatProduced(this);

  EcalEBPhase2TPGSpikeTaggerParamsHelper paramsHelper(iConfig);
  params_ = static_cast<EcalEBPhase2TPGSpikeTaggerParams>(paramsHelper);
}

// ------------ method called to produce the data  ------------
EcalEBTrigPrimPhase2SpikeTaggerESProducer::ReturnType EcalEBTrigPrimPhase2SpikeTaggerESProducer::produce(
    const EcalEBPhase2TPGSpikeTaggerParamsRcd& iRecord) {
  auto product = std::make_unique<EcalEBPhase2TPGSpikeTaggerParams>(params_);
  return product;
}

void EcalEBTrigPrimPhase2SpikeTaggerESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<unsigned int>("fwVersion", 1);

  // algoConfigs VPSet
  edm::ParameterSetDescription algoDefaults;
  algoDefaults.add<std::string>("algo", "ld");

  // perCrystalParams VPSet within algoConfigs
  edm::ParameterSetDescription perXtalDefaults;
  perXtalDefaults.add<std::string>("ietaRange", ":");
  perXtalDefaults.add<std::string>("iphiRange", ":");
  perXtalDefaults.add<unsigned int>("peakSampleIndex", 5);
  perXtalDefaults.add<double>("spikeThreshold", -0.1);
  perXtalDefaults.add<std::vector<double>>("weights",
                                           {
                                               1.5173,
                                               -2.1034,
                                               1.8117,
                                               -0.6451,
                                           });
  std::vector<edm::ParameterSet> perXtals;
  perXtals.emplace_back();
  algoDefaults.addVPSet("perCrystalParams", perXtalDefaults, perXtals);

  std::vector<edm::ParameterSet> algos;
  algos.emplace_back();
  desc.addVPSet("algoConfigs", algoDefaults, algos);

  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_EVENTSETUP_MODULE(EcalEBTrigPrimPhase2SpikeTaggerESProducer);
