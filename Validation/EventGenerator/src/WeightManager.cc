#include "Validation/EventGenerator/interface/WeightManager.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Handle.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"

using namespace edm;

WeightManager::WeightManager(const ParameterSet& iConfig, edm::ConsumesCollector iC)
    : _useHepMC(iConfig.getParameter<bool>("UseWeightFromHepMC")) {
  if (_useHepMC) {
    _hepmcCollection = iConfig.getParameter<InputTag>("hepmcCollection");
    hepmcCollectionToken_ = iC.consumes<HepMCProduct>(_hepmcCollection);
  } else {
    _genEventInfos = iConfig.getParameter<std::vector<InputTag> >("genEventInfos");
    for (const auto& _genEventInfo : _genEventInfos)
      genEventInfosTokens_.push_back(iC.consumes<std::vector<InputTag> >(_genEventInfo));
  }
}

double WeightManager::weight(const Event& iEvent) {
  if (_useHepMC) {
    edm::Handle<HepMCProduct> evt;
    iEvent.getByToken(hepmcCollectionToken_, evt);
    const HepMC::GenEvent* myGenEvent = evt->GetEvent();

    double weight = 1.;
    if (!myGenEvent->weights().empty())
      weight = myGenEvent->weights()[0];
    return weight;
  } else {
    double weight = 1.;
    for (auto genEventInfosToken : genEventInfosTokens_) {
      edm::Handle<GenEventInfoProduct> info;
      iEvent.getByToken(genEventInfosToken, info);
      weight *= info->weight();
    }
    return weight;
  }
}
