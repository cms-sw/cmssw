#include "Validation/EventGenerator/interface/WeightManager.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Handle.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"


using namespace edm;

WeightManager::WeightManager(const ParameterSet& iConfig):
_useHepMC(iConfig.getParameter<bool>("UseWeightFromHepMC"))
{
  if (_useHepMC)
    _hepmcCollection = iConfig.getParameter<InputTag>("hepmcCollection");
  else  
    _genEventInfos = iConfig.getParameter<std::vector<InputTag> >("genEventInfos");
}

double WeightManager::weight(const Event& iEvent){
  if (_useHepMC){
    edm::Handle<HepMCProduct> evt;
    iEvent.getByLabel(_hepmcCollection, evt);
    const HepMC::GenEvent *myGenEvent = evt->GetEvent();

    double weight = 1.;
    if (myGenEvent->weights().size() > 0)
      weight = myGenEvent->weights()[0];
    return weight;
  } else {
    double weight = 1.;
    for (unsigned int i = 0; i < _genEventInfos.size(); ++i){
      edm::Handle<GenEventInfoProduct> info;
      iEvent.getByLabel(_genEventInfos[i], info);
      weight *= info->weight();
    }
    return weight;
  }
}
