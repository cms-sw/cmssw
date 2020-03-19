// -*- C++ -*-
//
// Package:    SimMuon/Neutron
// Class:      EmptyHepMCProducer
//
/**\class EmptyHepMCProducer EmptyHepMCProducer.cc SimMuon/Neutron/plugins/EmptyHepMCProducer.cc

 Description:

 Utility for neutron simulation framework.
    * Creates an empty HepMCProduct to make MixingModule happy

*/
//
// Original Author:  Vadim Khotilovich
//         Created:  Mon Aug 09 19:11:42 CST 2010
//
//

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

class EmptyHepMCProducer : public edm::stream::EDProducer<> {
public:
  explicit EmptyHepMCProducer(const edm::ParameterSet&);
  ~EmptyHepMCProducer() override{};

private:
  virtual void beginJob();
  void produce(edm::Event&, const edm::EventSetup&) override;
  virtual void endJob();
};

EmptyHepMCProducer::EmptyHepMCProducer(const edm::ParameterSet& iConfig) { produces<edm::HepMCProduct>(); }

void EmptyHepMCProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // create an empty output collection
  std::unique_ptr<edm::HepMCProduct> theOutput(new edm::HepMCProduct());
  //theOutput->addHepMCData(theEvent);
  iEvent.put(std::move(theOutput));
}

void EmptyHepMCProducer::beginJob() {}

void EmptyHepMCProducer::endJob() {}

//define this as a plug-in
DEFINE_FWK_MODULE(EmptyHepMCProducer);
