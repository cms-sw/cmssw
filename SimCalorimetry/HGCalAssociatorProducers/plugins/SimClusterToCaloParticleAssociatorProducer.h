// Author: Felice Pantaleo, felice.pantaleo@cern.ch 07/2024

#ifndef SimCalorimetry_HGCalAssociatorProducers_SimClusterToCaloParticleAssociatorProducer_h
#define SimCalorimetry_HGCalAssociatorProducers_SimClusterToCaloParticleAssociatorProducer_h

#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "SimDataFormats/CaloAnalysis/interface/CaloParticle.h"
#include "SimDataFormats/CaloAnalysis/interface/SimCluster.h"
#include "SimDataFormats/Associations/interface/TICLAssociationMap.h"

class SimClusterToCaloParticleAssociatorProducer : public edm::global::EDProducer<> {
public:
  explicit SimClusterToCaloParticleAssociatorProducer(const edm::ParameterSet &);
  ~SimClusterToCaloParticleAssociatorProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  void produce(edm::StreamID, edm::Event &, const edm::EventSetup &) const override;

  edm::EDGetTokenT<std::vector<SimCluster>> simClusterToken_;
  edm::EDGetTokenT<std::vector<CaloParticle>> caloParticleToken_;
};

#endif
