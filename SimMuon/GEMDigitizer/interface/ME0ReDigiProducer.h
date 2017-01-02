#ifndef SimMuon_GEMDigitizer_ME0ReDigiProducer_h
#define SimMuon_GEMDigitizer_ME0ReDigiProducer_h

/*
 * This module smears and discretizes the timing and position of the 
 * ME0 pseudo digis.
 */

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "DataFormats/GEMDigi/interface/ME0DigiPreRecoCollection.h"

#include <string>

class ME0Geometry;
class ME0EtaPartition;

namespace CLHEP {
  class HepRandomEngine;
}

class ME0ReDigiProducer : public edm::stream::EDProducer<>
{
public:

  explicit ME0ReDigiProducer(const edm::ParameterSet& ps);

  virtual ~ME0ReDigiProducer();

  virtual void beginRun(const edm::Run&, const edm::EventSetup&) override;

  virtual void produce(edm::Event&, const edm::EventSetup&) override;
  
  void buildDigis(const ME0DigiPreRecoCollection &, ME0DigiPreRecoCollection &, CLHEP::HepRandomEngine* engine);
  
private:

  double correctSigmaU(const ME0EtaPartition* roll, double y);

  edm::EDGetTokenT<ME0DigiPreRecoCollection> token_; 
  const ME0Geometry* geometry_;
  double timeResolution_;
  int minBunch_;
  int maxBunch_;
  bool smearTiming_;
  bool discretizeTiming_;
  double radialResolution_;
  bool smearRadial_;
  double oldXResolution_;
  double oldYResolution_;
  double newXResolution_;
  double newYResolution_;
  bool discretizeX_;
  bool discretizeY_;
  bool verbose_;
  bool reDigitizeOnlyMuons_;
  bool reDigitizeNeutronBkg_;
  double instLumiDefault_;
  double rateFact_;
  double instLumi_;
  std::vector<double> centralTOF_;
  int nPartitions_;
};

#endif

