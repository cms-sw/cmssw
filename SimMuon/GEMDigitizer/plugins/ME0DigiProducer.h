#ifndef SimMuon_GEMDigitizer_ME0DigiProducer_h
#define SimMuon_GEMDigitizer_ME0DigiProducer_h

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/StripDigiSimLink.h"
#include "SimDataFormats/GEMDigiSimLink/interface/ME0DigiSimLink.h"

#include <string>

class ME0Geometry;
class ME0DigiModel;

class ME0DigiProducer : public edm::stream::EDProducer<> {
public:
  typedef edm::DetSetVector<StripDigiSimLink> StripDigiSimLinks;

  typedef edm::DetSetVector<ME0DigiSimLink> ME0DigiSimLinks;

  explicit ME0DigiProducer(const edm::ParameterSet& ps);

  ~ME0DigiProducer() override;

  void beginRun(const edm::Run&, const edm::EventSetup&) override;

  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  //Name of Collection used for create the XF
  edm::EDGetTokenT<CrossingFrame<PSimHit> > cf_token;

  std::unique_ptr<ME0DigiModel> ME0DigiModel_;
};

#endif
