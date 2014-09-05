#ifndef GEMDigitizer_ME0DigiPreRecoProducer_h
#define GEMDigitizer_ME0DigiPreRecoProducer_h

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"

#include <string>

class ME0Geometry;
class ME0DigiPreRecoModel;

class ME0DigiPreRecoProducer : public edm::EDProducer
{
public:

  explicit ME0DigiPreRecoProducer(const edm::ParameterSet& ps);

  virtual ~ME0DigiPreRecoProducer();

  virtual void beginRun(const edm::Run&, const edm::EventSetup&) override;

  virtual void produce(edm::Event&, const edm::EventSetup&) override;

private:

  //Name of Collection used for create the XF 
  edm::EDGetTokenT<CrossingFrame<PSimHit> > cf_token; 

  std::string digiPreRecoModelString_;
  ME0DigiPreRecoModel* me0DigiPreRecoModel_;
};

#endif

