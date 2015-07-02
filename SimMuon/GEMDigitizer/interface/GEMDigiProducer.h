#ifndef SimMuon_GEMDigitizer_GEMDigiProducer_h
#define SimMuon_GEMDigitizer_GEMDigiProducer_h

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/StripDigiSimLink.h"

#include <string>

class GEMGeometry;
class GEMDigiModel;

class GEMDigiProducer : public edm::stream::EDProducer<>
{
public:

  typedef edm::DetSetVector<StripDigiSimLink> StripDigiSimLinks;

  explicit GEMDigiProducer(const edm::ParameterSet& ps);

  virtual ~GEMDigiProducer();

  virtual void beginRun(const edm::Run&, const edm::EventSetup&) override;

  virtual void produce(edm::Event&, const edm::EventSetup&) override;

private:

  //Name of Collection used for create the XF 
  edm::EDGetTokenT<CrossingFrame<PSimHit> > cf_token; 
  
  std::string digiModelString_;
  GEMDigiModel* gemDigiModel_;
};

#endif

