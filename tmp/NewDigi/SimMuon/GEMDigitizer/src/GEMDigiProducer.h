#ifndef GEMDigitizer_GEMDigiProducer_h
#define GEMDigitizer_GEMDigiProducer_h

#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/StripDigiSimLink.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"

#include <string>

class GEMDigiModel; 
class GEMDetId;
class GEMGeometry;

class GEMDigiProducer : public edm::EDProducer
{
public:

  typedef edm::DetSetVector<StripDigiSimLink> StripDigiSimLinks;
  
  explicit GEMDigiProducer(const edm::ParameterSet&);
  virtual ~GEMDigiProducer();

  virtual void beginRun(edm::Run&, const edm::EventSetup&);

  virtual void produce(edm::Event&, const edm::EventSetup&);

  virtual void endRun(edm::Run&, const edm::EventSetup&) {}

private:
  GEMDigiModel* gemDigiModel_;
  // inputCollection to create the crossing frame
  std::string inputCollection_;
  std::string digiModelString_;
};

#endif

