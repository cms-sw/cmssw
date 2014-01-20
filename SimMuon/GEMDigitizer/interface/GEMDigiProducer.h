#ifndef GEMDigitizer_GEMDigiProducer_h
#define GEMDigitizer_GEMDigiProducer_h

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimMuon/GEMDigitizer/interface/GEMDigiModel.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/StripDigiSimLink.h"
#include "DataFormats/Common/interface/DetSetVector.h"

#include "string"

class GEMGeometry;

class GEMDigiProducer : public edm::EDProducer
{
public:

  typedef edm::DetSetVector<StripDigiSimLink> StripDigiSimLinks;

  explicit GEMDigiProducer(const edm::ParameterSet& ps);

  virtual ~GEMDigiProducer();

  virtual void beginRun( edm::Run&, const edm::EventSetup& ) {};

  virtual void produce(edm::Event& e, const edm::EventSetup& c);

  virtual void endRun( edm::Run&, const edm::EventSetup& ) {}

private:

  //Name of Collection used for create the XF 
  std::string collectionXF_;
  std::string digiModelString_;
  
  GEMDigiModel* gemDigiModel_;

};

#endif

