#ifndef ME0Digitizer_ME0DigiProducer_h
#define ME0Digitizer_ME0DigiProducer_h

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimMuon/GEMDigitizer/interface/ME0DigiModel.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/StripDigiSimLink.h"
#include "DataFormats/Common/interface/DetSetVector.h"

#include "string"

class ME0Geometry;

class ME0DigiProducer : public edm::EDProducer
{
public:

  typedef edm::DetSetVector<StripDigiSimLink> StripDigiSimLinks;

  explicit ME0DigiProducer(const edm::ParameterSet& ps);

  virtual ~ME0DigiProducer();

  virtual void beginRun( edm::Run&, const edm::EventSetup& ) {};

  virtual void produce(edm::Event& e, const edm::EventSetup& c);

  virtual void endRun( edm::Run&, const edm::EventSetup& ) {}

private:

  //Name of Collection used for create the XF 
  std::string collectionXF_;
  std::string digiModelString_;
  
  ME0DigiModel* me0DigiModel_;

};

#endif

