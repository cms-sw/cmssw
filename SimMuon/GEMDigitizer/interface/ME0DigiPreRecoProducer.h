#ifndef GEMDigitizer_ME0DigiPreRecoProducer_h
#define GEMDigitizer_ME0DigiPreRecoProducer_h

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimMuon/GEMDigitizer/interface/ME0DigiPreRecoModel.h"
#include "DataFormats/Common/interface/DetSetVector.h"

#include "string"

class ME0Geometry;

class ME0DigiPreRecoProducer : public edm::EDProducer
{
public:

  explicit ME0DigiPreRecoProducer(const edm::ParameterSet& ps);

  virtual ~ME0DigiPreRecoProducer();

  virtual void beginRun( edm::Run&, const edm::EventSetup& ) {};

  virtual void produce(edm::Event& e, const edm::EventSetup& c);

  virtual void endRun( edm::Run&, const edm::EventSetup& ) {}

private:

  //Name of Collection used for create the XF 
  std::string collectionXF_;
  std::string digiPreRecoModelString_;
  
  ME0DigiPreRecoModel* me0DigiPreRecoModel_;

};

#endif

