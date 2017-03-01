#ifndef SimMuon_GEMDigitizer_GEMPadDigiProducer_h
#define SimMuon_GEMDigitizer_GEMPadDigiProducer_h

#include <FWCore/Framework/interface/ConsumesCollector.h>
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"
#include "DataFormats/GEMDigi/interface/GEMPadDigiCollection.h"

class GEMGeometry;

/// \class GEMPadDigiProducer 
/// producer for GEM-CSC trigger pads

class GEMPadDigiProducer : public edm::stream::EDProducer<>
{
public:

  //typedef GEMDigitizer::StripDigiSimLinks StripDigiSimLinks;

  explicit GEMPadDigiProducer(const edm::ParameterSet& ps);

  virtual ~GEMPadDigiProducer();

  virtual void beginRun(const edm::Run&, const edm::EventSetup&) override;

  virtual void produce(edm::Event&, const edm::EventSetup&) override;

private:
  
  void buildPads(const GEMDigiCollection &digis, GEMPadDigiCollection &out_pads);

  /// Name of input digi Collection
  edm::EDGetTokenT<GEMDigiCollection> digi_token_;
  edm::InputTag digis_;

  const GEMGeometry * geometry_;
};

#endif

