#ifndef GEMCSCPadDigiProducer_h
#define GEMCSCPadDigiProducer_h

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"
#include "DataFormats/GEMDigi/interface/GEMCSCPadDigiCollection.h"

class GEMGeometry;

/// \class GEMCSCPadDigiProducer 
/// producer for GEM-CSC trigger pads

class GEMCSCPadDigiProducer : public edm::EDProducer
{
public:

  //typedef GEMDigitizer::StripDigiSimLinks StripDigiSimLinks;

  explicit GEMCSCPadDigiProducer(const edm::ParameterSet& ps);
  virtual ~GEMCSCPadDigiProducer();

  virtual void beginRun( const edm::Run&, const edm::EventSetup& ) override;

  /** Produces the EDM products */
  virtual void produce(edm::Event& e, const edm::EventSetup& c) override;

private:
  
  void buildPads(const GEMDigiCollection &digis, GEMCSCPadDigiCollection &out_pads, GEMCSCPadDigiCollection &out_co_pads);

  /// Name of input digi Collection
  edm::InputTag input_;

  /// max allowed BX differentce for pads in a copad;
  /// always use layer1 pad's BX as a copad's BX
  int maxDeltaBX_;

  const GEMGeometry * geometry_;
};

#endif

