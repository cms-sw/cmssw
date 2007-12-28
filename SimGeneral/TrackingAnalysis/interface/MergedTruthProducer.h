#ifndef TrackingAnalysis_MergedTruthProducer_h
#define TrackingAnalysis_MergedTruthProducer_h

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <string>

class MergedTruthProducer : public edm::EDProducer {

public:
  explicit MergedTruthProducer( const edm::ParameterSet & );
//  ~TrackingTruthProducer() { TimingReport::current()->dump(std::cout); }

private:
  void produce( edm::Event &, const edm::EventSetup & );

  edm::ParameterSet conf_;
  std::string MessageCategory_;
};

#endif
