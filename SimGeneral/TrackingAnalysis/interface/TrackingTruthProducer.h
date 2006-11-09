#ifndef TrackingAnalysis_TrackingTruthProducer_h
#define TrackingAnalysis_TrackingTruthProducer_h

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class TrackingTruthProducer : public edm::EDProducer {

public:
  explicit TrackingTruthProducer( const edm::ParameterSet & );

private:
  void produce( edm::Event &, const edm::EventSetup & );
  int LayerFromDetid(const unsigned int&);

  edm::ParameterSet conf_;
  double                   distanceCut_;
  std::vector<std::string> dataLabels_;
  std::vector<std::string> hitLabelsVector_;
  double                   volumeRadius_;
  double                   volumeZ_;
  bool                     discardOutVolume_;
  bool                     discardHitsFromDeltas_;
  std::string		   simHitLabel_;		   
  
};

#endif
