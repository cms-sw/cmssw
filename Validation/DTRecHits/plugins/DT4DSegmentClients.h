#ifndef Validation_DTRecHits_DT4DSegmentClients_h
#define Validation_DTRecHits_DT4DSegmentClients_h

/** \class DTDataIntegrityTest
 * *
 *  DQM Client to check the data integrity
 *
 *  \author S. Bolognesi - INFN TO
 *   
 */

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class DT4DSegmentClients: public edm::EDAnalyzer {
public:
  /// Constructor
  DT4DSegmentClients(const edm::ParameterSet& ps);

  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c) override;
  void endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& c) override;
};

#endif // Validation_DTRecHits_DT4DSegmentClients_h
