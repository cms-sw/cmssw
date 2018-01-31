#ifndef Validation_DTRecHits_DTSegment2DSLPhiQuality_h
#define Validation_DTRecHits_DTSegment2DSLPhiQuality_h

/** \class DTSegment2DSLPhiQuality
 *  Basic analyzer class which accesses 2D DTSegments reconstructed with both SL Phi
 *  and plot resolution comparing reconstructed and simulated quantities
 *
 *  \author S. Bolognesi and G. Cerminara - INFN Torino
 */

#include <map>
#include <string>
#include <vector>

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

#include "Histograms.h"

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}

class DTSegment2DSLPhiQuality : public edm::EDAnalyzer {
public:
  /// Constructor
  DTSegment2DSLPhiQuality(const edm::ParameterSet& pset);

  /// Perform the real analysis
  void analyze(const edm::Event & event, const edm::EventSetup& eventSetup) override;

  void beginRun(const edm::Run& iRun, const edm::EventSetup &setup) override;

private:
  // Switch for debug output
  bool debug_;

  // Labels to read from event
  edm::InputTag simHitLabel_;
  edm::InputTag segment4DLabel_;
  edm::EDGetTokenT<edm::PSimHitContainer> simHitToken_;
  edm::EDGetTokenT<DTRecSegment4DCollection> segment4DToken_;

  // Sigma resolution on position
  double sigmaResPos_;
  // Sigma resolution on angle
  double sigmaResAngle_;

  HRes2DHit *h2DHitSuperPhi_;
  HEff2DHit *h2DHitEff_SuperPhi_;
  DQMStore* dbe_;
  bool doall_;
  bool local_;
};

#endif // Validation_DTRecHits_DTSegment2DSLPhiQuality_h
