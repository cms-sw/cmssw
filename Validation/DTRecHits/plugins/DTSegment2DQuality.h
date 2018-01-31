#ifndef Validation_DTRecHits_DTSegment2DQuality_h
#define Validation_DTRecHits_DTSegment2DQuality_h

/** \class DTSegment2DQuality
 *  Basic analyzer class which accesses 2D DTSegments
 *  and plot resolution comparing reconstructed and simulated quantities
 *
 *  \author S. Bolognesi and G. Cerminara - INFN Torino
 */

#include <map>
#include <string>
#include <vector>

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment2DCollection.h"
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

class DTSegment2DQuality : public edm::EDAnalyzer {
public:
  /// Constructor
  DTSegment2DQuality(const edm::ParameterSet& pset);

  /// Perform the real analysis
  void analyze(const edm::Event & event, const edm::EventSetup& eventSetup) override;

  void beginRun(const edm::Run& iRun, const edm::EventSetup &setup) override;

private:
  // Switch for debug output
  bool debug_;

  // Labels to read from event
  edm::InputTag simHitLabel_;
  edm::InputTag segment2DLabel_;
  edm::EDGetTokenT<edm::PSimHitContainer> simHitToken_;
  edm::EDGetTokenT<DTRecSegment2DCollection> segment2DToken_;

  // Sigma resolution on position
  double sigmaResPos_;
  // Sigma resolution on angle
  double sigmaResAngle_;

  HRes2DHit *h2DHitRPhi_;
  HRes2DHit *h2DHitRZ_;
  HRes2DHit *h2DHitRZ_W0_;
  HRes2DHit *h2DHitRZ_W1_;
  HRes2DHit *h2DHitRZ_W2_;

  HEff2DHit *h2DHitEff_RPhi_;
  HEff2DHit *h2DHitEff_RZ_;
  HEff2DHit *h2DHitEff_RZ_W0_;
  HEff2DHit *h2DHitEff_RZ_W1_;
  HEff2DHit *h2DHitEff_RZ_W2_;
  DQMStore* dbe_;
};

#endif // Validation_DTRecHits_DTSegment2DQuality_h
