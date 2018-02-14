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

#include "DQMServices/Core/interface/ConcurrentMonitorElement.h"
#include "DQMServices/Core/interface/DQMGlobalEDAnalyzer.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment2DCollection.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}

class HRes2DHit;
class HEff2DHit;

namespace {
  struct Histograms {
    HRes2DHit *h2DHitRPhi;
    HRes2DHit *h2DHitRZ;
    HRes2DHit *h2DHitRZ_W0;
    HRes2DHit *h2DHitRZ_W1;
    HRes2DHit *h2DHitRZ_W2;

    HEff2DHit *h2DHitEff_RPhi;
    HEff2DHit *h2DHitEff_RZ;
    HEff2DHit *h2DHitEff_RZ_W0;
    HEff2DHit *h2DHitEff_RZ_W1;
    HEff2DHit *h2DHitEff_RZ_W2;
  };
}

class DTSegment2DQuality : public DQMGlobalEDAnalyzer<Histograms> {
public:
  /// Constructor
  DTSegment2DQuality(const edm::ParameterSet& pset);

private:
  /// Book the DQM plots
  void bookHistograms(DQMStore::ConcurrentBooker &, edm::Run const&, edm::EventSetup const&, Histograms &) const override;

  /// Perform the real analysis
  void dqmAnalyze(edm::Event const&, edm::EventSetup const&, Histograms const&) const override;

private:
  // Labels to read from event
  edm::InputTag simHitLabel_;
  edm::InputTag segment2DLabel_;
  edm::EDGetTokenT<edm::PSimHitContainer> simHitToken_;
  edm::EDGetTokenT<DTRecSegment2DCollection> segment2DToken_;

  // Sigma resolution on position
  double sigmaResPos_;

  // Sigma resolution on angle
  double sigmaResAngle_;

  // Switch for debug output
  bool debug_;
};

#endif // Validation_DTRecHits_DTSegment2DQuality_h
