#ifndef Validation_DTRecHits_DTSegment2DSLPhiQuality_h
#define Validation_DTRecHits_DTSegment2DSLPhiQuality_h

/** \class DTSegment2DSLPhiQuality
 *  Basic analyzer class which accesses 2D DTSegments reconstructed with both SL
 * Phi and plot resolution comparing reconstructed and simulated quantities
 *
 *  \author S. Bolognesi and G. Cerminara - INFN Torino
 */

#include <map>
#include <string>
#include <vector>

#include "DQMServices/Core/interface/DQMGlobalEDAnalyzer.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}  // namespace edm

class HRes2DHit;
class HEff2DHit;
namespace dtsegment2dsl {
  struct Histograms;
}

class DTSegment2DSLPhiQuality : public DQMGlobalEDAnalyzer<dtsegment2dsl::Histograms> {
public:
  /// Constructor
  DTSegment2DSLPhiQuality(const edm::ParameterSet &pset);

private:
  /// Book the DQM plots
  void bookHistograms(DQMStore::IBooker &,
                      edm::Run const &,
                      edm::EventSetup const &,
                      dtsegment2dsl::Histograms &) const override;

  /// Perform the real analysis
  void dqmAnalyze(edm::Event const &, edm::EventSetup const &, dtsegment2dsl::Histograms const &) const override;

private:
  // Labels to read from event
  edm::InputTag simHitLabel_;
  edm::InputTag segment4DLabel_;
  edm::EDGetTokenT<edm::PSimHitContainer> simHitToken_;
  edm::EDGetTokenT<DTRecSegment4DCollection> segment4DToken_;

  // Sigma resolution on position
  double sigmaResPos_;

  // Sigma resolution on angle
  double sigmaResAngle_;

  bool doall_;
  bool local_;

  // Switch for debug output
  bool debug_;
};

#endif  // Validation_DTRecHits_DTSegment2DSLPhiQuality_h
