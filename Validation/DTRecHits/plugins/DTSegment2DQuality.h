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

#include "DQMServices/Core/interface/DQMGlobalEDAnalyzer.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment2DCollection.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}  // namespace edm

class HRes2DHit;
class HEff2DHit;
namespace dtsegment2d {
  struct Histograms;
}

class DTSegment2DQuality : public DQMGlobalEDAnalyzer<dtsegment2d::Histograms> {
public:
  /// Constructor
  DTSegment2DQuality(const edm::ParameterSet &pset);

private:
  /// Book the DQM plots
  void bookHistograms(DQMStore::IBooker &,
                      edm::Run const &,
                      edm::EventSetup const &,
                      dtsegment2d::Histograms &) const override;

  /// Perform the real analysis
  void dqmAnalyze(edm::Event const &, edm::EventSetup const &, dtsegment2d::Histograms const &) const override;

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

#endif  // Validation_DTRecHits_DTSegment2DQuality_h
