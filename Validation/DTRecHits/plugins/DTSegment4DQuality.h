#ifndef Validation_DTRecHits_DTSegment4DQuality_h
#define Validation_DTRecHits_DTSegment4DQuality_h

/** \class DTSegment4DQuality
 *  Basic analyzer class which accesses 4D DTSegments
 *  and plots resolution comparing reconstructed and simulated quantities
 *
 *  Only true 4D segments are considered.
 *  Station 4 segments are not looked at.
 *  FIXME: Add flag to consider also
 *  segments with only phi view? Possible bias?
 *
 *  Residual/pull plots are filled for the reco segment with alpha closest
 *  to the simulated muon direction (defined from muon simhits in the chamber).
 *
 *  Efficiencies are defined as reconstructed 4D segments with alpha, beta, x,
 * y, within 5 sigma relative to the sim muon, with sigmas specified in the
 * config. Note that loss of even only one of the two views is considered as
 * inefficiency!
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

class HRes4DHit;
class HEff4DHit;
namespace dtsegment4d {
  struct Histograms;
}

class DTSegment4DQuality : public DQMGlobalEDAnalyzer<dtsegment4d::Histograms> {
public:
  /// Constructor
  DTSegment4DQuality(const edm::ParameterSet &pset);

private:
  /// Book the DQM plots
  void bookHistograms(DQMStore::IBooker &,
                      edm::Run const &,
                      edm::EventSetup const &,
                      dtsegment4d::Histograms &) const override;

  /// Perform the real analysis
  void dqmAnalyze(edm::Event const &, edm::EventSetup const &, dtsegment4d::Histograms const &) const override;

private:
  // Labels to read from event
  edm::InputTag simHitLabel_;
  edm::InputTag segment4DLabel_;
  edm::EDGetTokenT<edm::PSimHitContainer> simHitToken_;
  edm::EDGetTokenT<DTRecSegment4DCollection> segment4DToken_;

  // Sigma resolution on position
  double sigmaResX_;
  double sigmaResY_;

  // Sigma resolution on angle
  double sigmaResAlpha_;
  double sigmaResBeta_;

  bool doall_;
  bool local_;

  // Switch for debug output
  bool debug_;
};

#endif
