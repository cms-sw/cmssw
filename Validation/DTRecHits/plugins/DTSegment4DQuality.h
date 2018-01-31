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
 *  Efficiencies are defined as reconstructed 4D segments with alpha, beta, x, y,
 *  within 5 sigma relative to the sim muon, with sigmas specified in the config.
 *  Note that loss of even only one of the two views is considered as inefficiency!
 *
 *  \author S. Bolognesi and G. Cerminara - INFN Torino
 */

#include <map>
#include <string>
#include <vector>

#include "DQMServices/Core/interface/DQMStore.h"
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

class TFile;

class DTSegment4DQuality : public edm::EDAnalyzer {
public:
  /// Constructor
  DTSegment4DQuality(const edm::ParameterSet& pset);

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
  double sigmaResX_;
  double sigmaResY_;
  // Sigma resolution on angle
  double sigmaResAlpha_;
  double sigmaResBeta_;

  HRes4DHit *h4DHit_;
  HRes4DHit *h4DHit_W0_;
  HRes4DHit *h4DHit_W1_;
  HRes4DHit *h4DHit_W2_;
  HRes4DHit *h4DHitWS_[3][4];

  HEff4DHit *hEff_All_;
  HEff4DHit *hEff_W0_;
  HEff4DHit *hEff_W1_;
  HEff4DHit *hEff_W2_;
  HEff4DHit *hEffWS_[3][4];

  DQMStore* dbe_;
  bool doall_;
  bool local_;
};

#endif
