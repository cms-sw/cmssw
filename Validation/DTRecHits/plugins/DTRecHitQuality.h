#ifndef Validation_DTRecHits_DTRecHitQuality_h
#define Validation_DTRecHits_DTRecHitQuality_h

/** \class DTRecHitQuality
 *  Basic analyzer class which accesses 1D DTRecHits
 *  and plot resolution comparing reconstructed and simulated quantities
 *
 *  Residual/pull plots are filled for the rechit with distance from wire
 *  closer to that of the muon simhit.
 *
 *  Efficiencies are defined as the fraction of muon simhits with a rechit
 *  in the same cell, for the given reconstruction step. Hence, for S2 and S3
 *  the definition incorporate the segment reconstruction efficiency.
 *
 *  \author G. Cerminara - INFN Torino
 */

#include <map>
#include <string>
#include <vector>

#include "DQMServices/Core/interface/DQMGlobalEDAnalyzer.h"
#include "DataFormats/DTRecHit/interface/DTRecHitCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment2DCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}  // namespace edm

class PSimHit;
class DTLayer;
class DTWireId;
class DTGeometry;
class HRes1DHit;
class HEff1DHit;
namespace dtrechit {
  struct Histograms;
}

class DTRecHitQuality : public DQMGlobalEDAnalyzer<dtrechit::Histograms> {
public:
  /// Constructor
  DTRecHitQuality(const edm::ParameterSet &pset);

private:
  /// Book the DQM plots
  void bookHistograms(DQMStore::IBooker &,
                      edm::Run const &,
                      edm::EventSetup const &,
                      dtrechit::Histograms &) const override;

  /// Perform the real analysis
  void dqmAnalyze(edm::Event const &, edm::EventSetup const &, dtrechit::Histograms const &) const override;

private:
  // Switch for debug output
  bool debug_;

  // Root file name
  edm::EDGetTokenT<edm::PSimHitContainer> simHitToken_;
  edm::EDGetTokenT<DTRecHitCollection> recHitToken_;
  edm::EDGetTokenT<DTRecSegment2DCollection> segment2DToken_;
  edm::EDGetTokenT<DTRecSegment4DCollection> segment4DToken_;
  ;

  edm::InputTag simHitLabel_;
  edm::InputTag recHitLabel_;
  edm::InputTag segment2DLabel_;
  edm::InputTag segment4DLabel_;

  // Switches for analysis at various steps
  bool doStep1_;
  bool doStep2_;
  bool doStep3_;
  bool local_;
  bool doall_;

  // Return a map between DTRecHit1DPair and wireId
  std::map<DTWireId, std::vector<DTRecHit1DPair>> map1DRecHitsPerWire(const DTRecHitCollection *dt1DRecHitPairs) const;

  // Return a map between DTRecHit1D and wireId
  std::map<DTWireId, std::vector<DTRecHit1D>> map1DRecHitsPerWire(const DTRecSegment2DCollection *segment2Ds) const;

  // Return a map between DTRecHit1D and wireId
  std::map<DTWireId, std::vector<DTRecHit1D>> map1DRecHitsPerWire(const DTRecSegment4DCollection *segment4Ds) const;

  // Compute SimHit distance from wire (cm)
  float simHitDistFromWire(const DTLayer *layer, const DTWireId &wireId, const PSimHit &hit) const;

  // Compute SimHit impact angle (in direction perp to wire)
  float simHitImpactAngle(const DTLayer *layer, const DTWireId &wireId, const PSimHit &hit) const;

  // Compute SimHit distance from FrontEnd
  float simHitDistFromFE(const DTLayer *layer, const DTWireId &wireId, const PSimHit &hit) const;

  // Find the RecHit closest to the muon SimHit
  //   const DTRecHit1DPair*
  //   findBestRecHit(const DTLayer* layer,
  //               DTWireId wireId,
  //               const std::vector<DTRecHit1DPair>& recHits,
  //               const float simHitDist) const;
  template <typename type>
  const type *findBestRecHit(const DTLayer *layer,
                             const DTWireId &wireId,
                             const std::vector<type> &recHits,
                             float simHitDist) const;

  // Compute the distance from wire (cm) of a hits in a DTRecHit1DPair
  float recHitDistFromWire(const DTRecHit1DPair &hitPair, const DTLayer *layer) const;
  // Compute the distance from wire (cm) of a hits in a DTRecHit1D
  float recHitDistFromWire(const DTRecHit1D &recHit, const DTLayer *layer) const;

  // Return the error on the measured (cm) coordinate
  float recHitPositionError(const DTRecHit1DPair &recHit) const;
  float recHitPositionError(const DTRecHit1D &recHit) const;

  // Does the real job
  template <typename type>
  void compute(const DTGeometry *dtGeom,
               const std::map<DTWireId, std::vector<PSimHit>> &simHitsPerWire,
               const std::map<DTWireId, std::vector<type>> &recHitsPerWire,
               dtrechit::Histograms const &histograms,
               int step) const;
};

#endif  // Validation_DTRecHits_DTRecHitQuality_h
