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

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DataFormats/DTRecHit/interface/DTRecHitCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment2DCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
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

class PSimHit;
class TFile;
class DTLayer;
class DTWireId;
class DTGeometry;

class DTRecHitQuality : public edm::EDAnalyzer {
public:
  /// Constructor
  DTRecHitQuality(const edm::ParameterSet& pset);

  /// Perform the real analysis
  void analyze(const edm::Event & event, const edm::EventSetup& eventSetup) override;

  void beginRun(const edm::Run& iRun, const edm::EventSetup &setup) override;

private:
  // Switch for debug output
  bool debug_;

  // Root file name
  edm::EDGetTokenT<edm::PSimHitContainer> simHitToken_;
  edm::EDGetTokenT<DTRecHitCollection> recHitToken_;
  edm::EDGetTokenT<DTRecSegment2DCollection> segment2DToken_;
  edm::EDGetTokenT<DTRecSegment4DCollection> segment4DToken_;;

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
  std::map<DTWireId, std::vector<DTRecHit1DPair> >
    map1DRecHitsPerWire(const DTRecHitCollection* dt1DRecHitPairs);

  // Return a map between DTRecHit1D and wireId
  std::map<DTWireId, std::vector<DTRecHit1D> >
    map1DRecHitsPerWire(const DTRecSegment2DCollection* segment2Ds);

  // Return a map between DTRecHit1D and wireId
  std::map<DTWireId, std::vector<DTRecHit1D> >
    map1DRecHitsPerWire(const DTRecSegment4DCollection* segment4Ds);

  // Compute SimHit distance from wire (cm)
  float simHitDistFromWire(const DTLayer* layer,
                           DTWireId wireId,
                           const PSimHit& hit);

  // Compute SimHit impact angle (in direction perp to wire)
  float simHitImpactAngle(const DTLayer* layer,
                           DTWireId wireId,
                           const PSimHit& hit);

  // Compute SimHit distance from FrontEnd
  float simHitDistFromFE(const DTLayer* layer,
                           DTWireId wireId,
                           const PSimHit& hit);

  // Find the RecHit closest to the muon SimHit
//   const DTRecHit1DPair*
//   findBestRecHit(const DTLayer* layer,
//               DTWireId wireId,
//               const std::vector<DTRecHit1DPair>& recHits,
//               const float simHitDist);
  template <typename type>
  const type*
  findBestRecHit(const DTLayer* layer,
                 DTWireId wireId,
                 const std::vector<type>& recHits,
                 float simHitDist);

  // Compute the distance from wire (cm) of a hits in a DTRecHit1DPair
  float recHitDistFromWire(const DTRecHit1DPair& hitPair, const DTLayer* layer);
  // Compute the distance from wire (cm) of a hits in a DTRecHit1D
  float recHitDistFromWire(const DTRecHit1D& recHit, const DTLayer* layer);

  // Return the error on the measured (cm) coordinate
  float recHitPositionError(const DTRecHit1DPair& recHit);
  float recHitPositionError(const DTRecHit1D& recHit);

  // Does the real job
  template <typename type>
  void compute(const DTGeometry *dtGeom,
               const std::map<DTWireId, std::vector<PSimHit> >& simHitsPerWire,
               const std::map<DTWireId, std::vector<type> >& recHitsPerWire,
               int step);

  HRes1DHit *hRes_S1RPhi_;          // RecHits, 1. step, RPh
  HRes1DHit *hRes_S2RPhi_;          // RecHits, 2. step, RPhi
  HRes1DHit *hRes_S3RPhi_;          // RecHits, 3. step, RPhi

  HRes1DHit *hRes_S1RZ_;            // RecHits, 1. step, RZ
  HRes1DHit *hRes_S2RZ_;            // RecHits, 2. step, RZ
  HRes1DHit *hRes_S3RZ_;            // RecHits, 3. step, RZ

  HRes1DHit *hRes_S1RZ_W0_;         // RecHits, 1. step, RZ, wheel 0
  HRes1DHit *hRes_S2RZ_W0_;         // RecHits, 2. step, RZ, wheel 0
  HRes1DHit *hRes_S3RZ_W0_;         // RecHits, 3. step, RZ, wheel 0

  HRes1DHit *hRes_S1RZ_W1_;         // RecHits, 1. step, RZ, wheel +-1
  HRes1DHit *hRes_S2RZ_W1_;         // RecHits, 2. step, RZ, wheel +-1
  HRes1DHit *hRes_S3RZ_W1_;         // RecHits, 3. step, RZ, wheel +-1

  HRes1DHit *hRes_S1RZ_W2_;         // RecHits, 1. step, RZ, wheel +-2
  HRes1DHit *hRes_S2RZ_W2_;         // RecHits, 2. step, RZ, wheel +-2
  HRes1DHit *hRes_S3RZ_W2_;         // RecHits, 3. step, RZ, wheel +-2

  HRes1DHit *hRes_S1RPhi_W0_;       // RecHits, 1. step, RPhi, wheel 0
  HRes1DHit *hRes_S2RPhi_W0_;       // RecHits, 2. step, RPhi, wheel 0
  HRes1DHit *hRes_S3RPhi_W0_;       // RecHits, 3. step, RPhi, wheel 0

  HRes1DHit *hRes_S1RPhi_W1_;       // RecHits, 1. step, RPhi, wheel +-1
  HRes1DHit *hRes_S2RPhi_W1_;       // RecHits, 2. step, RPhi, wheel +-1
  HRes1DHit *hRes_S3RPhi_W1_;       // RecHits, 3. step, RPhi, wheel +-1

  HRes1DHit *hRes_S1RPhi_W2_;       // RecHits, 1. step, RPhi, wheel +-2
  HRes1DHit *hRes_S2RPhi_W2_;       // RecHits, 2. step, RPhi, wheel +-2
  HRes1DHit *hRes_S3RPhi_W2_;       // RecHits, 3. step, RPhi, wheel +-2

  HRes1DHit* hRes_S3RPhiWS_[3][4];  // RecHits, 3. step, by wheel/station
  HRes1DHit* hRes_S3RZWS_[3][4];    // RecHits, 3. step, by wheel/station

  HEff1DHit *hEff_S1RPhi_;          // RecHits, 1. step, RPhi
  HEff1DHit *hEff_S2RPhi_;          // RecHits, 2. step, RPhi
  HEff1DHit *hEff_S3RPhi_;          // RecHits, 3. step, RPhi

  HEff1DHit *hEff_S1RZ_;            // RecHits, 1. step, RZ
  HEff1DHit *hEff_S2RZ_;            // RecHits, 2. step, RZ
  HEff1DHit *hEff_S3RZ_;            // RecHits, 3. step, RZ

  HEff1DHit *hEff_S1RZ_W0_;         // RecHits, 1. step, RZ, wheel 0
  HEff1DHit *hEff_S2RZ_W0_;         // RecHits, 2. step, RZ, wheel 0
  HEff1DHit *hEff_S3RZ_W0_;         // RecHits, 3. step, RZ, wheel 0

  HEff1DHit *hEff_S1RZ_W1_;         // RecHits, 1. step, RZ, wheel +-1
  HEff1DHit *hEff_S2RZ_W1_;         // RecHits, 2. step, RZ, wheel +-1
  HEff1DHit *hEff_S3RZ_W1_;         // RecHits, 3. step, RZ, wheel +-1

  HEff1DHit *hEff_S1RZ_W2_;         // RecHits, 1. step, RZ, wheel +-2
  HEff1DHit *hEff_S2RZ_W2_;         // RecHits, 2. step, RZ, wheel +-2
  HEff1DHit *hEff_S3RZ_W2_;         // RecHits, 3. step, RZ, wheel +-2

  HEff1DHit* hEff_S1RPhiWS_[3][4];  // RecHits, 3. step, by wheel/station
  HEff1DHit* hEff_S3RPhiWS_[3][4];  // RecHits, 3. step, by wheel/station
  HEff1DHit* hEff_S1RZWS_[3][4];    // RecHits, 3. step, by wheel/station
  HEff1DHit* hEff_S3RZWS_[3][4];    // RecHits, 3. step, by wheel/station

  DQMStore* dbe_;
};

#endif // Validation_DTRecHits_DTRecHitQuality_h
