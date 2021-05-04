/*
 *  See header file for a description of this class.
 *
 *  \author G. Cerminara - INFN Torino
 */

#include <iostream>
#include <map>

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Validation/DTRecHits/interface/DTHitQualityUtils.h"

#include "DTRecHitQuality.h"
#include "Histograms.h"

using namespace std;
using namespace edm;

namespace dtrechit {
  struct Histograms {
    std::unique_ptr<HRes1DHit> hRes_S1RPhi;          // RecHits, 1. step, RPh
    std::unique_ptr<HRes1DHit> hRes_S2RPhi;          // RecHits, 2. step, RPhi
    std::unique_ptr<HRes1DHit> hRes_S3RPhi;          // RecHits, 3. step, RPhi
    std::unique_ptr<HRes1DHit> hRes_S1RZ;            // RecHits, 1. step, RZ
    std::unique_ptr<HRes1DHit> hRes_S2RZ;            // RecHits, 2. step, RZ
    std::unique_ptr<HRes1DHit> hRes_S3RZ;            // RecHits, 3. step, RZ
    std::unique_ptr<HRes1DHit> hRes_S1RZ_W0;         // RecHits, 1. step, RZ, wheel 0
    std::unique_ptr<HRes1DHit> hRes_S2RZ_W0;         // RecHits, 2. step, RZ, wheel 0
    std::unique_ptr<HRes1DHit> hRes_S3RZ_W0;         // RecHits, 3. step, RZ, wheel 0
    std::unique_ptr<HRes1DHit> hRes_S1RZ_W1;         // RecHits, 1. step, RZ, wheel +-1
    std::unique_ptr<HRes1DHit> hRes_S2RZ_W1;         // RecHits, 2. step, RZ, wheel +-1
    std::unique_ptr<HRes1DHit> hRes_S3RZ_W1;         // RecHits, 3. step, RZ, wheel +-1
    std::unique_ptr<HRes1DHit> hRes_S1RZ_W2;         // RecHits, 1. step, RZ, wheel +-2
    std::unique_ptr<HRes1DHit> hRes_S2RZ_W2;         // RecHits, 2. step, RZ, wheel +-2
    std::unique_ptr<HRes1DHit> hRes_S3RZ_W2;         // RecHits, 3. step, RZ, wheel +-2
    std::unique_ptr<HRes1DHit> hRes_S1RPhi_W0;       // RecHits, 1. step, RPhi, wheel 0
    std::unique_ptr<HRes1DHit> hRes_S2RPhi_W0;       // RecHits, 2. step, RPhi, wheel 0
    std::unique_ptr<HRes1DHit> hRes_S3RPhi_W0;       // RecHits, 3. step, RPhi, wheel 0
    std::unique_ptr<HRes1DHit> hRes_S1RPhi_W1;       // RecHits, 1. step, RPhi, wheel +-1
    std::unique_ptr<HRes1DHit> hRes_S2RPhi_W1;       // RecHits, 2. step, RPhi, wheel +-1
    std::unique_ptr<HRes1DHit> hRes_S3RPhi_W1;       // RecHits, 3. step, RPhi, wheel +-1
    std::unique_ptr<HRes1DHit> hRes_S1RPhi_W2;       // RecHits, 1. step, RPhi, wheel +-2
    std::unique_ptr<HRes1DHit> hRes_S2RPhi_W2;       // RecHits, 2. step, RPhi, wheel +-2
    std::unique_ptr<HRes1DHit> hRes_S3RPhi_W2;       // RecHits, 3. step, RPhi, wheel +-2
    std::unique_ptr<HRes1DHit> hRes_S3RPhiWS[3][4];  // RecHits, 3. step, by wheel/station
    std::unique_ptr<HRes1DHit> hRes_S3RZWS[3][4];    // RecHits, 3. step, by wheel/station

    std::unique_ptr<HEff1DHit> hEff_S1RPhi;          // RecHits, 1. step, RPhi
    std::unique_ptr<HEff1DHit> hEff_S2RPhi;          // RecHits, 2. step, RPhi
    std::unique_ptr<HEff1DHit> hEff_S3RPhi;          // RecHits, 3. step, RPhi
    std::unique_ptr<HEff1DHit> hEff_S1RZ;            // RecHits, 1. step, RZ
    std::unique_ptr<HEff1DHit> hEff_S2RZ;            // RecHits, 2. step, RZ
    std::unique_ptr<HEff1DHit> hEff_S3RZ;            // RecHits, 3. step, RZ
    std::unique_ptr<HEff1DHit> hEff_S1RZ_W0;         // RecHits, 1. step, RZ, wheel 0
    std::unique_ptr<HEff1DHit> hEff_S2RZ_W0;         // RecHits, 2. step, RZ, wheel 0
    std::unique_ptr<HEff1DHit> hEff_S3RZ_W0;         // RecHits, 3. step, RZ, wheel 0
    std::unique_ptr<HEff1DHit> hEff_S1RZ_W1;         // RecHits, 1. step, RZ, wheel +-1
    std::unique_ptr<HEff1DHit> hEff_S2RZ_W1;         // RecHits, 2. step, RZ, wheel +-1
    std::unique_ptr<HEff1DHit> hEff_S3RZ_W1;         // RecHits, 3. step, RZ, wheel +-1
    std::unique_ptr<HEff1DHit> hEff_S1RZ_W2;         // RecHits, 1. step, RZ, wheel +-2
    std::unique_ptr<HEff1DHit> hEff_S2RZ_W2;         // RecHits, 2. step, RZ, wheel +-2
    std::unique_ptr<HEff1DHit> hEff_S3RZ_W2;         // RecHits, 3. step, RZ, wheel +-2
    std::unique_ptr<HEff1DHit> hEff_S1RPhiWS[3][4];  // RecHits, 3. step, by wheel/station
    std::unique_ptr<HEff1DHit> hEff_S3RPhiWS[3][4];  // RecHits, 3. step, by wheel/station
    std::unique_ptr<HEff1DHit> hEff_S1RZWS[3][4];    // RecHits, 3. step, by wheel/station
    std::unique_ptr<HEff1DHit> hEff_S3RZWS[3][4];    // RecHits, 3. step, by wheel/station
  };
}  // namespace dtrechit

using namespace dtrechit;

// In phi SLs, The dependency on X and angle is specular in positive
// and negative wheels. Since positive and negative wheels are filled
// together into the same plots, it is useful to mirror negative wheels
// so that the actual dependency can be observerd instead of an artificially
// simmetrized one.
// Set mirrorMinusWheels to avoid this.
namespace {
  constexpr bool mirrorMinusWheels = true;
}

// Constructor
DTRecHitQuality::DTRecHitQuality(const ParameterSet &pset) {
  // Get the debug parameter for verbose output
  debug_ = pset.getUntrackedParameter<bool>("debug");
  // the name of the simhit collection
  simHitLabel_ = pset.getUntrackedParameter<InputTag>("simHitLabel");
  simHitToken_ = consumes<PSimHitContainer>(pset.getUntrackedParameter<InputTag>("simHitLabel"));
  // the name of the 1D rec hit collection
  recHitLabel_ = pset.getUntrackedParameter<InputTag>("recHitLabel");
  recHitToken_ = consumes<DTRecHitCollection>(pset.getUntrackedParameter<InputTag>("recHitLabel"));
  // the name of the 2D rec hit collection
  segment2DLabel_ = pset.getUntrackedParameter<InputTag>("segment2DLabel");
  segment2DToken_ = consumes<DTRecSegment2DCollection>(pset.getUntrackedParameter<InputTag>("segment2DLabel"));
  // the name of the 4D rec hit collection
  segment4DLabel_ = pset.getUntrackedParameter<InputTag>("segment4DLabel");
  segment4DToken_ = consumes<DTRecSegment4DCollection>(pset.getUntrackedParameter<InputTag>("segment4DLabel"));
  // Switches for analysis at various steps
  doStep1_ = pset.getUntrackedParameter<bool>("doStep1", false);
  doStep2_ = pset.getUntrackedParameter<bool>("doStep2", false);
  doStep3_ = pset.getUntrackedParameter<bool>("doStep3", false);
  doall_ = pset.getUntrackedParameter<bool>("doall", false);
  local_ = pset.getUntrackedParameter<bool>("local", true);
}

void DTRecHitQuality::bookHistograms(DQMStore::IBooker &booker,
                                     edm::Run const &run,
                                     edm::EventSetup const &setup,
                                     Histograms &histograms) const {
  if (doall_ && doStep1_) {
    histograms.hRes_S1RPhi = std::make_unique<HRes1DHit>("S1RPhi", booker, true, local_);  // RecHits, 1. step, RPhi
    histograms.hRes_S1RPhi_W0 =
        std::make_unique<HRes1DHit>("S1RPhi_W0", booker, true, local_);  // RecHits, 1. step, RZ, wheel 0
    histograms.hRes_S1RPhi_W1 =
        std::make_unique<HRes1DHit>("S1RPhi_W1", booker, true, local_);  // RecHits, 1. step, RZ, wheel +-1
    histograms.hRes_S1RPhi_W2 =
        std::make_unique<HRes1DHit>("S1RPhi_W2", booker, true, local_);  // RecHits, 1. step, RZ, wheel +-2
    histograms.hRes_S1RZ = std::make_unique<HRes1DHit>("S1RZ", booker, true, local_);  // RecHits, 1. step, RZ
    histograms.hRes_S1RZ_W0 =
        std::make_unique<HRes1DHit>("S1RZ_W0", booker, true, local_);  // RecHits, 1. step, RZ, wheel 0
    histograms.hRes_S1RZ_W1 =
        std::make_unique<HRes1DHit>("S1RZ_W1", booker, true, local_);  // RecHits, 1. step, RZ, wheel +-1
    histograms.hRes_S1RZ_W2 =
        std::make_unique<HRes1DHit>("S1RZ_W2", booker, true, local_);          // RecHits, 1. step, RZ, wheel +-2
    histograms.hEff_S1RPhi = std::make_unique<HEff1DHit>("S1RPhi", booker);    // RecHits, 1. step, RPhi
    histograms.hEff_S1RZ = std::make_unique<HEff1DHit>("S1RZ", booker);        // RecHits, 1. step, RZ
    histograms.hEff_S1RZ_W0 = std::make_unique<HEff1DHit>("S1RZ_W0", booker);  // RecHits, 1. step, RZ, wheel 0
    histograms.hEff_S1RZ_W1 = std::make_unique<HEff1DHit>("S1RZ_W1", booker);  // RecHits, 1. step, RZ, wheel +-1
    histograms.hEff_S1RZ_W2 = std::make_unique<HEff1DHit>("S1RZ_W2", booker);  // RecHits, 1. step, RZ, wheel +-2
  }
  if (doall_ && doStep2_) {
    histograms.hRes_S2RPhi = std::make_unique<HRes1DHit>("S2RPhi", booker, true, local_);  // RecHits, 2. step, RPhi
    histograms.hRes_S2RPhi_W0 =
        std::make_unique<HRes1DHit>("S2RPhi_W0", booker, true, local_);  // RecHits, 2. step, RPhi, wheel 0
    histograms.hRes_S2RPhi_W1 =
        std::make_unique<HRes1DHit>("S2RPhi_W1", booker, true, local_);  // RecHits, 2. step, RPhi, wheel +-1
    histograms.hRes_S2RPhi_W2 =
        std::make_unique<HRes1DHit>("S2RPhi_W2", booker, true, local_);  // RecHits, 2. step, RPhi, wheel +-2
    histograms.hRes_S2RZ = std::make_unique<HRes1DHit>("S2RZ", booker, true, local_);  // RecHits, 2. step, RZ
    histograms.hRes_S2RZ_W0 =
        std::make_unique<HRes1DHit>("S2RZ_W0", booker, true, local_);  // RecHits, 2. step, RZ, wheel 0
    histograms.hRes_S2RZ_W1 =
        std::make_unique<HRes1DHit>("S2RZ_W1", booker, true, local_);  // RecHits, 2. step, RZ, wheel +-1
    histograms.hRes_S2RZ_W2 =
        std::make_unique<HRes1DHit>("S2RZ_W2", booker, true, local_);          // RecHits, 2. step, RZ, wheel +-2
    histograms.hEff_S2RPhi = std::make_unique<HEff1DHit>("S2RPhi", booker);    // RecHits, 2. step, RPhi
    histograms.hEff_S2RZ_W0 = std::make_unique<HEff1DHit>("S2RZ_W0", booker);  // RecHits, 2. step, RZ, wheel 0
    histograms.hEff_S2RZ_W1 = std::make_unique<HEff1DHit>("S2RZ_W1", booker);  // RecHits, 2. step, RZ, wheel +-1
    histograms.hEff_S2RZ_W2 = std::make_unique<HEff1DHit>("S2RZ_W2", booker);  // RecHits, 2. step, RZ, wheel +-2
    histograms.hEff_S2RZ = std::make_unique<HEff1DHit>("S2RZ", booker);        // RecHits, 2. step, RZ
  }
  if (doStep3_) {
    histograms.hRes_S3RPhi = std::make_unique<HRes1DHit>("S3RPhi", booker, doall_, local_);  // RecHits, 3. step, RPhi
    histograms.hRes_S3RPhi_W0 =
        std::make_unique<HRes1DHit>("S3RPhi_W0", booker, doall_, local_);  // RecHits, 3. step, RPhi, wheel 0
    histograms.hRes_S3RPhi_W1 = std::make_unique<HRes1DHit>("S3RPhi_W1",
                                                            booker,
                                                            doall_,
                                                            local_);  // RecHits, 3. step, RPhi, wheel +-1
    histograms.hRes_S3RPhi_W2 = std::make_unique<HRes1DHit>("S3RPhi_W2",
                                                            booker,
                                                            doall_,
                                                            local_);  // RecHits, 3. step, RPhi, wheel +-2
    histograms.hRes_S3RZ = std::make_unique<HRes1DHit>("S3RZ", booker, doall_, local_);  // RecHits, 3. step, RZ
    histograms.hRes_S3RZ_W0 =
        std::make_unique<HRes1DHit>("S3RZ_W0", booker, doall_, local_);  // RecHits, 3. step, RZ, wheel 0
    histograms.hRes_S3RZ_W1 =
        std::make_unique<HRes1DHit>("S3RZ_W1", booker, doall_, local_);  // RecHits, 3. step, RZ, wheel +-1
    histograms.hRes_S3RZ_W2 =
        std::make_unique<HRes1DHit>("S3RZ_W2", booker, doall_, local_);  // RecHits, 3. step, RZ, wheel +-2

    if (local_) {
      // Plots with finer granularity, not to be included in DQM
      TString name1 = "RPhi_W";
      TString name2 = "RZ_W";
      for (long w = 0; w <= 2; ++w) {
        for (long s = 1; s <= 4; ++s) {
          histograms.hRes_S3RPhiWS[w][s - 1] =
              std::make_unique<HRes1DHit>(("S3" + name1 + w + "_St" + s).Data(), booker, doall_, local_);
          histograms.hEff_S1RPhiWS[w][s - 1] =
              std::make_unique<HEff1DHit>(("S1" + name1 + w + "_St" + s).Data(), booker);
          histograms.hEff_S3RPhiWS[w][s - 1] =
              std::make_unique<HEff1DHit>(("S3" + name1 + w + "_St" + s).Data(), booker);
          if (s != 4) {
            histograms.hRes_S3RZWS[w][s - 1] =
                std::make_unique<HRes1DHit>(("S3" + name2 + w + "_St" + s).Data(), booker, doall_, local_);
            histograms.hEff_S1RZWS[w][s - 1] =
                std::make_unique<HEff1DHit>(("S1" + name2 + w + "_St" + s).Data(), booker);
            histograms.hEff_S3RZWS[w][s - 1] =
                std::make_unique<HEff1DHit>(("S3" + name2 + w + "_St" + s).Data(), booker);
          }
        }
      }
    }

    if (doall_) {
      histograms.hEff_S3RPhi = std::make_unique<HEff1DHit>("S3RPhi", booker);    // RecHits, 3. step, RPhi
      histograms.hEff_S3RZ = std::make_unique<HEff1DHit>("S3RZ", booker);        // RecHits, 3. step, RZ
      histograms.hEff_S3RZ_W0 = std::make_unique<HEff1DHit>("S3RZ_W0", booker);  // RecHits, 3. step, RZ, wheel 0
      histograms.hEff_S3RZ_W1 = std::make_unique<HEff1DHit>("S3RZ_W1", booker);  // RecHits, 3. step, RZ, wheel +-1
      histograms.hEff_S3RZ_W2 = std::make_unique<HEff1DHit>("S3RZ_W2", booker);  // RecHits, 3. step, RZ, wheel +-2
    }
  }
}

// The real analysis
void DTRecHitQuality::dqmAnalyze(edm::Event const &event,
                                 edm::EventSetup const &setup,
                                 Histograms const &histograms) const {
  if (debug_) {
    cout << "--- [DTRecHitQuality] Analysing Event: #Run: " << event.id().run() << " #Event: " << event.id().event()
         << endl;
  }

  // Get the DT Geometry
  ESHandle<DTGeometry> dtGeom;
  setup.get<MuonGeometryRecord>().get(dtGeom);

  // Get the SimHit collection from the event
  Handle<PSimHitContainer> simHits;
  event.getByToken(simHitToken_, simHits);

  // Map simhits per wire
  map<DTWireId, PSimHitContainer> simHitsPerWire = DTHitQualityUtils::mapSimHitsPerWire(*(simHits.product()));

  //=======================================================================================
  // RecHit analysis at Step 1
  if (doStep1_ && doall_) {
    if (debug_) {
      cout << "  -- DTRecHit S1: begin analysis:" << endl;
    }
    // Get the rechit collection from the event
    Handle<DTRecHitCollection> dtRecHits;
    event.getByToken(recHitToken_, dtRecHits);

    if (!dtRecHits.isValid()) {
      if (debug_) {
        cout << "[DTRecHitQuality]**Warning: no 1DRechits with label: " << recHitLabel_ << " in this event, skipping!"
             << endl;
      }
      return;
    }

    // Map rechits per wire
    auto const &recHitsPerWire = map1DRecHitsPerWire(dtRecHits.product());
    compute(dtGeom.product(), simHitsPerWire, recHitsPerWire, histograms, 1);
  }

  //=======================================================================================
  // RecHit analysis at Step 2
  if (doStep2_ && doall_) {
    if (debug_) {
      cout << "  -- DTRecHit S2: begin analysis:" << endl;
    }

    // Get the 2D rechits from the event
    Handle<DTRecSegment2DCollection> segment2Ds;
    event.getByToken(segment2DToken_, segment2Ds);

    if (!segment2Ds.isValid()) {
      if (debug_) {
        cout << "[DTRecHitQuality]**Warning: no 2DSegments with label: " << segment2DLabel_
             << " in this event, skipping!" << endl;
      }

    } else {
      // Map rechits per wire
      auto const &recHitsPerWire = map1DRecHitsPerWire(segment2Ds.product());
      compute(dtGeom.product(), simHitsPerWire, recHitsPerWire, histograms, 2);
    }
  }

  //=======================================================================================
  // RecHit analysis at Step 3
  if (doStep3_) {
    if (debug_) {
      cout << "  -- DTRecHit S3: begin analysis:" << endl;
    }

    // Get the 4D rechits from the event
    Handle<DTRecSegment4DCollection> segment4Ds;
    event.getByToken(segment4DToken_, segment4Ds);

    if (!segment4Ds.isValid()) {
      if (debug_) {
        cout << "[DTRecHitQuality]**Warning: no 4D Segments with label: " << segment4DLabel_
             << " in this event, skipping!" << endl;
      }
      return;
    }

    // Map rechits per wire
    auto const &recHitsPerWire = map1DRecHitsPerWire(segment4Ds.product());
    compute(dtGeom.product(), simHitsPerWire, recHitsPerWire, histograms, 3);
  }
}

// Return a map between DTRecHit1DPair and wireId
map<DTWireId, vector<DTRecHit1DPair>> DTRecHitQuality::map1DRecHitsPerWire(
    const DTRecHitCollection *dt1DRecHitPairs) const {
  map<DTWireId, vector<DTRecHit1DPair>> ret;

  for (const auto &dt1DRecHitPair : *dt1DRecHitPairs) {
    ret[dt1DRecHitPair.wireId()].push_back(dt1DRecHitPair);
  }

  return ret;
}

// Return a map between DTRecHit1D at S2 and wireId
map<DTWireId, vector<DTRecHit1D>> DTRecHitQuality::map1DRecHitsPerWire(
    const DTRecSegment2DCollection *segment2Ds) const {
  map<DTWireId, vector<DTRecHit1D>> ret;

  // Loop over all 2D segments
  for (const auto &segment2D : *segment2Ds) {
    vector<DTRecHit1D> component1DHits = segment2D.specificRecHits();
    // Loop over all component 1D hits
    for (auto &component1DHit : component1DHits) {
      ret[component1DHit.wireId()].push_back(component1DHit);
    }
  }
  return ret;
}

// Return a map between DTRecHit1D at S3 and wireId
map<DTWireId, std::vector<DTRecHit1D>> DTRecHitQuality::map1DRecHitsPerWire(
    const DTRecSegment4DCollection *segment4Ds) const {
  map<DTWireId, vector<DTRecHit1D>> ret;
  // Loop over all 4D segments
  for (const auto &segment4D : *segment4Ds) {
    // Get component 2D segments
    vector<const TrackingRecHit *> segment2Ds = segment4D.recHits();
    // Loop over 2D segments:
    for (auto &segment2D : segment2Ds) {
      // Get 1D component rechits
      vector<const TrackingRecHit *> hits = segment2D->recHits();
      // Loop over them
      for (auto &hit : hits) {
        const auto *hit1D = dynamic_cast<const DTRecHit1D *>(hit);
        ret[hit1D->wireId()].push_back(*hit1D);
      }
    }
  }

  return ret;
}

// Compute SimHit distance from wire (cm)
float DTRecHitQuality::simHitDistFromWire(const DTLayer *layer, const DTWireId &wireId, const PSimHit &hit) const {
  float xwire = layer->specificTopology().wirePosition(wireId.wire());
  LocalPoint entryP = hit.entryPoint();
  LocalPoint exitP = hit.exitPoint();
  float xEntry = entryP.x() - xwire;
  float xExit = exitP.x() - xwire;

  return fabs(xEntry - (entryP.z() * (xExit - xEntry)) / (exitP.z() - entryP.z()));  // FIXME: check...
}

// Compute SimHit impact angle (in direction perp to wire), in the SL RF
float DTRecHitQuality::simHitImpactAngle(const DTLayer *layer, const DTWireId &wireId, const PSimHit &hit) const {
  LocalPoint entryP = hit.entryPoint();
  LocalPoint exitP = hit.exitPoint();
  float theta = (exitP.x() - entryP.x()) / (exitP.z() - entryP.z());
  return atan(theta);
}

// Compute SimHit distance from FrontEnd
float DTRecHitQuality::simHitDistFromFE(const DTLayer *layer, const DTWireId &wireId, const PSimHit &hit) const {
  LocalPoint entryP = hit.entryPoint();
  LocalPoint exitP = hit.exitPoint();
  float wireLenght = layer->specificTopology().cellLenght();
  // FIXME: should take only wireLenght/2.;
  // moreover, pos+cellLenght/2. is shorter than the distance from FE.
  // In fact it would make more sense to make plots vs y.
  return (entryP.y() + exitP.y()) / 2. + wireLenght;
}

// Find the RecHit closest to the muon SimHit
template <typename type>
const type *DTRecHitQuality::findBestRecHit(const DTLayer *layer,
                                            const DTWireId &wireId,
                                            const vector<type> &recHits,
                                            const float simHitDist) const {
  float res = 99999;
  const type *theBestRecHit = nullptr;
  // Loop over RecHits within the cell
  for (auto recHit = recHits.begin(); recHit != recHits.end(); recHit++) {
    float distTmp = recHitDistFromWire(*recHit, layer);
    if (fabs(distTmp - simHitDist) < res) {
      res = fabs(distTmp - simHitDist);
      theBestRecHit = &(*recHit);
    }
  }  // End of loop over RecHits within the cell

  return theBestRecHit;
}

// Compute the distance from wire (cm) of a hits in a DTRecHit1DPair
float DTRecHitQuality::recHitDistFromWire(const DTRecHit1DPair &hitPair, const DTLayer *layer) const {
  // Compute the rechit distance from wire
  return fabs(hitPair.localPosition(DTEnums::Left).x() - hitPair.localPosition(DTEnums::Right).x()) / 2.;
}

// Compute the distance from wire (cm) of a hits in a DTRecHit1D
float DTRecHitQuality::recHitDistFromWire(const DTRecHit1D &recHit, const DTLayer *layer) const {
  return fabs(recHit.localPosition().x() - layer->specificTopology().wirePosition(recHit.wireId().wire()));
}

template <typename type>
void DTRecHitQuality::compute(const DTGeometry *dtGeom,
                              const std::map<DTWireId, std::vector<PSimHit>> &simHitsPerWire,
                              const std::map<DTWireId, std::vector<type>> &recHitsPerWire,
                              Histograms const &histograms,
                              int step) const {
  // Loop over cells with a muon SimHit
  for (const auto &wireAndSHits : simHitsPerWire) {
    DTWireId wireId = wireAndSHits.first;
    int wheel = wireId.wheel();
    int sl = wireId.superLayer();

    vector<PSimHit> simHitsInCell = wireAndSHits.second;

    // Get the layer
    const DTLayer *layer = dtGeom->layer(wireId);

    // Look for a mu hit in the cell
    const PSimHit *muSimHit = DTHitQualityUtils::findMuSimHit(simHitsInCell);
    if (muSimHit == nullptr) {
      if (debug_) {
        cout << "   No mu SimHit in channel: " << wireId << ", skipping! " << endl;
      }
      continue;  // Skip this cell
    }

    // Find the distance of the simhit from the wire
    float simHitWireDist = simHitDistFromWire(layer, wireId, *muSimHit);
    // Skip simhits out of the cell
    if (simHitWireDist > 2.1) {
      if (debug_) {
        cout << "  [DTRecHitQuality]###Warning: The mu SimHit in out of the "
                "cell, skipping!"
             << endl;
      }
      continue;  // Skip this cell
    }
    GlobalPoint simHitGlobalPos = layer->toGlobal(muSimHit->localPosition());

    // find SH impact angle
    float simHitTheta = simHitImpactAngle(layer, wireId, *muSimHit);

    // find SH distance from FE
    float simHitFEDist = simHitDistFromFE(layer, wireId, *muSimHit);

    bool recHitReconstructed = false;

    // Look for RecHits in the same cell
    if (recHitsPerWire.find(wireId) == recHitsPerWire.end()) {
      // No RecHit found in this cell
      if (debug_) {
        cout << "   No RecHit found at Step: " << step << " in cell: " << wireId << endl;
      }
    } else {
      recHitReconstructed = true;
      // vector<type> recHits = (*wireAndRecHits).second;
      const vector<type> &recHits = recHitsPerWire.at(wireId);
      if (debug_) {
        cout << "   " << recHits.size() << " RecHits, Step " << step << " in channel: " << wireId << endl;
      }

      // Find the best RecHit
      const type *theBestRecHit = findBestRecHit(layer, wireId, recHits, simHitWireDist);

      float recHitWireDist = recHitDistFromWire(*theBestRecHit, layer);
      if (debug_) {
        cout << "    SimHit distance from wire: " << simHitWireDist << endl
             << "    SimHit distance from FE:   " << simHitFEDist << endl
             << "    SimHit angle in layer RF:  " << simHitTheta << endl
             << "    RecHit distance from wire: " << recHitWireDist << endl;
      }
      float recHitErr = recHitPositionError(*theBestRecHit);
      HRes1DHit *hRes = nullptr;
      HRes1DHit *hResTot = nullptr;

      // Mirror angle in phi so that + and - wheels can be plotted together
      if (mirrorMinusWheels && wheel < 0 && sl != 2) {
        simHitTheta *= -1.;
        // Note: local X, if used, would have to be mirrored as well
      }

      // Fill residuals and pulls
      // Select the histo to be filled
      if (step == 1) {
        // Step 1
        if (sl != 2) {
          hResTot = histograms.hRes_S1RPhi.get();
          if (wheel == 0) {
            hRes = histograms.hRes_S1RPhi_W0.get();
          }
          if (abs(wheel) == 1) {
            hRes = histograms.hRes_S1RPhi_W1.get();
          }
          if (abs(wheel) == 2) {
            hRes = histograms.hRes_S1RPhi_W2.get();
          }
        } else {
          hResTot = histograms.hRes_S1RZ.get();
          if (wheel == 0) {
            hRes = histograms.hRes_S1RZ_W0.get();
          }
          if (abs(wheel) == 1) {
            hRes = histograms.hRes_S1RZ_W1.get();
          }
          if (abs(wheel) == 2) {
            hRes = histograms.hRes_S1RZ_W2.get();
          }
        }

      } else if (step == 2) {
        // Step 2
        if (sl != 2) {
          hRes = histograms.hRes_S2RPhi.get();
          if (wheel == 0) {
            hRes = histograms.hRes_S2RPhi_W0.get();
          }
          if (abs(wheel) == 1) {
            hRes = histograms.hRes_S2RPhi_W1.get();
          }
          if (abs(wheel) == 2) {
            hRes = histograms.hRes_S2RPhi_W2.get();
          }
        } else {
          hResTot = histograms.hRes_S2RZ.get();
          if (wheel == 0) {
            hRes = histograms.hRes_S2RZ_W0.get();
          }
          if (abs(wheel) == 1) {
            hRes = histograms.hRes_S2RZ_W1.get();
          }
          if (abs(wheel) == 2) {
            hRes = histograms.hRes_S2RZ_W2.get();
          }
        }

      } else if (step == 3) {
        // Step 3
        if (sl != 2) {
          hResTot = histograms.hRes_S3RPhi.get();
          if (wheel == 0) {
            hRes = histograms.hRes_S3RPhi_W0.get();
          }
          if (abs(wheel) == 1) {
            hRes = histograms.hRes_S3RPhi_W1.get();
          }
          if (abs(wheel) == 2) {
            hRes = histograms.hRes_S3RPhi_W2.get();
          }
          if (local_) {
            histograms.hRes_S3RPhiWS[abs(wheel)][wireId.station() - 1]->fill(simHitWireDist,
                                                                             simHitTheta,
                                                                             simHitFEDist,
                                                                             recHitWireDist,
                                                                             simHitGlobalPos.eta(),
                                                                             simHitGlobalPos.phi(),
                                                                             recHitErr,
                                                                             wireId.station());
          }
        } else {
          hResTot = histograms.hRes_S3RZ.get();
          if (wheel == 0) {
            hRes = histograms.hRes_S3RZ_W0.get();
          }
          if (abs(wheel) == 1) {
            hRes = histograms.hRes_S3RZ_W1.get();
          }
          if (abs(wheel) == 2) {
            hRes = histograms.hRes_S3RZ_W2.get();
          }
          if (local_) {
            histograms.hRes_S3RZWS[abs(wheel)][wireId.station() - 1]->fill(simHitWireDist,
                                                                           simHitTheta,
                                                                           simHitFEDist,
                                                                           recHitWireDist,
                                                                           simHitGlobalPos.eta(),
                                                                           simHitGlobalPos.phi(),
                                                                           recHitErr,
                                                                           wireId.station());
          }
        }
      }

      // Fill
      hRes->fill(simHitWireDist,
                 simHitTheta,
                 simHitFEDist,
                 recHitWireDist,
                 simHitGlobalPos.eta(),
                 simHitGlobalPos.phi(),
                 recHitErr,
                 wireId.station());
      if (hResTot != nullptr) {
        hResTot->fill(simHitWireDist,
                      simHitTheta,
                      simHitFEDist,
                      recHitWireDist,
                      simHitGlobalPos.eta(),
                      simHitGlobalPos.phi(),
                      recHitErr,
                      wireId.station());
      }
    }

    // Fill Efficiencies
    if (doall_) {
      HEff1DHit *hEff = nullptr;
      HEff1DHit *hEffTot = nullptr;
      if (step == 1) {
        // Step 1
        if (sl != 2) {
          hEff = histograms.hEff_S1RPhi.get();
          if (local_) {
            histograms.hEff_S1RPhiWS[abs(wheel)][wireId.station() - 1]->fill(
                simHitWireDist, simHitGlobalPos.eta(), simHitGlobalPos.phi(), recHitReconstructed);
          }
        } else {
          hEffTot = histograms.hEff_S1RZ.get();
          if (wheel == 0) {
            hEff = histograms.hEff_S1RZ_W0.get();
          }
          if (abs(wheel) == 1) {
            hEff = histograms.hEff_S1RZ_W1.get();
          }
          if (abs(wheel) == 2) {
            hEff = histograms.hEff_S1RZ_W2.get();
          }
          if (local_) {
            histograms.hEff_S1RZWS[abs(wheel)][wireId.station() - 1]->fill(
                simHitWireDist, simHitGlobalPos.eta(), simHitGlobalPos.phi(), recHitReconstructed);
          }
        }

      } else if (step == 2) {
        // Step 2
        if (sl != 2) {
          hEff = histograms.hEff_S2RPhi.get();
        } else {
          hEffTot = histograms.hEff_S2RZ.get();
          if (wheel == 0) {
            hEff = histograms.hEff_S2RZ_W0.get();
          }
          if (abs(wheel) == 1) {
            hEff = histograms.hEff_S2RZ_W1.get();
          }
          if (abs(wheel) == 2) {
            hEff = histograms.hEff_S2RZ_W2.get();
          }
        }

      } else if (step == 3) {
        // Step 3
        if (sl != 2) {
          hEff = histograms.hEff_S3RPhi.get();
          if (local_) {
            histograms.hEff_S3RPhiWS[abs(wheel)][wireId.station() - 1]->fill(
                simHitWireDist, simHitGlobalPos.eta(), simHitGlobalPos.phi(), recHitReconstructed);
          }
        } else {
          hEffTot = histograms.hEff_S3RZ.get();
          if (wheel == 0) {
            hEff = histograms.hEff_S3RZ_W0.get();
          }
          if (abs(wheel) == 1) {
            hEff = histograms.hEff_S3RZ_W1.get();
          }
          if (abs(wheel) == 2) {
            hEff = histograms.hEff_S3RZ_W2.get();
          }
          if (local_) {
            histograms.hEff_S3RZWS[abs(wheel)][wireId.station() - 1]->fill(
                simHitWireDist, simHitGlobalPos.eta(), simHitGlobalPos.phi(), recHitReconstructed);
          }
        }
      }
      // Fill
      hEff->fill(simHitWireDist, simHitGlobalPos.eta(), simHitGlobalPos.phi(), recHitReconstructed);
      if (hEffTot != nullptr) {
        hEffTot->fill(simHitWireDist, simHitGlobalPos.eta(), simHitGlobalPos.phi(), recHitReconstructed);
      }
    }
  }
}

// Return the error on the measured (cm) coordinate
float DTRecHitQuality::recHitPositionError(const DTRecHit1DPair &recHit) const {
  return sqrt(recHit.localPositionError(DTEnums::Left).xx());
}

// Return the error on the measured (cm) coordinate
float DTRecHitQuality::recHitPositionError(const DTRecHit1D &recHit) const {
  return sqrt(recHit.localPositionError().xx());
}

// declare this as a framework plugin
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(DTRecHitQuality);
