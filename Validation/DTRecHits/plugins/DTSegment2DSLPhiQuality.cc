/*
 *  See header file for a description of this class.
 *
 *  \author S. Bolognesi and G. Cerminara - INFN Torino
 */

#include <iostream>
#include <map>

#include "DataFormats/DTRecHit/interface/DTRecHitCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "Geometry/DTGeometry/interface/DTChamber.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "Validation/DTRecHits/interface/DTHitQualityUtils.h"

#include "DTSegment2DSLPhiQuality.h"
#include "Histograms.h"

using namespace std;
using namespace edm;

namespace dtsegment2dsl {
  struct Histograms {
    std::unique_ptr<HRes2DHit> h2DHitSuperPhi;
    std::unique_ptr<HEff2DHit> h2DHitEff_SuperPhi;
  };
}  // namespace dtsegment2dsl

using namespace dtsegment2dsl;

// Constructor
DTSegment2DSLPhiQuality::DTSegment2DSLPhiQuality(const ParameterSet &pset) {
  // Get the debug parameter for verbose output
  debug_ = pset.getUntrackedParameter<bool>("debug");
  DTHitQualityUtils::debug = debug_;

  // the name of the simhit collection
  simHitLabel_ = pset.getUntrackedParameter<InputTag>("simHitLabel");
  simHitToken_ = consumes<PSimHitContainer>(pset.getUntrackedParameter<InputTag>("simHitLabel"));
  // the name of the 2D rec hit collection
  segment4DLabel_ = pset.getUntrackedParameter<InputTag>("segment4DLabel");
  segment4DToken_ = consumes<DTRecSegment4DCollection>(pset.getUntrackedParameter<InputTag>("segment4DLabel"));

  // sigma resolution on position
  sigmaResPos_ = pset.getParameter<double>("sigmaResPos");
  // sigma resolution on angle
  sigmaResAngle_ = pset.getParameter<double>("sigmaResAngle");
  doall_ = pset.getUntrackedParameter<bool>("doall", false);
  local_ = pset.getUntrackedParameter<bool>("local", false);
}

void DTSegment2DSLPhiQuality::bookHistograms(DQMStore::IBooker &booker,
                                             edm::Run const &run,
                                             edm::EventSetup const &setup,
                                             Histograms &histograms) const {
  // Book the histos
  histograms.h2DHitSuperPhi = std::make_unique<HRes2DHit>("SuperPhi", booker, doall_, local_);
  if (doall_) {
    histograms.h2DHitEff_SuperPhi = std::make_unique<HEff2DHit>("SuperPhi", booker);
  }
}

// The real analysis
void DTSegment2DSLPhiQuality::dqmAnalyze(edm::Event const &event,
                                         edm::EventSetup const &setup,
                                         Histograms const &histograms) const {
  // Get the DT Geometry
  ESHandle<DTGeometry> dtGeom;
  setup.get<MuonGeometryRecord>().get(dtGeom);

  // Get the SimHit collection from the event
  edm::Handle<PSimHitContainer> simHits;
  event.getByToken(simHitToken_, simHits);  // FIXME: second string to be removed

  // Map simHits by chamber
  map<DTChamberId, PSimHitContainer> simHitsPerCh;
  for (const auto &simHit : *simHits) {
    // Create the id of the chamber (the simHits in the DT known their wireId)
    DTChamberId chamberId = (((DTWireId(simHit.detUnitId())).layerId()).superlayerId()).chamberId();
    // Fill the map
    simHitsPerCh[chamberId].push_back(simHit);
  }

  // Get the 4D rechits from the event
  Handle<DTRecSegment4DCollection> segment4Ds;
  event.getByToken(segment4DToken_, segment4Ds);

  if (!segment4Ds.isValid()) {
    if (debug_) {
      cout << "[DTSegment2DSLPhiQuality]**Warning: no 4D Segments with label: " << segment4DLabel_
           << " in this event, skipping!" << endl;
    }
    return;
  }

  // Loop over all chambers containing a segment
  DTRecSegment4DCollection::id_iterator chamberId;
  for (chamberId = segment4Ds->id_begin(); chamberId != segment4Ds->id_end(); ++chamberId) {
    //------------------------- simHits ---------------------------//
    // Get simHits of each chamber
    PSimHitContainer simHits = simHitsPerCh[(*chamberId)];

    // Map simhits per wire
    map<DTWireId, PSimHitContainer> simHitsPerWire = DTHitQualityUtils::mapSimHitsPerWire(simHits);
    map<DTWireId, const PSimHit *> muSimHitPerWire = DTHitQualityUtils::mapMuSimHitsPerWire(simHitsPerWire);
    int nMuSimHit = muSimHitPerWire.size();
    if (nMuSimHit == 0 || nMuSimHit == 1) {
      if (debug_ && nMuSimHit == 1) {
        cout << "[DTSegment2DSLPhiQuality] Only " << nMuSimHit << " mu SimHit in this chamber, skipping!" << endl;
      }
      continue;  // If no or only one mu SimHit is found skip this chamber
    }
    if (debug_) {
      cout << "=== Chamber " << (*chamberId) << " has " << nMuSimHit << " SimHits" << endl;
    }

    // Find outer and inner mu SimHit to build a segment
    pair<const PSimHit *, const PSimHit *> inAndOutSimHit = DTHitQualityUtils::findMuSimSegment(muSimHitPerWire);

    // Find direction and position of the sim Segment in Chamber RF
    pair<LocalVector, LocalPoint> dirAndPosSimSegm =
        DTHitQualityUtils::findMuSimSegmentDirAndPos(inAndOutSimHit, (*chamberId), &(*dtGeom));

    LocalVector simSegmLocalDir = dirAndPosSimSegm.first;
    LocalPoint simSegmLocalPos = dirAndPosSimSegm.second;
    const DTChamber *chamber = dtGeom->chamber(*chamberId);
    GlobalPoint simSegmGlobalPos = chamber->toGlobal(simSegmLocalPos);

    // Atan(x/z) angle and x position in SL RF
    float angleSimSeg = DTHitQualityUtils::findSegmentAlphaAndBeta(simSegmLocalDir).first;
    float posSimSeg = simSegmLocalPos.x();
    // Position (in eta, phi coordinates) in lobal RF
    float etaSimSeg = simSegmGlobalPos.eta();
    float phiSimSeg = simSegmGlobalPos.phi();

    if (debug_) {
      cout << "  Simulated segment:  local direction " << simSegmLocalDir << endl
           << "                      local position  " << simSegmLocalPos << endl
           << "                      angle           " << angleSimSeg << endl;
    }

    //---------------------------- recHits --------------------------//
    // Get the range of rechit for the corresponding chamberId
    bool recHitFound = false;
    DTRecSegment4DCollection::range range = segment4Ds->get(*chamberId);
    int nsegm = distance(range.first, range.second);
    if (debug_) {
      cout << "   Chamber: " << *chamberId << " has " << nsegm << " 4D segments" << endl;
    }

    if (nsegm != 0) {
      // Find the best RecHit: look for the 4D RecHit with the phi angle closest
      // to that of segment made of SimHits.
      // RecHits must have delta alpha and delta position within 5 sigma of
      // the residual distribution (we are looking for residuals of segments
      // usefull to the track fit) for efficency purpose
      const DTRecSegment2D *bestRecHit = nullptr;
      bool bestRecHitFound = false;
      double deltaAlpha = 99999;

      // Loop over the recHits of this chamberId
      for (DTRecSegment4DCollection::const_iterator segment4D = range.first; segment4D != range.second; ++segment4D) {
        // Check the dimension
        if ((*segment4D).dimension() != 4) {
          if (debug_) {
            cout << "[DTSegment2DSLPhiQuality]***Error: This is not 4D "
                    "segment!!!"
                 << endl;
          }
          continue;
        }

        // Get 2D superPhi segments from 4D segments
        const DTChamberRecSegment2D *phiSegment2D = (*segment4D).phiSegment();
        if ((*phiSegment2D).dimension() != 2) {
          if (debug_) {
            cout << "[DTSegment2DQuality]***Error: This is not 2D segment!!!" << endl;
          }
          abort();
        }

        // Segment Local Direction and position (in Chamber RF)
        LocalVector recSegDirection = (*phiSegment2D).localDirection();

        float recSegAlpha = DTHitQualityUtils::findSegmentAlphaAndBeta(recSegDirection).first;
        if (debug_) {
          cout << "  RecSegment direction: " << recSegDirection << endl
               << "             position : " << (*phiSegment2D).localPosition() << endl
               << "             alpha    : " << recSegAlpha << endl;
        }

        if (fabs(recSegAlpha - angleSimSeg) < deltaAlpha) {
          deltaAlpha = fabs(recSegAlpha - angleSimSeg);
          bestRecHit = &(*phiSegment2D);
          bestRecHitFound = true;
        }
      }  // End of Loop over all 4D RecHits of this chambers

      if (bestRecHitFound) {
        // Best rechit direction and position in Chamber RF
        LocalPoint bestRecHitLocalPos = bestRecHit->localPosition();
        LocalVector bestRecHitLocalDir = bestRecHit->localDirection();

        LocalError bestRecHitLocalPosErr = bestRecHit->localPositionError();
        LocalError bestRecHitLocalDirErr = bestRecHit->localDirectionError();

        float angleBestRHit = DTHitQualityUtils::findSegmentAlphaAndBeta(bestRecHitLocalDir).first;
        if (fabs(angleBestRHit - angleSimSeg) < 5 * sigmaResAngle_ &&
            fabs(bestRecHitLocalPos.x() - posSimSeg) < 5 * sigmaResPos_) {
          recHitFound = true;
        }

        // Fill Residual histos
        histograms.h2DHitSuperPhi->fill(angleSimSeg,
                                        angleBestRHit,
                                        posSimSeg,
                                        bestRecHitLocalPos.x(),
                                        etaSimSeg,
                                        phiSimSeg,
                                        sqrt(bestRecHitLocalPosErr.xx()),
                                        sqrt(bestRecHitLocalDirErr.xx()));
      }
    }  // end of if (nsegm!= 0)

    // Fill Efficiency plot
    if (doall_) {
      histograms.h2DHitEff_SuperPhi->fill(etaSimSeg, phiSimSeg, posSimSeg, angleSimSeg, recHitFound);
    }
  }  // End of loop over chambers
}

// declare this as a framework plugin
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(DTSegment2DSLPhiQuality);
