/*
 *  See header file for a description of this class.
 *
 *  \author S. Bolognesi and G. Cerminara - INFN Torino
 */

#include <iostream>
#include <map>

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DataFormats/DTRecHit/interface/DTRecHitCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment2DCollection.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/DTGeometry/interface/DTSuperLayer.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "Validation/DTRecHits/interface/DTHitQualityUtils.h"

#include "DTSegment2DQuality.h"
#include "Histograms.h"

using namespace std;
using namespace edm;

// Constructor
DTSegment2DQuality::DTSegment2DQuality(const ParameterSet& pset)  {
  // get the debug parameter for verbose output
  debug_ = pset.getUntrackedParameter<bool>("debug");
  DTHitQualityUtils::debug = debug_;
  // the name of the simhit collection
  simHitLabel_ = pset.getUntrackedParameter<InputTag>("simHitLabel");
  simHitToken_ = consumes<PSimHitContainer>(pset.getUntrackedParameter<InputTag>("simHitLabel"));
  // the name of the 2D rec hit collection
  segment2DLabel_ = pset.getUntrackedParameter<InputTag>("segment2DLabel");
  segment2DToken_ = consumes<DTRecSegment2DCollection>(pset.getUntrackedParameter<InputTag>("segment2DLabel"));

  // sigma resolution on position
  sigmaResPos_ = pset.getParameter<double>("sigmaResPos");
  // sigma resolution on angle
  sigmaResAngle_ = pset.getParameter<double>("sigmaResAngle");

  if (debug_)
    cout << "[DTSegment2DQuality] Constructor called " << endl;
}

void DTSegment2DQuality::beginRun(const edm::Run& iRun, const edm::EventSetup &setup) {

  // ----------------------
  // get hold of back-end interface
  dbe_ = nullptr;
  dbe_ = Service<DQMStore>().operator->();
  if ( dbe_ ) {
    if (debug_) {
      dbe_->setVerbose(1);
    } else {
      dbe_->setVerbose(0);
    }
  }
  if ( dbe_ ) {
    if (debug_) dbe_->showDirStructure();
  }

  h2DHitRPhi_ = new HRes2DHit ("RPhi", dbe_, true, true);
  h2DHitRZ_ = new HRes2DHit ("RZ", dbe_, true, true);
  h2DHitRZ_W0_ = new HRes2DHit ("RZ_W0", dbe_, true, true);
  h2DHitRZ_W1_ = new HRes2DHit ("RZ_W1", dbe_, true, true);
  h2DHitRZ_W2_ = new HRes2DHit ("RZ_W2", dbe_, true, true);

  h2DHitEff_RPhi_ = new HEff2DHit ("RPhi", dbe_);
  h2DHitEff_RZ_ = new HEff2DHit ("RZ", dbe_);
  h2DHitEff_RZ_W0_ = new HEff2DHit ("RZ_W0", dbe_);
  h2DHitEff_RZ_W1_ = new HEff2DHit ("RZ_W1", dbe_);
  h2DHitEff_RZ_W2_ = new HEff2DHit ("RZ_W2", dbe_);
  if (debug_)
    cout << "[DTSegment2DQuality] hitsos created " << endl;
}

/* FIXME these shoud be moved to the harvesting step
void DTSegment2DQuality::endJob() {
  h2DHitEff_RPhi_->computeEfficiency();
  h2DHitEff_RZ_->computeEfficiency();
  h2DHitEff_RZ_W0_->computeEfficiency();
  h2DHitEff_RZ_W1_->computeEfficiency();
  h2DHitEff_RZ_W2_->computeEfficiency();
}
*/

// The real analysis
void DTSegment2DQuality::analyze(const Event & event, const EventSetup& eventSetup) {
  // theFile->cd();

  // Get the DT Geometry
  ESHandle<DTGeometry> dtGeom;
  eventSetup.get<MuonGeometryRecord>().get(dtGeom);

  // Get the SimHit collection from the event
  edm::Handle<PSimHitContainer> simHits;
  event.getByToken(simHitToken_, simHits); // FIXME: second string to be removed

  // Map simHits by sl
  map<DTSuperLayerId, PSimHitContainer > simHitsPerSl;
  for (PSimHitContainer::const_iterator simHit = simHits->begin();
      simHit != simHits->end(); simHit++) {
    // Create the id of the sl (the simHits in the DT known their wireId)
    DTSuperLayerId slId = ((DTWireId(simHit->detUnitId())).layerId()).superlayerId();
    // Fill the map
    simHitsPerSl[slId].push_back(*simHit);
  }

  // Get the 2D rechits from the event
  Handle<DTRecSegment2DCollection> segment2Ds;
  event.getByToken(segment2DToken_, segment2Ds);

  if (not segment2Ds.isValid()) {
    if (debug_) cout << "[DTSegment2DQuality]**Warning: no 2DSegments with label: " << segment2DLabel_
      << " in this event, skipping !" << endl;
    return;
  }

  // Loop over all superlayers containing a segment
  DTRecSegment2DCollection::id_iterator slId;
  for (slId = segment2Ds->id_begin();
      slId != segment2Ds->id_end();
      ++slId) {

    //------------------------- simHits ---------------------------//
    // Get simHits of each superlayer
    PSimHitContainer simHits =  simHitsPerSl[(*slId)];

    // Map simhits per wire
    map<DTWireId, PSimHitContainer> simHitsPerWire = DTHitQualityUtils::mapSimHitsPerWire(simHits);
    map<DTWireId, const PSimHit*> muSimHitPerWire = DTHitQualityUtils::mapMuSimHitsPerWire(simHitsPerWire);
    int nMuSimHit = muSimHitPerWire.size();
    if (nMuSimHit == 0 or nMuSimHit == 1) {
      if (debug_ and nMuSimHit == 1)
        cout << "[DTSegment2DQuality] Only " << nMuSimHit << " mu SimHit in this SL, skipping !" << endl;
      continue; // If no or only one mu SimHit is found skip this SL
    }
    if (debug_)
      cout << "=== SL " << (*slId) << " has " << nMuSimHit << " SimHits" << endl;

    // Find outer and inner mu SimHit to build a segment
    pair<const PSimHit*, const PSimHit*> inAndOutSimHit = DTHitQualityUtils::findMuSimSegment(muSimHitPerWire);
    // Check that outermost and innermost SimHit are not the same
    if (inAndOutSimHit.first == inAndOutSimHit.second ) {
      cout << "[DTHitQualityUtils]***Warning: outermost and innermost SimHit are the same !" << endl;
      continue;
    }

    // Find direction and position of the sim Segment in SL RF
    pair<LocalVector, LocalPoint> dirAndPosSimSegm = DTHitQualityUtils::findMuSimSegmentDirAndPos(inAndOutSimHit,
        (*slId),&(*dtGeom));

    LocalVector simSegmLocalDir = dirAndPosSimSegm.first;
    LocalPoint simSegmLocalPos = dirAndPosSimSegm.second;
    if (debug_)
      cout << "  Simulated segment:  local direction " << simSegmLocalDir << endl
           << "                      local position  " << simSegmLocalPos << endl;
    const DTSuperLayer* superLayer = dtGeom->superLayer(*slId);
    GlobalPoint simSegmGlobalPos = superLayer->toGlobal(simSegmLocalPos);

    // Atan(x/z) angle and x position in SL RF
    float angleSimSeg = DTHitQualityUtils::findSegmentAlphaAndBeta(simSegmLocalDir).first;
    float posSimSeg = simSegmLocalPos.x();
    // Position (in eta, phi coordinates) in the global RF
    float etaSimSeg = simSegmGlobalPos.eta();
    float phiSimSeg = simSegmGlobalPos.phi();

    //---------------------------- recHits --------------------------//
    // Get the range of rechit for the corresponding slId
    bool recHitFound = false;
    DTRecSegment2DCollection::range range = segment2Ds->get(*slId);
    int nsegm = distance(range.first, range.second);
    if (debug_)
      cout << "   Sl: " << *slId << " has " << nsegm
           << " 2D segments" << endl;

    if (nsegm != 0) {
      // Find the best RecHit: look for the 2D RecHit with the angle closest
      // to that of segment made of SimHits.
      // RecHits must have delta alpha and delta position within 5 sigma of
      // the residual distribution (we are looking for residuals of segments
      // usefull to the track fit) for efficency purpose
      const DTRecSegment2D* bestRecHit = nullptr;
      bool bestRecHitFound = false;
      double deltaAlpha = 99999;

      // Loop over the recHits of this slId
      for (DTRecSegment2DCollection::const_iterator segment2D = range.first;
          segment2D != range.second;
          ++segment2D) {
        // Check the dimension
        if ((*segment2D).dimension() != 2) {
          if (debug_) cout << "[DTSegment2DQuality]***Error: This is not 2D segment !!!" << endl;
          abort();
        }
        // Segment Local Direction and position (in SL RF)
        LocalVector recSegDirection = (*segment2D).localDirection();
        LocalPoint recSegPosition = (*segment2D).localPosition();

        float recSegAlpha = DTHitQualityUtils::findSegmentAlphaAndBeta(recSegDirection).first;
        if (debug_)
          cout << "  RecSegment direction: " << recSegDirection << endl
               << "             position : " << recSegPosition << endl
               << "             alpha    : " << recSegAlpha << endl;

        if (fabs(recSegAlpha - angleSimSeg) < deltaAlpha) {
          deltaAlpha = fabs(recSegAlpha - angleSimSeg);
          bestRecHit = &(*segment2D);
          bestRecHitFound = true;
        }
      }  // End of Loop over all 2D RecHits

      if (bestRecHitFound) {
        // Best rechit direction and position in SL RF
        LocalPoint bestRecHitLocalPos = bestRecHit->localPosition();
        LocalVector bestRecHitLocalDir = bestRecHit->localDirection();

        LocalError bestRecHitLocalPosErr = bestRecHit->localPositionError();
        LocalError bestRecHitLocalDirErr = bestRecHit->localDirectionError();

        float angleBestRHit = DTHitQualityUtils::findSegmentAlphaAndBeta(bestRecHitLocalDir).first;

        if (fabs(angleBestRHit - angleSimSeg) < 5*sigmaResAngle_ and
            fabs(bestRecHitLocalPos.x() - posSimSeg) < 5*sigmaResPos_) {
          recHitFound = true;
        }

        // Fill Residual histos
        HRes2DHit *hRes = nullptr;
        if ((*slId).superlayer() == 1 or (*slId).superlayer() == 3) { // RPhi SL
          hRes = h2DHitRPhi_;
        } else if ((*slId).superlayer() == 2) { // RZ SL
          h2DHitRZ_->Fill(angleSimSeg,
              angleBestRHit,
              posSimSeg,
              bestRecHitLocalPos.x(),
              etaSimSeg,
              phiSimSeg,
              sqrt(bestRecHitLocalPosErr.xx()),
              sqrt(bestRecHitLocalDirErr.xx())
              );
          if (abs((*slId).wheel()) == 0)
            hRes = h2DHitRZ_W0_;
          else if (abs((*slId).wheel()) == 1)
            hRes = h2DHitRZ_W1_;
          else if (abs((*slId).wheel()) == 2)
            hRes = h2DHitRZ_W2_;
        }
        hRes->Fill(angleSimSeg,
            angleBestRHit,
            posSimSeg,
            bestRecHitLocalPos.x(),
            etaSimSeg,
            phiSimSeg,
            sqrt(bestRecHitLocalPosErr.xx()),
            sqrt(bestRecHitLocalDirErr.xx())
            );
      }
    } // end of if (nsegm != 0)

    // Fill Efficiency plot
    HEff2DHit *hEff = nullptr;
    if ((*slId).superlayer() == 1 or (*slId).superlayer() == 3) { // RPhi SL
      hEff = h2DHitEff_RPhi_;
    } else if ((*slId).superlayer() == 2) { // RZ SL
      h2DHitEff_RZ_->Fill(etaSimSeg, phiSimSeg, posSimSeg, angleSimSeg, recHitFound);
      if (abs((*slId).wheel()) == 0)
        hEff = h2DHitEff_RZ_W0_;
      else if (abs((*slId).wheel()) == 1)
        hEff = h2DHitEff_RZ_W1_;
      else if (abs((*slId).wheel()) == 2)
        hEff = h2DHitEff_RZ_W2_;
    }
    hEff->Fill(etaSimSeg, phiSimSeg, posSimSeg, angleSimSeg, recHitFound);
  } // End of loop over superlayers
}
