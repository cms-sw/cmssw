/*
 *  See header file for a description of this class.
 *
 *  \author G. Cerminara - INFN Torino
 */

#include <iostream>
#include <map>

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Validation/DTRecHits/interface/DTHitQualityUtils.h"

#include "DTRecHitQuality.h"

using namespace std;
using namespace edm;

// In phi SLs, The dependency on X and angle is specular in positive
// and negative wheels. Since positive and negative wheels are filled
// together into the same plots, it is useful to mirror negative wheels
// so that the actual dependency can be observerd instead of an artificially
// simmetrized one.
// Set mirrorMinusWheels to avoid this.
namespace {
  bool mirrorMinusWheels = true;
}

// Constructor
DTRecHitQuality::DTRecHitQuality(const ParameterSet& pset) {
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
  doall_   = pset.getUntrackedParameter<bool>("doall", false);
  local_   = pset.getUntrackedParameter<bool>("local", true);
}

void DTRecHitQuality::beginRun(const edm::Run& iRun, const edm::EventSetup &setup) {

  // ----------------------
  // get hold of back-end interface
  dbe_ = nullptr;
  dbe_ = Service<DQMStore>().operator->();
  dbe_->setVerbose(0);
  if (doall_ && doStep1_) {
    hRes_S1RPhi_ = new HRes1DHit("S1RPhi", dbe_, true, local_);    // RecHits, 1. step, RPhi
    hRes_S1RPhi_W0_ = new HRes1DHit("S1RPhi_W0", dbe_, true, local_);   // RecHits, 1. step, RZ, wheel 0
    hRes_S1RPhi_W1_ = new HRes1DHit("S1RPhi_W1", dbe_, true, local_);   // RecHits, 1. step, RZ, wheel +-1
    hRes_S1RPhi_W2_ = new HRes1DHit("S1RPhi_W2", dbe_, true, local_);   // RecHits, 1. step, RZ, wheel +-2
    hRes_S1RZ_ = new HRes1DHit("S1RZ", dbe_, true, local_);         // RecHits, 1. step, RZ
    hRes_S1RZ_W0_ = new HRes1DHit("S1RZ_W0", dbe_, true, local_);   // RecHits, 1. step, RZ, wheel 0
    hRes_S1RZ_W1_ = new HRes1DHit("S1RZ_W1", dbe_, true, local_);   // RecHits, 1. step, RZ, wheel +-1
    hRes_S1RZ_W2_ = new HRes1DHit("S1RZ_W2", dbe_, true, local_);   // RecHits, 1. step, RZ, wheel +-2
    hEff_S1RPhi_ = new HEff1DHit("S1RPhi", dbe_);     // RecHits, 1. step, RPhi
    hEff_S1RZ_ = new HEff1DHit("S1RZ", dbe_);         // RecHits, 1. step, RZ
    hEff_S1RZ_W0_ = new HEff1DHit("S1RZ_W0", dbe_);   // RecHits, 1. step, RZ, wheel 0
    hEff_S1RZ_W1_ = new HEff1DHit("S1RZ_W1", dbe_);   // RecHits, 1. step, RZ, wheel +-1
    hEff_S1RZ_W2_ = new HEff1DHit("S1RZ_W2", dbe_);   // RecHits, 1. step, RZ, wheel +-2
  }
  if (doall_ && doStep2_) {
    hRes_S2RPhi_ = new HRes1DHit("S2RPhi", dbe_, true, local_);     // RecHits, 2. step, RPhi
    hRes_S2RPhi_W0_ = new HRes1DHit("S2RPhi_W0", dbe_, true, local_);   // RecHits, 2. step, RPhi, wheel 0
    hRes_S2RPhi_W1_ = new HRes1DHit("S2RPhi_W1", dbe_, true, local_);   // RecHits, 2. step, RPhi, wheel +-1
    hRes_S2RPhi_W2_ = new HRes1DHit("S2RPhi_W2", dbe_, true, local_);   // RecHits, 2. step, RPhi, wheel +-2
    hRes_S2RZ_ = new HRes1DHit("S2RZ", dbe_, true, local_);	    // RecHits, 2. step, RZ
    hRes_S2RZ_W0_ = new HRes1DHit("S2RZ_W0", dbe_, true, local_);   // RecHits, 2. step, RZ, wheel 0
    hRes_S2RZ_W1_ = new HRes1DHit("S2RZ_W1", dbe_, true, local_);   // RecHits, 2. step, RZ, wheel +-1
    hRes_S2RZ_W2_ = new HRes1DHit("S2RZ_W2", dbe_, true, local_);   // RecHits, 2. step, RZ, wheel +-2
    hEff_S2RPhi_ = new HEff1DHit("S2RPhi", dbe_);     // RecHits, 2. step, RPhi
    hEff_S2RZ_W0_ = new HEff1DHit("S2RZ_W0", dbe_);   // RecHits, 2. step, RZ, wheel 0
    hEff_S2RZ_W1_ = new HEff1DHit("S2RZ_W1", dbe_);   // RecHits, 2. step, RZ, wheel +-1
    hEff_S2RZ_W2_ = new HEff1DHit("S2RZ_W2", dbe_);   // RecHits, 2. step, RZ, wheel +-2
    hEff_S2RZ_ = new HEff1DHit("S2RZ", dbe_);	    // RecHits, 2. step, RZ
  }
  if (doStep3_) {
    hRes_S3RPhi_ = new HRes1DHit("S3RPhi", dbe_, doall_, local_);     // RecHits, 3. step, RPhi
    hRes_S3RPhi_W0_ = new HRes1DHit("S3RPhi_W0", dbe_, doall_, local_);   // RecHits, 3. step, RPhi, wheel 0
    hRes_S3RPhi_W1_ = new HRes1DHit("S3RPhi_W1", dbe_, doall_, local_);   // RecHits, 3. step, RPhi, wheel +-1
    hRes_S3RPhi_W2_ = new HRes1DHit("S3RPhi_W2", dbe_, doall_, local_);   // RecHits, 3. step, RPhi, wheel +-2
    hRes_S3RZ_ = new HRes1DHit("S3RZ", dbe_, doall_, local_);	    // RecHits, 3. step, RZ
    hRes_S3RZ_W0_ = new HRes1DHit("S3RZ_W0", dbe_, doall_, local_);   // RecHits, 3. step, RZ, wheel 0
    hRes_S3RZ_W1_ = new HRes1DHit("S3RZ_W1", dbe_, doall_, local_);   // RecHits, 3. step, RZ, wheel +-1
    hRes_S3RZ_W2_ = new HRes1DHit("S3RZ_W2", dbe_, doall_, local_);   // RecHits, 3. step, RZ, wheel +-2

    if (local_) {
      // Plots with finer granularity, not to be included in DQM
      TString name1 ="RPhi_W";
      TString name2 ="RZ_W";
      for (long w = 0;w<= 2;++w) {
	for (long s = 1;s<= 4;++s) {
	  hRes_S3RPhiWS_[w][s-1] = new HRes1DHit(("S3"+name1+w+"_St"+s).Data(), dbe_, doall_, local_);
	  hEff_S1RPhiWS_[w][s-1] = new HEff1DHit(("S1"+name1+w+"_St"+s).Data(), dbe_);
	  hEff_S3RPhiWS_[w][s-1] = new HEff1DHit(("S3"+name1+w+"_St"+s).Data(), dbe_);
	  if (s!= 4) {
	    hRes_S3RZWS_[w][s-1] = new HRes1DHit(("S3"+name2+w+"_St"+s).Data(), dbe_, doall_, local_);
	    hEff_S1RZWS_[w][s-1] = new HEff1DHit(("S1"+name2+w+"_St"+s).Data(), dbe_);
	    hEff_S3RZWS_[w][s-1] = new HEff1DHit(("S3"+name2+w+"_St"+s).Data(), dbe_);
	  }
	}
      }
    }


    if (doall_) {
      hEff_S3RPhi_ = new HEff1DHit("S3RPhi", dbe_);     // RecHits, 3. step, RPhi
      hEff_S3RZ_ = new HEff1DHit("S3RZ", dbe_);	    // RecHits, 3. step, RZ
      hEff_S3RZ_W0_ = new HEff1DHit("S3RZ_W0", dbe_);   // RecHits, 3. step, RZ, wheel 0
      hEff_S3RZ_W1_ = new HEff1DHit("S3RZ_W1", dbe_);   // RecHits, 3. step, RZ, wheel +-1
      hEff_S3RZ_W2_ = new HEff1DHit("S3RZ_W2", dbe_);   // RecHits, 3. step, RZ, wheel +-2
    }
  }
}

/* FIXME these shoud be moved to the harvesting step
void DTRecHitQuality::endJob() {
  // Write the histos to file
  if (doall_) {
    if (doStep1_) {
      hEff_S1RPhi_->computeEfficiency();
      hEff_S1RZ_->computeEfficiency();
      hEff_S1RZ_W0_->computeEfficiency();
      hEff_S1RZ_W1_->computeEfficiency();
      hEff_S1RZ_W2_->computeEfficiency();
    }
    if (doStep2_) {
      hEff_S2RPhi_->computeEfficiency();
      hEff_S2RZ_->computeEfficiency();
      hEff_S2RZ_W0_->computeEfficiency();
      hEff_S2RZ_W1_->computeEfficiency();
      hEff_S2RZ_W2_->computeEfficiency();
    }
    if (doStep3_) {
      hEff_S3RPhi_->computeEfficiency();
      hEff_S3RZ_->computeEfficiency();
      hEff_S3RZ_W0_->computeEfficiency();
      hEff_S3RZ_W1_->computeEfficiency();
      hEff_S3RZ_W2_->computeEfficiency();
    }
  }
}
*/

// The real analysis
  void DTRecHitQuality::analyze(const Event & event, const EventSetup& eventSetup) {
    if (debug_)
      cout << "--- [DTRecHitQuality] Analysing Event: #Run: " << event.id().run()
        << " #Event: " << event.id().event() << endl;

    // Get the DT Geometry
    ESHandle<DTGeometry> dtGeom;
    eventSetup.get<MuonGeometryRecord>().get(dtGeom);

    // Get the SimHit collection from the event
    Handle<PSimHitContainer> simHits;
    event.getByToken(simHitToken_, simHits);

    // Map simhits per wire
    map<DTWireId, PSimHitContainer > simHitsPerWire =
      DTHitQualityUtils::mapSimHitsPerWire(*(simHits.product()));

    //=======================================================================================
    // RecHit analysis at Step 1
    if (doStep1_ && doall_) {
      if (debug_)
        cout << "  -- DTRecHit S1: begin analysis:" << endl;
      // Get the rechit collection from the event
      Handle<DTRecHitCollection> dtRecHits;
      event.getByToken(recHitToken_, dtRecHits);

      if (!dtRecHits.isValid()) {
	if (debug_) cout << "[DTRecHitQuality]**Warning: no 1DRechits with label: " << recHitLabel_ << " in this event, skipping!" << endl;
	return;
      }

     // Map rechits per wire
      map<DTWireId, vector<DTRecHit1DPair> > recHitsPerWire =
        map1DRecHitsPerWire(dtRecHits.product());

      compute(dtGeom.product(), simHitsPerWire, recHitsPerWire, 1);
    }


    //=======================================================================================
    // RecHit analysis at Step 2
    if (doStep2_ && doall_) {
      if (debug_)
        cout << "  -- DTRecHit S2: begin analysis:" << endl;

      // Get the 2D rechits from the event
      Handle<DTRecSegment2DCollection> segment2Ds;
      event.getByToken(segment2DToken_, segment2Ds);

      if (!segment2Ds.isValid()) {
       if (debug_) cout << "[DTRecHitQuality]**Warning: no 2DSegments with label: " << segment2DLabel_
		      << " in this event, skipping!" << endl;

      }
      else{
	// Map rechits per wire
	map<DTWireId, vector<DTRecHit1D> > recHitsPerWire =
	  map1DRecHitsPerWire(segment2Ds.product());

	compute(dtGeom.product(), simHitsPerWire, recHitsPerWire, 2);
      }
    }

    //=======================================================================================
    // RecHit analysis at Step 3
    if (doStep3_) {
      if (debug_)
        cout << "  -- DTRecHit S3: begin analysis:" << endl;

      // Get the 4D rechits from the event
      Handle<DTRecSegment4DCollection> segment4Ds;
      event.getByToken(segment4DToken_, segment4Ds);

      if (!segment4Ds.isValid()) {
        if (debug_) cout << "[DTRecHitQuality]**Warning: no 4D Segments with label: " << segment4DLabel_
		       << " in this event, skipping!" << endl;
	return;
      }

      // Map rechits per wire
      map<DTWireId, vector<DTRecHit1D> > recHitsPerWire =
        map1DRecHitsPerWire(segment4Ds.product());

      compute(dtGeom.product(), simHitsPerWire, recHitsPerWire, 3);
    }

  }



// Return a map between DTRecHit1DPair and wireId
map<DTWireId, vector<DTRecHit1DPair> >
DTRecHitQuality::map1DRecHitsPerWire(const DTRecHitCollection* dt1DRecHitPairs) {
  map<DTWireId, vector<DTRecHit1DPair> > ret;

  for (DTRecHitCollection::const_iterator rechit = dt1DRecHitPairs->begin();
      rechit != dt1DRecHitPairs->end(); rechit++) {
    ret[(*rechit).wireId()].push_back(*rechit);
  }

  return ret;
}


// Return a map between DTRecHit1D at S2 and wireId
map<DTWireId, vector<DTRecHit1D> >
DTRecHitQuality::map1DRecHitsPerWire(const DTRecSegment2DCollection* segment2Ds) {
  map<DTWireId, vector<DTRecHit1D> > ret;

  // Loop over all 2D segments
  for (DTRecSegment2DCollection::const_iterator segment = segment2Ds->begin();
      segment != segment2Ds->end();
      segment++) {
    vector<DTRecHit1D> component1DHits = (*segment).specificRecHits();
    // Loop over all component 1D hits
    for (vector<DTRecHit1D>::const_iterator hit = component1DHits.begin();
        hit != component1DHits.end();
        hit++) {
      ret[(*hit).wireId()].push_back(*hit);
    }
  }
  return ret;
}



// Return a map between DTRecHit1D at S3 and wireId
map<DTWireId, std::vector<DTRecHit1D> >
DTRecHitQuality::map1DRecHitsPerWire(const DTRecSegment4DCollection* segment4Ds) {
  map<DTWireId, vector<DTRecHit1D> > ret;
  // Loop over all 4D segments
  for (DTRecSegment4DCollection::const_iterator segment = segment4Ds->begin();
      segment != segment4Ds->end();
      segment++) {
    // Get component 2D segments
    vector<const TrackingRecHit*> segment2Ds = (*segment).recHits();
    // Loop over 2D segments:
    for (vector<const TrackingRecHit*>::const_iterator segment2D = segment2Ds.begin();
        segment2D != segment2Ds.end();
        segment2D++) {
      // Get 1D component rechits
      vector<const TrackingRecHit*> hits = (*segment2D)->recHits();
      // Loop over them
      for (vector<const TrackingRecHit*>::const_iterator hit = hits.begin();
          hit != hits.end(); hit++) {
        const DTRecHit1D* hit1D = dynamic_cast<const DTRecHit1D*>(*hit);
        ret[hit1D->wireId()].push_back(*hit1D);
      }
    }
  }

  return ret;
}

// Compute SimHit distance from wire (cm)
float DTRecHitQuality::simHitDistFromWire(const DTLayer* layer,
                                          DTWireId wireId,
                                          const PSimHit& hit) {
  float xwire = layer->specificTopology().wirePosition(wireId.wire());
  LocalPoint entryP = hit.entryPoint();
  LocalPoint exitP = hit.exitPoint();
  float xEntry = entryP.x()-xwire;
  float xExit  = exitP.x()-xwire;

  return fabs(xEntry - (entryP.z()*(xExit-xEntry))/(exitP.z()-entryP.z()));// FIXME: check...
}

// Compute SimHit impact angle (in direction perp to wire), in the SL RF
float DTRecHitQuality::simHitImpactAngle(const DTLayer* layer,
                                         DTWireId wireId,
                                         const PSimHit& hit) {
  LocalPoint entryP = hit.entryPoint();
  LocalPoint exitP = hit.exitPoint();
  float theta =(exitP.x()-entryP.x())/(exitP.z()-entryP.z());
  return atan(theta);
}

// Compute SimHit distance from FrontEnd
float DTRecHitQuality::simHitDistFromFE(const DTLayer* layer,
                                        DTWireId wireId,
                                        const PSimHit& hit) {
  LocalPoint entryP = hit.entryPoint();
  LocalPoint exitP = hit.exitPoint();
  float wireLenght = layer->specificTopology().cellLenght();
  // FIXME: should take only wireLenght/2.;
  // moreover, pos+cellLenght/2. is shorter than the distance from FE.
  // In fact it would make more sense to make plots vs y.
  return (entryP.y()+exitP.y())/2.+wireLenght;
}


// Find the RecHit closest to the muon SimHit
template  <typename type>
const type*
DTRecHitQuality::findBestRecHit(const DTLayer* layer,
                                DTWireId wireId,
                                const vector<type>& recHits,
                                const float simHitDist) {
  float res = 99999;
  const type* theBestRecHit = nullptr;
  // Loop over RecHits within the cell
  for (typename vector<type>::const_iterator recHit = recHits.begin();
      recHit != recHits.end();
      recHit++) {
    float distTmp = recHitDistFromWire(*recHit, layer);
    if (fabs(distTmp-simHitDist) < res) {
      res = fabs(distTmp-simHitDist);
      theBestRecHit = &(*recHit);
    }
  } // End of loop over RecHits within the cell

  return theBestRecHit;
}


// Compute the distance from wire (cm) of a hits in a DTRecHit1DPair
float
DTRecHitQuality::recHitDistFromWire(const DTRecHit1DPair& hitPair, const DTLayer* layer) {
  // Compute the rechit distance from wire
  return fabs(hitPair.localPosition(DTEnums::Left).x() -
              hitPair.localPosition(DTEnums::Right).x())/2.;
}



// Compute the distance from wire (cm) of a hits in a DTRecHit1D
float
DTRecHitQuality::recHitDistFromWire(const DTRecHit1D& recHit, const DTLayer* layer) {
  return fabs(recHit.localPosition().x() - layer->specificTopology().wirePosition(recHit.wireId().wire()));
}


template  <typename type>
void DTRecHitQuality::compute(const DTGeometry *dtGeom,
                              const std::map<DTWireId, std::vector<PSimHit> >& simHitsPerWire,
                              const std::map<DTWireId, std::vector<type> >& recHitsPerWire,
                              int step) {
  // Loop over cells with a muon SimHit
  for (map<DTWireId, vector<PSimHit> >::const_iterator wireAndSHits = simHitsPerWire.begin();
      wireAndSHits != simHitsPerWire.end();
      wireAndSHits++) {
    DTWireId wireId = (*wireAndSHits).first;
    int wheel = wireId.wheel();
    int sl = wireId.superLayer();

    vector<PSimHit> simHitsInCell = (*wireAndSHits).second;

    // Get the layer
    const DTLayer* layer = dtGeom->layer(wireId);

    // Look for a mu hit in the cell
    const PSimHit* muSimHit = DTHitQualityUtils::findMuSimHit(simHitsInCell);
    if (muSimHit == nullptr) {
      if (debug_)
        cout << "   No mu SimHit in channel: " << wireId << ", skipping! " << endl;
      continue; // Skip this cell
    }

    // Find the distance of the simhit from the wire
    float simHitWireDist = simHitDistFromWire(layer, wireId, *muSimHit);
    // Skip simhits out of the cell
    if (simHitWireDist>2.1) {
      if (debug_)
        cout << "  [DTRecHitQuality]###Warning: The mu SimHit in out of the cell, skipping!" << endl;
      continue; // Skip this cell
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
      if (debug_)
        cout << "   No RecHit found at Step: " << step << " in cell: " << wireId << endl;
    } else {
      recHitReconstructed = true;
      // vector<type> recHits = (*wireAndRecHits).second;
      const vector<type>& recHits = recHitsPerWire.at(wireId);
      if (debug_)
        cout << "   " << recHits.size() << " RecHits, Step " << step << " in channel: " << wireId << endl;

      // Find the best RecHit
      const type* theBestRecHit = findBestRecHit(layer, wireId, recHits, simHitWireDist);


      float recHitWireDist =  recHitDistFromWire(*theBestRecHit, layer);
      if (debug_)
        cout << "    SimHit distance from wire: " << simHitWireDist << endl
	     << "    SimHit distance from FE:   " << simHitFEDist << endl
	     << "    SimHit angle in layer RF:  " << simHitTheta << endl
	     << "    RecHit distance from wire: " << recHitWireDist << endl;
      float recHitErr = recHitPositionError(*theBestRecHit);
      HRes1DHit *hRes = nullptr;
      HRes1DHit *hResTot = nullptr;

      // Mirror angle in phi so that + and - wheels can be plotted together
      if (mirrorMinusWheels && wheel<0 && sl!= 2) {
	simHitTheta *= -1.;
	// Note: local X, if used, would have to be mirrored as well
      }

      // Fill residuals and pulls
      // Select the histo to be filled
      if (step == 1) {
        // Step 1
        if (sl != 2) {
          hResTot = hRes_S1RPhi_;
          if (wheel == 0)
            hRes = hRes_S1RPhi_W0_;
          if (abs(wheel) == 1)
            hRes = hRes_S1RPhi_W1_;
          if (abs(wheel) == 2)
            hRes = hRes_S1RPhi_W2_;
        } else {
          hResTot = hRes_S1RZ_;
          if (wheel == 0)
            hRes = hRes_S1RZ_W0_;
          if (abs(wheel) == 1)
            hRes = hRes_S1RZ_W1_;
          if (abs(wheel) == 2)
            hRes = hRes_S1RZ_W2_;
        }

      } else if (step == 2) {
        // Step 2
        if (sl != 2) {
          hRes = hRes_S2RPhi_;
          if (wheel == 0)
            hRes = hRes_S2RPhi_W0_;
          if (abs(wheel) == 1)
            hRes = hRes_S2RPhi_W1_;
          if (abs(wheel) == 2)
            hRes = hRes_S2RPhi_W2_;
        } else {
          hResTot = hRes_S2RZ_;
          if (wheel == 0)
            hRes = hRes_S2RZ_W0_;
          if (abs(wheel) == 1)
            hRes = hRes_S2RZ_W1_;
          if (abs(wheel) == 2)
            hRes = hRes_S2RZ_W2_;
        }

      } else if (step == 3) {
        // Step 3
        if (sl != 2) {
          hResTot = hRes_S3RPhi_;
          if (wheel == 0)
            hRes = hRes_S3RPhi_W0_;
          if (abs(wheel) == 1)
            hRes = hRes_S3RPhi_W1_;
          if (abs(wheel) == 2)
            hRes = hRes_S3RPhi_W2_;
	  if (local_) hRes_S3RPhiWS_[abs(wheel)][wireId.station()-1]->Fill(simHitWireDist, simHitTheta, simHitFEDist, recHitWireDist, simHitGlobalPos.eta(), simHitGlobalPos.phi(), recHitErr, wireId.station());

        } else {
          hResTot = hRes_S3RZ_;
          if (wheel == 0)
            hRes = hRes_S3RZ_W0_;
          if (abs(wheel) == 1)
            hRes = hRes_S3RZ_W1_;
          if (abs(wheel) == 2)
            hRes = hRes_S3RZ_W2_;

	  if (local_) hRes_S3RZWS_[abs(wheel)][wireId.station()-1]->Fill(simHitWireDist, simHitTheta, simHitFEDist, recHitWireDist, simHitGlobalPos.eta(), simHitGlobalPos.phi(), recHitErr, wireId.station());
        }
      }
      // Fill
      hRes->Fill(simHitWireDist, simHitTheta, simHitFEDist, recHitWireDist, simHitGlobalPos.eta(),
                 simHitGlobalPos.phi(), recHitErr, wireId.station());
      if (hResTot != nullptr)
        hResTot->Fill(simHitWireDist, simHitTheta, simHitFEDist, recHitWireDist, simHitGlobalPos.eta(),
                      simHitGlobalPos.phi(), recHitErr, wireId.station());
    }

    // Fill Efficiencies
    if (doall_) {
      HEff1DHit *hEff = nullptr;
      HEff1DHit *hEffTot = nullptr;
      if (step == 1) {
	// Step 1
	if (sl != 2) {
	  hEff = hEff_S1RPhi_;
	  if (local_) hEff_S1RPhiWS_[abs(wheel)][wireId.station()-1]->Fill(simHitWireDist, simHitGlobalPos.eta(), simHitGlobalPos.phi(), recHitReconstructed);
	} else {
	  hEffTot = hEff_S1RZ_;
	  if (wheel == 0)
	    hEff = hEff_S1RZ_W0_;
	  if (abs(wheel) == 1)
	    hEff = hEff_S1RZ_W1_;
	  if (abs(wheel) == 2)
	    hEff = hEff_S1RZ_W2_;
	  if (local_) hEff_S1RZWS_[abs(wheel)][wireId.station()-1]->Fill(simHitWireDist, simHitGlobalPos.eta(), simHitGlobalPos.phi(), recHitReconstructed);
	}

      } else if (step == 2) {
	// Step 2
	if (sl != 2) {
	  hEff = hEff_S2RPhi_;
	} else {
	  hEffTot = hEff_S2RZ_;
	  if (wheel == 0)
	    hEff = hEff_S2RZ_W0_;
	  if (abs(wheel) == 1)
	    hEff = hEff_S2RZ_W1_;
	  if (abs(wheel) == 2)
	    hEff = hEff_S2RZ_W2_;
	}

      } else if (step == 3) {
	// Step 3
	if (sl != 2) {
	  hEff = hEff_S3RPhi_;
	  if (local_) hEff_S3RPhiWS_[abs(wheel)][wireId.station()-1]->Fill(simHitWireDist, simHitGlobalPos.eta(), simHitGlobalPos.phi(), recHitReconstructed);
	} else {
	  hEffTot = hEff_S3RZ_;
	  if (wheel == 0)
	    hEff = hEff_S3RZ_W0_;
	  if (abs(wheel) == 1)
	    hEff = hEff_S3RZ_W1_;
	  if (abs(wheel) == 2)
	    hEff = hEff_S3RZ_W2_;
	  if (local_) hEff_S3RZWS_[abs(wheel)][wireId.station()-1]->Fill(simHitWireDist, simHitGlobalPos.eta(), simHitGlobalPos.phi(), recHitReconstructed);
	}

      }
      // Fill
      hEff->Fill(simHitWireDist, simHitGlobalPos.eta(), simHitGlobalPos.phi(), recHitReconstructed);
      if (hEffTot != nullptr)
	hEffTot->Fill(simHitWireDist, simHitGlobalPos.eta(), simHitGlobalPos.phi(), recHitReconstructed);
    }
  }
}

// Return the error on the measured (cm) coordinate
float DTRecHitQuality::recHitPositionError(const DTRecHit1DPair& recHit) {
  return sqrt(recHit.localPositionError(DTEnums::Left).xx());
}

// Return the error on the measured (cm) coordinate
float DTRecHitQuality::recHitPositionError(const DTRecHit1D& recHit) {
  return sqrt(recHit.localPositionError().xx());
}

