
/*
 *  See header file for a description of this class.
 *
 *  \author G. Cerminara - INFN Torino
 */

#include "DTRecHitQuality.h"
#include "Validation/DTRecHits/interface/DTHitQualityUtils.h"

#include "Geometry/DTGeometry/interface/DTLayer.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"


#include <iostream>
#include <map>

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
DTRecHitQuality::DTRecHitQuality(const ParameterSet& pset){
  // Get the debug parameter for verbose output
  debug = pset.getUntrackedParameter<bool>("debug");
  // the name of the simhit collection
  simHitLabel = pset.getUntrackedParameter<InputTag>("simHitLabel");
  // the name of the 1D rec hit collection
  recHitLabel = pset.getUntrackedParameter<InputTag>("recHitLabel");
  // the name of the 2D rec hit collection
  segment2DLabel = pset.getUntrackedParameter<InputTag>("segment2DLabel");
  // the name of the 4D rec hit collection
  segment4DLabel = pset.getUntrackedParameter<InputTag>("segment4DLabel");

  // Switches for analysis at various steps
  doStep1 = pset.getUntrackedParameter<bool>("doStep1", false);
  doStep2 = pset.getUntrackedParameter<bool>("doStep2", false);
  doStep3 = pset.getUntrackedParameter<bool>("doStep3", false);
  doall = pset.getUntrackedParameter<bool>("doall", false);
  local = pset.getUntrackedParameter<bool>("local", true);
}

void DTRecHitQuality::beginRun(const edm::Run& iRun, const edm::EventSetup &setup) {

  // ----------------------                 
  // get hold of back-end interface 
  dbe_ = 0;
  dbe_ = Service<DQMStore>().operator->();
  /*if ( dbe_ ) {
    if (debug) {
      dbe_->setVerbose(1);
    } else {
      dbe_->setVerbose(0);
    }
    }*/
  dbe_->setVerbose(0);
  /*if ( dbe_ ) {
    if ( debug ) dbe_->showDirStructure();
    }*/
  if(doall && doStep1){
    hRes_S1RPhi= new HRes1DHit("S1RPhi",dbe_,true,local);    // RecHits, 1. step, RPhi
    hRes_S1RPhi_W0= new HRes1DHit("S1RPhi_W0",dbe_,true,local);   // RecHits, 1. step, RZ, wheel 0
    hRes_S1RPhi_W1= new HRes1DHit("S1RPhi_W1",dbe_,true,local);   // RecHits, 1. step, RZ, wheel +-1
    hRes_S1RPhi_W2= new HRes1DHit("S1RPhi_W2",dbe_,true,local);   // RecHits, 1. step, RZ, wheel +-2
    hRes_S1RZ= new HRes1DHit("S1RZ",dbe_,true,local);         // RecHits, 1. step, RZ
    hRes_S1RZ_W0= new HRes1DHit("S1RZ_W0",dbe_,true,local);   // RecHits, 1. step, RZ, wheel 0
    hRes_S1RZ_W1= new HRes1DHit("S1RZ_W1",dbe_,true,local);   // RecHits, 1. step, RZ, wheel +-1
    hRes_S1RZ_W2= new HRes1DHit("S1RZ_W2",dbe_,true,local);   // RecHits, 1. step, RZ, wheel +-2
    hEff_S1RPhi= new HEff1DHit("S1RPhi",dbe_);     // RecHits, 1. step, RPhi
    hEff_S1RZ= new HEff1DHit("S1RZ",dbe_);         // RecHits, 1. step, RZ
    hEff_S1RZ_W0= new HEff1DHit("S1RZ_W0",dbe_);   // RecHits, 1. step, RZ, wheel 0
    hEff_S1RZ_W1= new HEff1DHit("S1RZ_W1",dbe_);   // RecHits, 1. step, RZ, wheel +-1
    hEff_S1RZ_W2= new HEff1DHit("S1RZ_W2",dbe_);   // RecHits, 1. step, RZ, wheel +-2
  }
  if(doall && doStep2){
    hRes_S2RPhi= new HRes1DHit("S2RPhi",dbe_,true,local);     // RecHits, 2. step, RPhi
    hRes_S2RPhi_W0= new HRes1DHit("S2RPhi_W0",dbe_,true,local);   // RecHits, 2. step, RPhi, wheel 0
    hRes_S2RPhi_W1= new HRes1DHit("S2RPhi_W1",dbe_,true,local);   // RecHits, 2. step, RPhi, wheel +-1
    hRes_S2RPhi_W2= new HRes1DHit("S2RPhi_W2",dbe_,true,local);   // RecHits, 2. step, RPhi, wheel +-2
    hRes_S2RZ= new HRes1DHit("S2RZ",dbe_,true,local);	    // RecHits, 2. step, RZ
    hRes_S2RZ_W0= new HRes1DHit("S2RZ_W0",dbe_,true,local);   // RecHits, 2. step, RZ, wheel 0
    hRes_S2RZ_W1= new HRes1DHit("S2RZ_W1",dbe_,true,local);   // RecHits, 2. step, RZ, wheel +-1
    hRes_S2RZ_W2= new HRes1DHit("S2RZ_W2",dbe_,true,local);   // RecHits, 2. step, RZ, wheel +-2
    hEff_S2RPhi= new HEff1DHit("S2RPhi",dbe_);     // RecHits, 2. step, RPhi
    hEff_S2RZ_W0= new HEff1DHit("S2RZ_W0",dbe_);   // RecHits, 2. step, RZ, wheel 0
    hEff_S2RZ_W1= new HEff1DHit("S2RZ_W1",dbe_);   // RecHits, 2. step, RZ, wheel +-1
    hEff_S2RZ_W2= new HEff1DHit("S2RZ_W2",dbe_);   // RecHits, 2. step, RZ, wheel +-2
    hEff_S2RZ= new HEff1DHit("S2RZ",dbe_);	    // RecHits, 2. step, RZ
  }
  if(doStep3){
    hRes_S3RPhi= new HRes1DHit("S3RPhi",dbe_,doall,local);     // RecHits, 3. step, RPhi
    hRes_S3RPhi_W0= new HRes1DHit("S3RPhi_W0",dbe_,doall,local);   // RecHits, 3. step, RPhi, wheel 0
    hRes_S3RPhi_W1= new HRes1DHit("S3RPhi_W1",dbe_,doall,local);   // RecHits, 3. step, RPhi, wheel +-1
    hRes_S3RPhi_W2= new HRes1DHit("S3RPhi_W2",dbe_,doall,local);   // RecHits, 3. step, RPhi, wheel +-2
    hRes_S3RZ= new HRes1DHit("S3RZ",dbe_,doall,local);	    // RecHits, 3. step, RZ
    hRes_S3RZ_W0= new HRes1DHit("S3RZ_W0",dbe_,doall,local);   // RecHits, 3. step, RZ, wheel 0
    hRes_S3RZ_W1= new HRes1DHit("S3RZ_W1",dbe_,doall,local);   // RecHits, 3. step, RZ, wheel +-1
    hRes_S3RZ_W2= new HRes1DHit("S3RZ_W2",dbe_,doall,local);   // RecHits, 3. step, RZ, wheel +-2

    if (local) {
      // Plots with finer granularity, not to be included in DQM
      TString name1="RPhi_W";
      TString name2="RZ_W";
      for (long w=0;w<=2;++w) {
	for (long s=1;s<=4;++s){
	  hRes_S3RPhiWS[w][s-1] = new HRes1DHit(("S3"+name1+w+"_St"+s).Data(),dbe_,doall,local); 
	  hEff_S1RPhiWS[w][s-1] = new HEff1DHit(("S1"+name1+w+"_St"+s).Data(),dbe_); 
	  hEff_S3RPhiWS[w][s-1] = new HEff1DHit(("S3"+name1+w+"_St"+s).Data(),dbe_); 
	  if (s!=4) {
	    hRes_S3RZWS[w][s-1] = new HRes1DHit(("S3"+name2+w+"_St"+s).Data(),dbe_,doall,local); 
	    hEff_S1RZWS[w][s-1] = new HEff1DHit(("S1"+name2+w+"_St"+s).Data(),dbe_); 
	    hEff_S3RZWS[w][s-1] = new HEff1DHit(("S3"+name2+w+"_St"+s).Data(),dbe_); 
	  }
	}
      }
    }
    
    
    if(doall){
      hEff_S3RPhi= new HEff1DHit("S3RPhi",dbe_);     // RecHits, 3. step, RPhi
      hEff_S3RZ= new HEff1DHit("S3RZ",dbe_);	    // RecHits, 3. step, RZ
      hEff_S3RZ_W0= new HEff1DHit("S3RZ_W0",dbe_);   // RecHits, 3. step, RZ, wheel 0
      hEff_S3RZ_W1= new HEff1DHit("S3RZ_W1",dbe_);   // RecHits, 3. step, RZ, wheel +-1
      hEff_S3RZ_W2= new HEff1DHit("S3RZ_W2",dbe_);   // RecHits, 3. step, RZ, wheel +-2
    }
  }
}


// Destructor
DTRecHitQuality::~DTRecHitQuality(){
}


void DTRecHitQuality::endLuminosityBlock(edm::LuminosityBlock const& lumiSeg,
    edm::EventSetup const& c){

}

void DTRecHitQuality::endJob() {
  // Write the histos to file
  if(doall){
    if(doStep1){
      hEff_S1RPhi->ComputeEfficiency();
      hEff_S1RZ->ComputeEfficiency();
      hEff_S1RZ_W0->ComputeEfficiency();
      hEff_S1RZ_W1->ComputeEfficiency();
      hEff_S1RZ_W2->ComputeEfficiency();
    }
    if(doStep2){
      hEff_S2RPhi->ComputeEfficiency();
      hEff_S2RZ->ComputeEfficiency();
      hEff_S2RZ_W0->ComputeEfficiency();
      hEff_S2RZ_W1->ComputeEfficiency();
      hEff_S2RZ_W2->ComputeEfficiency();
    }
    if(doStep3){
      hEff_S3RPhi->ComputeEfficiency();
      hEff_S3RZ->ComputeEfficiency();
      hEff_S3RZ_W0->ComputeEfficiency();
      hEff_S3RZ_W1->ComputeEfficiency();
      hEff_S3RZ_W2->ComputeEfficiency();
    }
  }
}

// The real analysis
  void DTRecHitQuality::analyze(const Event & event, const EventSetup& eventSetup){
    if(debug)
      cout << "--- [DTRecHitQuality] Analysing Event: #Run: " << event.id().run()
        << " #Event: " << event.id().event() << endl;
    //theFile->cd();
    // Get the DT Geometry
    ESHandle<DTGeometry> dtGeom;
    eventSetup.get<MuonGeometryRecord>().get(dtGeom);

    // Get the SimHit collection from the event
    Handle<PSimHitContainer> simHits;
    event.getByLabel(simHitLabel, simHits);

    // Map simhits per wire
    map<DTWireId, PSimHitContainer > simHitsPerWire =
      DTHitQualityUtils::mapSimHitsPerWire(*(simHits.product()));



     //=======================================================================================
    // RecHit analysis at Step 1
    if(doStep1 && doall) {
      if(debug)
        cout << "  -- DTRecHit S1: begin analysis:" << endl;
      // Get the rechit collection from the event
      Handle<DTRecHitCollection> dtRecHits;
      event.getByLabel(recHitLabel, dtRecHits);

      if(!dtRecHits.isValid()) {
	if(debug) cout << "[DTRecHitQuality]**Warning: no 1DRechits with label: " << recHitLabel << " in this event, skipping!" << endl;
	return;
      }
     
     // Map rechits per wire
      map<DTWireId,vector<DTRecHit1DPair> > recHitsPerWire = 
        map1DRecHitsPerWire(dtRecHits.product());

      compute(dtGeom.product(), simHitsPerWire, recHitsPerWire, 1);
    }


    //=======================================================================================
    // RecHit analysis at Step 2
    if(doStep2 && doall) {
      if(debug)
        cout << "  -- DTRecHit S2: begin analysis:" << endl;

      // Get the 2D rechits from the event
      Handle<DTRecSegment2DCollection> segment2Ds;
      event.getByLabel(segment2DLabel, segment2Ds);

      if(!segment2Ds.isValid()) {
       if(debug) cout << "[DTRecHitQuality]**Warning: no 2DSegments with label: " << segment2DLabel
		      << " in this event, skipping!" << endl;
       
      }
      else{
	// Map rechits per wire
	map<DTWireId,vector<DTRecHit1D> > recHitsPerWire = 
	  map1DRecHitsPerWire(segment2Ds.product());
	
	compute(dtGeom.product(), simHitsPerWire, recHitsPerWire, 2);
      }
    }

    //=======================================================================================
    // RecHit analysis at Step 3
    if(doStep3) {
      if(debug)
        cout << "  -- DTRecHit S3: begin analysis:" << endl;

      // Get the 4D rechits from the event
      Handle<DTRecSegment4DCollection> segment4Ds;
      event.getByLabel(segment4DLabel, segment4Ds);

      if(!segment4Ds.isValid()) {
        if(debug) cout << "[DTRecHitQuality]**Warning: no 4D Segments with label: " << segment4DLabel
		       << " in this event, skipping!" << endl;
	return;
      }

      // Map rechits per wire
      map<DTWireId,vector<DTRecHit1D> > recHitsPerWire = 
        map1DRecHitsPerWire(segment4Ds.product());

      compute(dtGeom.product(), simHitsPerWire, recHitsPerWire, 3);
    }

  }



// Return a map between DTRecHit1DPair and wireId
map<DTWireId, vector<DTRecHit1DPair> >
DTRecHitQuality::map1DRecHitsPerWire(const DTRecHitCollection* dt1DRecHitPairs) {
  map<DTWireId, vector<DTRecHit1DPair> > ret;

  for(DTRecHitCollection::const_iterator rechit = dt1DRecHitPairs->begin();
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
  for(DTRecSegment2DCollection::const_iterator segment = segment2Ds->begin();
      segment != segment2Ds->end();
      segment++) {
    vector<DTRecHit1D> component1DHits= (*segment).specificRecHits();
    // Loop over all component 1D hits
    for(vector<DTRecHit1D>::const_iterator hit = component1DHits.begin();
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
  for(DTRecSegment4DCollection::const_iterator segment = segment4Ds->begin();
      segment != segment4Ds->end();
      segment++) {
    // Get component 2D segments
    vector<const TrackingRecHit*> segment2Ds = (*segment).recHits();
    // Loop over 2D segments:
    for(vector<const TrackingRecHit*>::const_iterator segment2D = segment2Ds.begin();
        segment2D != segment2Ds.end();
        segment2D++) {
      // Get 1D component rechits
      vector<const TrackingRecHit*> hits = (*segment2D)->recHits();
      // Loop over them
      for(vector<const TrackingRecHit*>::const_iterator hit = hits.begin();
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

  return fabs(xEntry - (entryP.z()*(xExit-xEntry))/(exitP.z()-entryP.z()));//FIXME: check...
}

// Compute SimHit impact angle (in direction perp to wire), in the SL RF 
float DTRecHitQuality::simHitImpactAngle(const DTLayer* layer,
                                         DTWireId wireId,
                                         const PSimHit& hit) {
  LocalPoint entryP = hit.entryPoint();
  LocalPoint exitP = hit.exitPoint();
  float theta=(exitP.x()-entryP.x())/(exitP.z()-entryP.z());
  return atan(theta);
}

// Compute SimHit distance from FrontEnd
float DTRecHitQuality::simHitDistFromFE(const DTLayer* layer,
                                        DTWireId wireId,
                                        const PSimHit& hit) {
  LocalPoint entryP = hit.entryPoint();
  LocalPoint exitP = hit.exitPoint();
  float wireLenght=layer->specificTopology().cellLenght();
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
  const type* theBestRecHit = 0;
  // Loop over RecHits within the cell
  for(typename vector<type>::const_iterator recHit = recHits.begin();
      recHit != recHits.end();
      recHit++) {
    float distTmp = recHitDistFromWire(*recHit, layer);
    if(fabs(distTmp-simHitDist) < res) {
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
  for(map<DTWireId, vector<PSimHit> >::const_iterator wireAndSHits = simHitsPerWire.begin();
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
    if (muSimHit==0) {
      if (debug) 
        cout << "   No mu SimHit in channel: " << wireId << ", skipping! " << endl;
      continue; // Skip this cell
    }

    // Find the distance of the simhit from the wire
    float simHitWireDist = simHitDistFromWire(layer, wireId, *muSimHit);
    // Skip simhits out of the cell
    if(simHitWireDist>2.1) {
      if(debug) 
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
    if(recHitsPerWire.find(wireId) == recHitsPerWire.end()) {
      // No RecHit found in this cell
      if(debug)
        cout << "   No RecHit found at Step: " << step << " in cell: " << wireId << endl;
    } else {
      recHitReconstructed = true;
      // vector<type> recHits = (*wireAndRecHits).second;
      vector<type> recHits = recHitsPerWire.at(wireId);
      if(debug)
        cout << "   " << recHits.size() << " RecHits, Step " << step << " in channel: " << wireId << endl;

      // Find the best RecHit
      const type* theBestRecHit = findBestRecHit(layer, wireId, recHits, simHitWireDist);


      float recHitWireDist =  recHitDistFromWire(*theBestRecHit, layer);
      if(debug)
        cout << "    SimHit distance from wire: " << simHitWireDist << endl
	     << "    SimHit distance from FE:   " << simHitFEDist << endl
	     << "    SimHit angle in layer RF:  " << simHitTheta << endl
	     << "    RecHit distance from wire: " << recHitWireDist << endl;
      float recHitErr = recHitPositionError(*theBestRecHit);
      HRes1DHit *hRes = 0;
      HRes1DHit *hResTot = 0;

      // Mirror angle in phi so that + and - wheels can be plotted together
      if (mirrorMinusWheels && wheel<0 && sl!=2){
	simHitTheta *= -1.;
	// Note: local X, if used, would have to be mirrored as well
      }

      // Fill residuals and pulls
      // Select the histo to be filled
      if(step == 1) {
        // Step 1
        if(sl != 2) {
          hResTot = hRes_S1RPhi;
          if(wheel == 0)
            hRes = hRes_S1RPhi_W0;
          if(abs(wheel) == 1)
            hRes = hRes_S1RPhi_W1;
          if(abs(wheel) == 2)
            hRes = hRes_S1RPhi_W2;
        } else {
          hResTot = hRes_S1RZ;
          if(wheel == 0)
            hRes = hRes_S1RZ_W0;
          if(abs(wheel) == 1)
            hRes = hRes_S1RZ_W1;
          if(abs(wheel) == 2)
            hRes = hRes_S1RZ_W2;
        }

      } else if(step == 2) {
        // Step 2
        if(sl != 2) {
          hRes = hRes_S2RPhi;
          if(wheel == 0)
            hRes = hRes_S2RPhi_W0;
          if(abs(wheel) == 1)
            hRes = hRes_S2RPhi_W1;
          if(abs(wheel) == 2)
            hRes = hRes_S2RPhi_W2;
        } else {
          hResTot = hRes_S2RZ;
          if(wheel == 0)
            hRes = hRes_S2RZ_W0;
          if(abs(wheel) == 1)
            hRes = hRes_S2RZ_W1;
          if(abs(wheel) == 2)
            hRes = hRes_S2RZ_W2;
        }

      } else if(step == 3) {
        // Step 3
        if(sl != 2) {
          hResTot = hRes_S3RPhi;
          if(wheel == 0)
            hRes = hRes_S3RPhi_W0;
          if(abs(wheel) == 1)
            hRes = hRes_S3RPhi_W1;
          if(abs(wheel) == 2)
            hRes = hRes_S3RPhi_W2;
	  if (local) hRes_S3RPhiWS[abs(wheel)][wireId.station()-1]->Fill(simHitWireDist, simHitTheta, simHitFEDist, recHitWireDist, simHitGlobalPos.eta(),simHitGlobalPos.phi(),recHitErr,wireId.station());
	  
        } else {
          hResTot = hRes_S3RZ;
          if(wheel == 0)
            hRes = hRes_S3RZ_W0;
          if(abs(wheel) == 1)
            hRes = hRes_S3RZ_W1;
          if(abs(wheel) == 2)
            hRes = hRes_S3RZ_W2;

	  if (local) hRes_S3RZWS[abs(wheel)][wireId.station()-1]->Fill(simHitWireDist, simHitTheta, simHitFEDist, recHitWireDist, simHitGlobalPos.eta(),simHitGlobalPos.phi(),recHitErr,wireId.station());
        }
      }
      // Fill
      hRes->Fill(simHitWireDist, simHitTheta, simHitFEDist, recHitWireDist, simHitGlobalPos.eta(),
                 simHitGlobalPos.phi(),recHitErr,wireId.station());
      if(hResTot != 0)
        hResTot->Fill(simHitWireDist, simHitTheta, simHitFEDist, recHitWireDist, simHitGlobalPos.eta(),
                      simHitGlobalPos.phi(),recHitErr,wireId.station());
    }

    // Fill Efficiencies
    if(doall){
      HEff1DHit *hEff = 0;
      HEff1DHit *hEffTot = 0;
      if(step == 1) {
	// Step 1
	if(sl != 2) {
	  hEff = hEff_S1RPhi;
	  if (local) hEff_S1RPhiWS[abs(wheel)][wireId.station()-1]->Fill(simHitWireDist, simHitGlobalPos.eta(), simHitGlobalPos.phi(), recHitReconstructed);
	} else {
	  hEffTot = hEff_S1RZ;
	  if(wheel == 0)
	    hEff = hEff_S1RZ_W0;
	  if(abs(wheel) == 1)
	    hEff = hEff_S1RZ_W1;
	  if(abs(wheel) == 2)
	    hEff = hEff_S1RZ_W2;
	  if (local) hEff_S1RZWS[abs(wheel)][wireId.station()-1]->Fill(simHitWireDist, simHitGlobalPos.eta(), simHitGlobalPos.phi(), recHitReconstructed);
	}
	
      } else if(step == 2) {
	// Step 2
	if(sl != 2) {
	  hEff = hEff_S2RPhi;
	} else {
	  hEffTot = hEff_S2RZ;
	  if(wheel == 0)
	    hEff = hEff_S2RZ_W0;
	  if(abs(wheel) == 1)
	    hEff = hEff_S2RZ_W1;
	  if(abs(wheel) == 2)
	    hEff = hEff_S2RZ_W2;
	}
	
      } else if(step == 3) {
	// Step 3
	if(sl != 2) {
	  hEff = hEff_S3RPhi;
	  if (local) hEff_S3RPhiWS[abs(wheel)][wireId.station()-1]->Fill(simHitWireDist, simHitGlobalPos.eta(), simHitGlobalPos.phi(), recHitReconstructed);
	} else {
	  hEffTot = hEff_S3RZ;
	  if(wheel == 0)
	    hEff = hEff_S3RZ_W0;
	  if(abs(wheel) == 1)
	    hEff = hEff_S3RZ_W1;
	  if(abs(wheel) == 2)
	    hEff = hEff_S3RZ_W2;
	  if (local) hEff_S3RZWS[abs(wheel)][wireId.station()-1]->Fill(simHitWireDist, simHitGlobalPos.eta(), simHitGlobalPos.phi(), recHitReconstructed);
	}

      }
      // Fill
      hEff->Fill(simHitWireDist, simHitGlobalPos.eta(), simHitGlobalPos.phi(), recHitReconstructed);
      if(hEffTot != 0)
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

