/*
 *  See header file for a description of this class.
 *
 *  $Date: 2006/08/04 10:35:03 $
 *  $Revision: 1.4 $
 *  \author D. Pagano - Dip. Fis. Nucl. e Teo. & INFN Pavia
 */

#include "RPCRecHitQuality.h"



//#include "DTHitQualityUtils.h"

#include "Geometry/RPCGeometry/interface/RPCRoll.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"



#include "Histograms.h"



#include "TFile.h"

#include <iostream>
#include <map>



using namespace std;
using namespace edm;
/*

HRes1DHit hRes_S1RPhi("S1RPhi");     // RecHits, 1. step, RPhi
HRes1DHit hRes_S1RZ("S1RZ");         // RecHits, 1. step, RZ
HRes1DHit hRes_S1RZ_W0("S1RZ_W0");   // RecHits, 1. step, RZ, wheel 0
HRes1DHit hRes_S1RZ_W1("S1RZ_W1");   // RecHits, 1. step, RZ, wheel +-1
HRes1DHit hRes_S1RZ_W2("S1RZ_W2");   // RecHits, 1. step, RZ, wheel +-2
HEff1DHit hEff_S1RPhi("S1RPhi");     // RecHits, 1. step, RPhi
HEff1DHit hEff_S1RZ("S1RZ");         // RecHits, 1. step, RZ
HEff1DHit hEff_S1RZ_W0("S1RZ_W0");   // RecHits, 1. step, RZ, wheel 0
HEff1DHit hEff_S1RZ_W1("S1RZ_W1");   // RecHits, 1. step, RZ, wheel +-1
HEff1DHit hEff_S1RZ_W2("S1RZ_W2");   // RecHits, 1. step, RZ, wheel +-2




  // Constructor
  RPCRecHitQuality::RPCRecHitQuality(const ParameterSet& pset){


  
  // Get the debug parameter for verbose output
  debug = pset.getUntrackedParameter<bool>("debug");
  rootFileName = pset.getUntrackedParameter<string>("rootFileName");
  


  // the name of the simhit collection
  simHitLabel = pset.getUntrackedParameter<string>("simHitLabel", "SimG4Object");
  


  // the name of the 1D rec hit collection
  recHitLabel = pset.getUntrackedParameter<string>("recHitLabel", "RPCRecHit1DProducer");
  


  if(debug)
    cout << "[RPCRecHitQuality] Constructor called" << endl;


  // Create the root file
  theFile = new TFile(rootFileName.c_str(), "RECREATE");
  theFile->cd();

}



// Destructor
RPCRecHitQuality::~RPCRecHitQuality(){
  if(debug) 
    cout << "[RPCRecHitQuality] Destructor called" << endl;
}




void RPCRecHitQuality::endJob() {
  // Write the histos to file
  theFile->cd();

  hEff_S1RPhi.ComputeEfficiency();
  hEff_S1RZ.ComputeEfficiency();
  hEff_S1RZ_W0.ComputeEfficiency();
  hEff_S1RZ_W1.ComputeEfficiency();
  hEff_S1RZ_W2.ComputeEfficiency();
 

  // Write histos to file
  hRes_S1RPhi.Write();
  hRes_S1RZ.Write();
  hRes_S1RZ_W0.Write();
  hRes_S1RZ_W1.Write();
  hRes_S1RZ_W2.Write();
  hEff_S1RPhi.Write();
  hEff_S1RZ.Write();
  hEff_S1RZ_W0.Write();
  hEff_S1RZ_W1.Write();
  hEff_S1RZ_W2.Write();
  theFile->Close();
}




// The real analysis
void RPCRecHitQuality::analyze(const Event & event, const EventSetup& eventSetup){
  if(debug)
    cout << "--- [RPCRecHitQuality] Analysing Event: #Run: " << event.id().run()
	 << " #Event: " << event.id().event() << endl;
  theFile->cd();
  // Get the RPC Geometry
  ESHandle<RPCGeometry> RPCGeom;
  eventSetup.get<MuonGeometryRecord>().get(RPCGeom);


  // Get the SimHit collection from the event
  Handle<PSimHitContainer> simHits;
  event.getByLabel(simHitLabel, "MuonRPCHits", simHits);

  // Map simhits per wire
  map<DTWireId, PSimHitContainer > simHitsPerWire =
  DTHitQualityUtils::mapSimHitsPerWire(*(simHits.product()));

 


  //=======================================================================================
  // RecHit analysis at Step 1
  if(doStep1) {
    if(debug)
      cout << "  -- DTRecHit S1: begin analysis:" << endl;
    // Get the rechit collection from the event
    Handle<DTRecHitCollection> dtRecHits;
    event.getByLabel(recHitLabel, dtRecHits);
    
    // Map rechits per wire
    map<DTWireId,vector<DTRecHit1DPair> > recHitsPerWire = 
      map1DRecHitsPerWire(dtRecHits.product());

    compute(dtGeom.product(), simHitsPerWire, recHitsPerWire, 1);
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
			      std::map<DTWireId, std::vector<PSimHit> > simHitsPerWire,
			      std::map<DTWireId, std::vector<type> > recHitsPerWire,
			      int step) {
  // Loop over cells with a muon SimHit
  for(map<DTWireId, vector<PSimHit> >::const_iterator wireAndSHits = simHitsPerWire.begin();
      wireAndSHits != simHitsPerWire.end();
      wireAndSHits++) {
    DTWireId wireId = (*wireAndSHits).first;
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
      cout << "  [DTRecHitQuality]###Warning: The mu SimHit in out of the cell, skipping!" << endl;
      continue; // Skip this cell
    }
    GlobalPoint simHitGlobalPos = layer->toGlobal(muSimHit->localPosition());
      
    bool recHitReconstructed = false;

    // Look for RecHits in the same cell
    if(recHitsPerWire.find(wireId) == recHitsPerWire.end()) {
      // No RecHit found in this cell
      if(debug)
	cout << "   No RecHit found at Step: " << step << " in cell: " << wireId << endl;
    } else {
      recHitReconstructed = true;
      // vector<type> recHits = (*wireAndRecHits).second;
      vector<type> recHits = recHitsPerWire[wireId];
      if(debug)
	cout << "   " << recHits.size() << " RecHits, Step " << step << " in channel: " << wireId << endl;
	 
      // Find the best RecHit
      const type* theBestRecHit = findBestRecHit(layer, wireId, recHits, simHitWireDist);

	 
      float recHitWireDist =  recHitDistFromWire(*theBestRecHit, layer);
      if(debug)
	cout << "    SimHit distance from wire: " << simHitWireDist << endl
	     << "    RecHit distance from wire: " << recHitWireDist << endl;
      float recHitErr = recHitPositionError(*theBestRecHit);

      HRes1DHit *hRes = 0;
      HRes1DHit *hResTot = 0;

      // Fill residuals and pulls
      // Select the histo to be filled
      
	if(wireId.superLayer() != 2) {
	  hRes = &hRes_S1RPhi;
	} else {
	  hResTot = &hRes_S1RZ;
	  if(wireId.wheel() == 0)
	    hRes = &hRes_S1RZ_W0;
	  if(abs(wireId.wheel()) == 1)
	    hRes = &hRes_S1RZ_W1;
	  if(abs(wireId.wheel()) == 2)
	    hRes = &hRes_S1RZ_W2;
	}

       

      // Fill
      hRes->Fill(simHitWireDist, recHitWireDist, simHitGlobalPos.eta(),
		 simHitGlobalPos.phi(),recHitErr);
      if(hResTot != 0)
	hResTot->Fill(simHitWireDist, recHitWireDist, simHitGlobalPos.eta(),
		      simHitGlobalPos.phi(),recHitErr);
      
    }

    // Fill Efficiencies
    HEff1DHit *hEff = 0;
    HEff1DHit *hEffTot = 0;
    
    if(wireId.superlayer() != 2) {
	hEff = &hEff_S1RPhi;
      } else {
	hEffTot = &hEff_S1RZ;
	if(wireId.wheel() == 0)
	  hEff = &hEff_S1RZ_W0;
	if(abs(wireId.wheel()) == 1)
	  hEff = &hEff_S1RZ_W1;
	if(abs(wireId.wheel()) == 2)
	  hEff = &hEff_S1RZ_W2;
      }

     
 
    // Fill
    hEff->Fill(simHitWireDist, simHitGlobalPos.eta(), simHitGlobalPos.phi(), recHitReconstructed);
    if(hEffTot != 0)
      hEffTot->Fill(simHitWireDist, simHitGlobalPos.eta(), simHitGlobalPos.phi(), recHitReconstructed);
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


*/
