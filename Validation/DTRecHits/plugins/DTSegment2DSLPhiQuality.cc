/*
 *  See header file for a description of this class.
 *
 *  \author S. Bolognesi and G. Cerminara - INFN Torino
 */

#include "DTSegment2DSLPhiQuality.h"
#include "Validation/DTRecHits/interface/DTHitQualityUtils.h"

#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "Geometry/DTGeometry/interface/DTChamber.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "DataFormats/DTRecHit/interface/DTRecHitCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "Histograms.h"
#include "TStyle.h"
#include "TFile.h"

#include <iostream>
#include <map>
//#include "utils.C"

using namespace std;
using namespace edm;
//TStyle * mystyle;


// Constructor
DTSegment2DSLPhiQuality::DTSegment2DSLPhiQuality(const ParameterSet& pset)  {
   // Get the debug parameter for verbose output
  debug = pset.getUntrackedParameter<bool>("debug");
  DTHitQualityUtils::debug = debug;

  // the name of the simhit collection
  simHitLabel = pset.getUntrackedParameter<InputTag>("simHitLabel");
  // the name of the 4D rec hit collection
  segment4DLabel = pset.getUntrackedParameter<InputTag>("segment4DLabel");

  //sigma resolution on position
  sigmaResPos = pset.getParameter<double>("sigmaResPos");
  //sigma resolution on angle
  sigmaResAngle = pset.getParameter<double>("sigmaResAngle");
  doall = pset.getUntrackedParameter<bool>("doall", false);
  local = pset.getUntrackedParameter<bool>("local", false);
}


void DTSegment2DSLPhiQuality::beginRun(const edm::Run& iRun, const edm::EventSetup &setup) {

  // get hold of back-end interface 
  dbe_ = 0;
  dbe_ = Service<DQMStore>().operator->();
  if ( dbe_ ) {
    if (debug) {
      dbe_->setVerbose(1);
    } else {
      dbe_->setVerbose(0);
    }
  }
  if ( dbe_ ) {
    if ( debug ) dbe_->showDirStructure();
  }

  // Book the histos
  h2DHitSuperPhi = new HRes2DHit ("SuperPhi",dbe_,doall,local);
  if(doall) h2DHitEff_SuperPhi = new HEff2DHit ("SuperPhi",dbe_);
}

// Destructor
DTSegment2DSLPhiQuality::~DTSegment2DSLPhiQuality(){
}

void DTSegment2DSLPhiQuality::endLuminosityBlock(edm::LuminosityBlock const& lumiSeg,
    edm::EventSetup const& c){
}




void DTSegment2DSLPhiQuality::endJob() {
  // Write the histos to file
  //theFile->cd();
  //h2DHitSuperPhi->Write();

  if(doall) h2DHitEff_SuperPhi->ComputeEfficiency();
  //h2DHitEff_SuperPhi->Write();

  //if ( rootFileName.size() != 0 && dbe_ ) dbe_->save(rootFileName); 
  //theFile->Close();
} 

// The real analysis
void DTSegment2DSLPhiQuality::analyze(const Event & event, const EventSetup& eventSetup){
  //theFile->cd();

  // Get the DT Geometry
  ESHandle<DTGeometry> dtGeom;
  eventSetup.get<MuonGeometryRecord>().get(dtGeom);

  // Get the SimHit collection from the event
  edm::Handle<PSimHitContainer> simHits;
  event.getByLabel(simHitLabel, simHits); //FIXME: second string to be removed

  //Map simHits by chamber
  map<DTChamberId, PSimHitContainer > simHitsPerCh;
  for(PSimHitContainer::const_iterator simHit = simHits->begin();
      simHit != simHits->end(); simHit++){
    // Create the id of the chamber (the simHits in the DT known their wireId)
    DTChamberId chamberId = (((DTWireId(simHit->detUnitId())).layerId()).superlayerId()).chamberId();
    // Fill the map
    simHitsPerCh[chamberId].push_back(*simHit);
  }

  // Get the 4D rechits from the event
  Handle<DTRecSegment4DCollection> segment4Ds;
  event.getByLabel(segment4DLabel, segment4Ds);

  if(!segment4Ds.isValid()) {
    if(debug) cout << "[DTSegment2DSLPhiQuality]**Warning: no 4D Segments with label: " << segment4DLabel
   << " in this event, skipping!" << endl;
    return;
  }
    
  // Loop over all chambers containing a segment
  DTRecSegment4DCollection::id_iterator chamberId;
  for (chamberId = segment4Ds->id_begin();
       chamberId != segment4Ds->id_end();
       ++chamberId){
    
    //------------------------- simHits ---------------------------//
    //Get simHits of each chamber
    PSimHitContainer simHits =  simHitsPerCh[(*chamberId)];
       
    // Map simhits per wire
    map<DTWireId, PSimHitContainer > simHitsPerWire = DTHitQualityUtils::mapSimHitsPerWire(simHits);
    map<DTWireId, const PSimHit*> muSimHitPerWire = DTHitQualityUtils::mapMuSimHitsPerWire(simHitsPerWire);
    int nMuSimHit = muSimHitPerWire.size();
    if(nMuSimHit == 0 || nMuSimHit == 1) {
      if(debug && nMuSimHit == 1)
        cout << "[DTSegment2DSLPhiQuality] Only " << nMuSimHit << " mu SimHit in this chamber, skipping!" << endl;
      continue; // If no or only one mu SimHit is found skip this chamber
    } 
    if(debug)
      cout << "=== Chamber " << (*chamberId) << " has " << nMuSimHit << " SimHits" << endl;
  
    //Find outer and inner mu SimHit to build a segment
    pair<const PSimHit*, const PSimHit*> inAndOutSimHit = DTHitQualityUtils::findMuSimSegment(muSimHitPerWire); 

    //Find direction and position of the sim Segment in Chamber RF
    pair<LocalVector, LocalPoint> dirAndPosSimSegm = DTHitQualityUtils::findMuSimSegmentDirAndPos(inAndOutSimHit,
												  (*chamberId),&(*dtGeom));

    LocalVector simSegmLocalDir = dirAndPosSimSegm.first;
    LocalPoint simSegmLocalPos = dirAndPosSimSegm.second;
    const DTChamber* chamber = dtGeom->chamber(*chamberId);
    GlobalPoint simSegmGlobalPos = chamber->toGlobal(simSegmLocalPos);

    //Atan(x/z) angle and x position in SL RF
    float angleSimSeg = DTHitQualityUtils::findSegmentAlphaAndBeta(simSegmLocalDir).first;
    float posSimSeg = simSegmLocalPos.x(); 
    //Position (in eta,phi coordinates) in lobal RF
    float etaSimSeg = simSegmGlobalPos.eta(); 
    float phiSimSeg = simSegmGlobalPos.phi();

    if(debug)
      cout<<"  Simulated segment:  local direction "<<simSegmLocalDir<<endl
	  <<"                      local position  "<<simSegmLocalPos<<endl
	  <<"                      angle           "<<angleSimSeg<<endl;
    
    //---------------------------- recHits --------------------------//
    // Get the range of rechit for the corresponding chamberId
    bool recHitFound = false;
    DTRecSegment4DCollection::range range = segment4Ds->get(*chamberId);
    int nsegm = distance(range.first, range.second);
    if(debug)
      cout << "   Chamber: " << *chamberId << " has " << nsegm
	   << " 4D segments" << endl;

    if (nsegm!=0) {
      // Find the best RecHit: look for the 4D RecHit with the phi angle closest
      // to that of segment made of SimHits. 
      // RecHits must have delta alpha and delta position within 5 sigma of
      // the residual distribution (we are looking for residuals of segments
      // usefull to the track fit) for efficency purpose
      const DTRecSegment2D* bestRecHit = 0;
      bool bestRecHitFound = false;
      double deltaAlpha = 99999;

      // Loop over the recHits of this chamberId
      for (DTRecSegment4DCollection::const_iterator segment4D = range.first;
           segment4D!=range.second;
           ++segment4D){
        // Check the dimension
        if((*segment4D).dimension() != 4) {
          if(debug) cout << "[DTSegment2DSLPhiQuality]***Error: This is not 4D segment!!!" << endl;
          continue;
        }

        //Get 2D superPhi segments from 4D segments
        const DTChamberRecSegment2D* phiSegment2D = (*segment4D).phiSegment();
        if((*phiSegment2D).dimension() != 2) {
          if(debug) cout << "[DTSegment2DQuality]***Error: This is not 2D segment!!!" << endl;
          abort();
        }

        // Segment Local Direction and position (in Chamber RF)
        LocalVector recSegDirection = (*phiSegment2D).localDirection();

        float recSegAlpha = DTHitQualityUtils::findSegmentAlphaAndBeta(recSegDirection).first;
        if(debug)
          cout << "  RecSegment direction: " << recSegDirection << endl
            << "             position : " <<  (*phiSegment2D).localPosition() << endl
            << "             alpha    : " << recSegAlpha << endl;

        if(fabs(recSegAlpha - angleSimSeg) < deltaAlpha) {
          deltaAlpha = fabs(recSegAlpha - angleSimSeg);
          bestRecHit = &(*phiSegment2D);
          bestRecHitFound = true;
        }
      }  // End of Loop over all 4D RecHits of this chambers

      if(bestRecHitFound) {
        // Best rechit direction and position in Chamber RF
        LocalPoint bestRecHitLocalPos = bestRecHit->localPosition();
        LocalVector bestRecHitLocalDir = bestRecHit->localDirection();

        LocalError bestRecHitLocalPosErr = bestRecHit->localPositionError();
        LocalError bestRecHitLocalDirErr = bestRecHit->localDirectionError();

        float angleBestRHit = DTHitQualityUtils::findSegmentAlphaAndBeta(bestRecHitLocalDir).first;
        if(fabs(angleBestRHit - angleSimSeg) < 5*sigmaResAngle &&
           fabs(bestRecHitLocalPos.x() - posSimSeg) < 5*sigmaResPos) {
          recHitFound = true;
        }

        // Fill Residual histos
        h2DHitSuperPhi->Fill(angleSimSeg,
                            angleBestRHit,
                            posSimSeg,
                            bestRecHitLocalPos.x(),
                            etaSimSeg,
                            phiSimSeg,
                            sqrt(bestRecHitLocalPosErr.xx()),
                            sqrt(bestRecHitLocalDirErr.xx())
                           );
      }
    } //end of if(nsegm!=0)

      // Fill Efficiency plot
    if(doall) {h2DHitEff_SuperPhi->Fill(etaSimSeg,
                            phiSimSeg,
                            posSimSeg,
                            angleSimSeg,
					recHitFound);}
  } // End of loop over chambers
}


// Fit a histogram in the range (minfit, maxfit) with a gaussian and
// draw it in the range (min, max)
