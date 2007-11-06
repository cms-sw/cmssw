// -*- C++ -*-
//
// Package:    BetaFromTOF
// Class:      BetaFromTOF
// 
/**\class BetaFromTOF BetaFromTOF.cc SUSYBSMAnalysis/BetaFromTOF/src/BetaFromTOF.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Traczyk Piotr
//         Created:  Thu Oct 11 15:01:28 CEST 2007
// $Id: BetaFromTOF.cc,v 1.3 2007/10/25 12:33:31 ptraczyk Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"

#include "DataFormats/Common/interface/Ref.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"
#include "Geometry/DTGeometry/interface/DTChamber.h"
#include "Geometry/DTGeometry/interface/DTSuperLayer.h"
#include "DataFormats/DTRecHit/interface/DTSLRecSegment2D.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment2D.h"
#include "RecoLocalMuon/DTSegment/src/DTSegmentUpdator.h"
#include "RecoLocalMuon/DTSegment/src/DTSegmentCleaner.h"
#include "RecoLocalMuon/DTSegment/src/DTHitPairForFit.h"
#include "RecoLocalMuon/DTSegment/src/DTSegmentCand.h"
#include "Geometry/DTGeometry/interface/DTTopology.h"

#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment2D.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment2DCollection.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"

#include <TROOT.h>
#include <TSystem.h>
#include <vector>
#include <string>
#include <iostream>

namespace edm {
  class ParameterSet;
  class EventSetup;
  class InputTag;
}

class TFile;
class TH1F;
class TH2F;
class MuonServiceProxy;

using namespace std;
using namespace edm;
using namespace reco;

struct segmData {
  int nhits;
  double t0;
  int station;
  double dist;
  bool isPhi;
  bool isMatched;
};

typedef std::vector<segmData> segmVect;

//
// class decleration
//

class BetaFromTOF : public edm::EDProducer {
  public:
      explicit BetaFromTOF(const edm::ParameterSet&);
      ~BetaFromTOF();

  private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      double fitT0(double &a, double &b, vector<double> xl, vector<double> yl, vector<double> xr, vector<double> yr );
      void rawFit(double &a, double &b, const vector<double> hitsx, const vector<double> hitsy);

      edm::InputTag TKtrackTags_; 
      edm::InputTag STAtrackTags_; 
      edm::InputTag MuonTags_; 
      edm::InputTag DTSegmentTags_; 

  int theHitsMin;
  bool debug,onlyMatched;

  Handle<reco::TrackCollection> TKTrackCollection;
  Handle<reco::TrackCollection> STATrackCollection;
  Handle<reco::TrackCollection> GLBTrackCollection;
  
  MuonServiceProxy* theService;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
BetaFromTOF::BetaFromTOF(const edm::ParameterSet& iConfig)
  :
  MuonTags_(iConfig.getUntrackedParameter<edm::InputTag>("Muons")),
  DTSegmentTags_(iConfig.getUntrackedParameter<edm::InputTag>("DTsegments")),
  debug(iConfig.getParameter<bool>("debug")),
  onlyMatched(iConfig.getParameter<bool>("OnlyMatched")),
  theHitsMin(iConfig.getParameter<int>("HitsMin"))
{
  // service parameters
  ParameterSet serviceParameters = iConfig.getParameter<ParameterSet>("ServiceParameters");
  // the services
  theService = new MuonServiceProxy(serviceParameters);
  produces<std::vector<float> >();
}


BetaFromTOF::~BetaFromTOF()
{
 
  if (theService) delete theService;

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
BetaFromTOF::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  //using namespace edm;
  using reco::TrackCollection;

  if (debug) 
    cout << " *** Beta from TOF Start ***" << endl;

  ESHandle<GlobalTrackingGeometry> theTrackingGeometry;
  iSetup.get<GlobalTrackingGeometryRecord>().get(theTrackingGeometry);

  ESHandle<DTGeometry> theDTGeometry;
  iSetup.get<MuonGeometryRecord>().get(theDTGeometry);

  vector<float> *outputCollection = new vector<float>;

//  iEvent.getByLabel( TKtrackTags_, TKTrackCollection);
//  const reco::TrackCollection tkTC = *(TKTrackCollection.product());

  Handle<reco::TrackCollection> staMuonsH;
  iEvent.getByLabel("standAloneMuons",staMuonsH);
  const reco::TrackCollection & staMuons = * staMuonsH.product();
  if (debug) 
    cout << " STA Muon collection size: " << staMuons.size() << endl;

  Handle<reco::MuonCollection> allMuons;
  iEvent.getByLabel(MuonTags_,allMuons);
  const reco::MuonCollection & muons = * allMuons.product();
  if (debug) 
    cout << "     Muon collection size: " << muons.size() << endl;

  // Get the DT-Segment collection from the Event
  edm::Handle<DTRecSegment4DCollection> dtRecHits;
  iEvent.getByLabel(DTSegmentTags_, dtRecHits);  
  if (debug) 
    cout << "DT Segment collection size: " << dtRecHits->size() << endl;

  for(reco::MuonCollection::const_iterator mi = muons.begin(); mi != muons.end() ; mi++) {
    TrackRef staMuon = mi->standAloneMuon();

    double betaMeasurements[4]={0,0,0,0};
    double invbeta=0;

  for (reco::TrackCollection::const_iterator candTrack = staMuons.begin(); candTrack != staMuons.end(); ++candTrack) {
    
    
//    cout << " Old STA: eta=" << staMuon->momentum().eta() << "  phi=" << staMuon->momentum().phi() << endl;
//    cout << " New STA: eta=" << candTrack->momentum().eta() << "  phi=" << candTrack->momentum().phi() << endl;
//    cout << " Diff: " << (staMuon->momentum().unit()-candTrack->momentum().unit()) << endl;

    // find the standalone muon matching the global muon
    if ((staMuon->momentum().unit()-candTrack->momentum().unit()).Mag2()>0.01) continue;
//    const Track* candTrack = staMuon.get();

    vector <double> dstnc, dsegm, dtraj, hitWeight;
    int totalWeight=0;

    if (debug) 
      cout << " STA Track:   RecHits: " << (*candTrack).recHitsSize() 
           << " momentum: " << (*candTrack).p() << endl;

    for (trackingRecHit_iterator hi=(*candTrack).recHitsBegin(); hi!=(*candTrack).recHitsEnd(); hi++)

      if (( (*hi)->geographicalId().subdetId() == MuonSubdetId::DT ) && ((*hi)->geographicalId().det() == 2)) {
  
        // Create the ChamberId
        DetId id = (*hi)->geographicalId();
        DTChamberId chamberId(id.rawId());
        DTLayerId layerId(id.rawId());
        DTSuperLayerId slayerId(id.rawId());
        int station = chamberId.station();
        
        // since the rec hits in the trajectory no longer are DTSegments and don't remember their t0, we need to 
        // get the hits directly from the chamber and match them...

        // Look for reconstructed 4D Segments
        DTRecSegment4DCollection::range range = dtRecHits->get(chamberId);
  
        for (DTRecSegment4DCollection::const_iterator rechit = range.first; rechit!=range.second;++rechit){

          // match with the current recHit
          if ((rechit->localPosition()-(*hi)->localPosition()).mag()<0.01) {

            // Check if both phi and theta segments exist in the 4D Segment
	    if ((rechit->hasPhi() && rechit->hasZed()) || !onlyMatched) {

              double t0, dist, ibphi=0, ibtheta=0;

  	      if (rechit->hasPhi()) {
	        const DTRecSegment2D* si = dynamic_cast<const DTRecSegment2D*>(rechit->phiSegment());
	        if (si->specificRecHits().size()>=theHitsMin) {
                const GeomDet* geomDet = theTrackingGeometry->idToDet(si->geographicalId());
                dist = geomDet->toGlobal(si->localPosition()).mag();
  	        t0 = si->t0();
    	        ibphi=1.+t0/dist*30.;
    	        ibtheta=ibphi;
  	        if (debug) cout << " Station " << station << "   Phi 1/beta = " << ibphi << " t0 = " << t0 << endl;
  	        
  	        const vector<DTRecHit1D> hits1d = si->specificRecHits();
  	        if (debug) cout << "             Hits: " << hits1d.size() << endl;
  	        
  	        double a=0, b=0;
  	        vector <double> hitxl,hitxr,hityl,hityr;

  	        for (vector<DTRecHit1D>::const_iterator hiti=hits1d.begin(); hiti!=hits1d.end(); hiti++) {
  	          const GeomDet* dtcell = theTrackingGeometry->idToDet(hiti->geographicalId());
  	          if (hiti->lrSide()==DTEnums::Left) {
   	            hitxl.push_back(geomDet->toLocal(dtcell->toGlobal(hiti->localPosition())).z());
  	            hityl.push_back(geomDet->toLocal(dtcell->toGlobal(hiti->localPosition())).x());
  	          } else {
   	            hitxr.push_back(geomDet->toLocal(dtcell->toGlobal(hiti->localPosition())).z());
  	            hityr.push_back(geomDet->toLocal(dtcell->toGlobal(hiti->localPosition())).x());
  	          }    
  	        }
  	        
  	        fitT0(a,b,hitxl,hityl,hitxr,hityr);  
  	        
  	        for (vector<DTRecHit1D>::const_iterator hiti=hits1d.begin(); hiti!=hits1d.end(); hiti++) {
  	          const GeomDet* dtcell = theTrackingGeometry->idToDet(hiti->geographicalId());

                  dist = dtcell->toGlobal(hiti->localPosition()).mag();
                  double layerZ  = geomDet->toLocal(dtcell->toGlobal(hiti->localPosition())).z();
//		  double segmLocalPos = si->localPosition().x()-layerZ*si->localDirection().x();
		  double segmLocalPos = b+layerZ*a;
		  double hitLocalPos = geomDet->toLocal(dtcell->toGlobal(hiti->localPosition())).x();
                  int hitSide;
                  if (hiti->lrSide()==DTEnums::Left) hitSide=-1; else hitSide=1;
                  double t0_segm = (-(hitSide*segmLocalPos)+(hitSide*hitLocalPos))/0.00543;
                  
                  dstnc.push_back(dist);
                  dsegm.push_back(t0_segm);
                  hitWeight.push_back(((double)hits1d.size()-2.)/(double)hits1d.size());

                  if (debug) cout << "             dist: " << dist << " pos: " << hitLocalPos
                                  << " Z: " << layerZ << " Segm: " << segmLocalPos
                                  << " t0: " << t0_segm << " 1/beta: " << 1.+t0_segm/dist*30. <<
                                  endl;
  	        }
  	        
                totalWeight+=hits1d.size()-2;
  	        }
  	      }
	    
	      if (rechit->hasZed()) {
	        const DTRecSegment2D* si = dynamic_cast<const DTRecSegment2D*>(rechit->zSegment());
	        if (si->specificRecHits().size()>=theHitsMin) {
                const GeomDet* geomDet = theTrackingGeometry->idToDet(si->geographicalId());
                dist = geomDet->toGlobal(si->localPosition()).mag();
  	        t0 = si->t0();
	        ibtheta=1.+t0/dist*30.;
	        if (!ibphi) ibphi=ibtheta;
 	        if (debug) cout << " Station " << station << " Theta 1/beta = " << ibtheta << " t0 = " << t0 << endl;

  	        const vector<DTRecHit1D> hits1d = si->specificRecHits();
  	        if (debug) cout << "             Hits: " << hits1d.size() << endl;

  	        double a=0, b=0;
  	        vector <double> hitxl,hitxr,hityl,hityr;

  	        for (vector<DTRecHit1D>::const_iterator hiti=hits1d.begin(); hiti!=hits1d.end(); hiti++) {
  	          const GeomDet* dtcell = theTrackingGeometry->idToDet(hiti->geographicalId());
  	          if (hiti->lrSide()==DTEnums::Left) {
   	            hitxl.push_back(geomDet->toLocal(dtcell->toGlobal(hiti->localPosition())).z());
  	            hityl.push_back(geomDet->toLocal(dtcell->toGlobal(hiti->localPosition())).x());
  	          } else {
   	            hitxr.push_back(geomDet->toLocal(dtcell->toGlobal(hiti->localPosition())).z());
  	            hityr.push_back(geomDet->toLocal(dtcell->toGlobal(hiti->localPosition())).x());
  	          }    
  	        }
  	        
  	        fitT0(a,b,hitxl,hityl,hitxr,hityr);  

  	        for (vector<DTRecHit1D>::const_iterator hiti=hits1d.begin(); hiti!=hits1d.end(); hiti++) {
  	          const GeomDet* dtcell = theTrackingGeometry->idToDet(hiti->geographicalId());

                  dist = dtcell->toGlobal(hiti->localPosition()).mag();
//                  double layerZ  = geomDet->toLocal(dtcell->position()).z();
                  double layerZ  = geomDet->toLocal(dtcell->toGlobal(hiti->localPosition())).z();
//		  double segmLocalPos = si->localPosition().x()-layerZ*si->localDirection().x();
		  double segmLocalPos = b+layerZ*a;
		  double hitLocalPos = geomDet->toLocal(dtcell->toGlobal(hiti->localPosition())).x();
                  int hitSide;
                  if (hiti->lrSide()==DTEnums::Left) hitSide=-1; else hitSide=1;
                  double t0_segm = (-(hitSide*segmLocalPos)+(hitSide*hitLocalPos))/0.00543;
                  
                  dstnc.push_back(dist);
                  dsegm.push_back(t0_segm);
                  hitWeight.push_back(((double)hits1d.size()-2.)/(double)hits1d.size());

                  if (debug) cout << "             dist: " << dist << " pos: " << hitLocalPos
                                  << " Z: " << layerZ << " Segm: " << segmLocalPos
                                  << " t0: " << t0_segm << " 1/beta: " << 1.+t0_segm/dist*30. <<
                                  endl;
  	        }
  	        
                totalWeight+=hits1d.size()-2;
  	        }
	      }  
	      
	      // Compute the inverse beta for this station by averaging phi and theta calculations
	      // (for the segments that passed the cut on number of hits)
	      // TODO: A weighted average?
	      betaMeasurements[station-1]=(ibphi+ibtheta)/2.;
	    }
          }
        } // rechit
    } // hi

    invbeta=0;
    // calculate the value and error of 1/beta from the complete set of 1D hits
    if (debug)
      cout << " Points for global fit: " << endl;
    for (int i=0;i<dstnc.size();i++) {
//      cout << "    Dstnc: " << dstnc.at(i) << "   delta t0(hit-segment): " << dsegm.at(i) << "   weight: " << hitWeight.at(i); 
//      cout << " Local 1/beta: " << 1.+dsegm.at(i)/dstnc.at(i)*30. << endl;
      invbeta+=(1.+dsegm.at(i)/dstnc.at(i)*30.)*hitWeight.at(i)/totalWeight;
    }

    double invbetaerr=0,diff;
    for (int i=0;i<dstnc.size();i++) {
      diff=(1.+dsegm.at(i)/dstnc.at(i)*30.)-invbeta;
      invbetaerr+=diff*diff*hitWeight.at(i);
    }
    
    invbetaerr=sqrt(invbetaerr)/totalWeight;

    if (debug)
      cout << " Measured 1/beta: " << invbeta << " +/- " << invbetaerr << endl;

    
    // End the loop over STA muons - since we already found the matching one
    break;
  }  //candTrack

    outputCollection->push_back(invbeta);
 
    invbeta=0;    
    int mcount=0;
    
    // Average the nonzero measurements from the muon stations
    for (int s=0;s<4;s++) 
      if (betaMeasurements[s]) {
        invbeta+=betaMeasurements[s];
        mcount++;
      }
    
    if (mcount) invbeta/=mcount;
    
    if (debug)
      cout << " (1/beta from segments): " << invbeta << endl;

  } // mi

  std::auto_ptr<vector<float> > estimator(outputCollection);
  iEvent.put(estimator);

}

double
BetaFromTOF::fitT0(double &a, double &b, vector<double> xl, vector<double> yl, vector<double> xr, vector<double> yr ) {

  double ar=0,br=0,al=0,bl=0;
  
  // Do the fit separately for left and right hits
  if (xl.size()>1) rawFit(al,bl,xl,yl); 
    else if (xl.size()==1) bl=yl[0];
  if (xr.size()>1) rawFit(ar,br,xr,yr);
    else if (xr.size()==1) br=yr[0];

  // If there's only 1 hit on one side, take the slope from the other side and adjust the constant
  // so that the line passes through the single hit  
  
  if (al==0) { 
    al=ar; 
    if (bl==0) bl=br; 
      else bl-=al*xl[0];
  }    
  if (ar==0) {
    ar=al;
    if (br==0) br=bl; 
      else br-=ar*xr[0];
  }

  // The best fit is the average of the left and right fits

  a=(al+ar)/2.;
  b=(bl+br)/2.;

  // Now we can calculate the t0 correction for the hits

  double t0_left=0, t0_right=0, t0_corr;
  if (xl.size()) 
    for (unsigned int i=0; i<xl.size(); i++) 
      t0_left+=yl[i]-a*xl[i]-b;
  if (xr.size()) 
    for (unsigned int i=0; i<xr.size(); i++) 
      t0_right+=yr[i]-a*xr[i]-b;
  
  t0_corr=(t0_right-t0_left)/(xl.size()+xr.size());
  if ((t0_left==0) || (t0_right==0)) t0_corr=0;
  // convert drift distance to time
  // TODO: a smarter conversion? (using 1D rechit algo?)
  t0_corr/=0.00543;
  
  return t0_corr;
}


void 
BetaFromTOF::rawFit(double &a, double &b, const vector<double> hitsx, const vector<double> hitsy) {

  double s=0,sx=0,sy=0,x,y;
  double sxx=0,sxy=0;

  a=b=0;
  if (hitsx.size()==0) return;
    
  if (hitsx.size()==1) {
    b=hitsy[0];
  } else {
    for (unsigned int i = 0; i != hitsx.size(); i++) {
      x=hitsx[i];
      y=hitsy[i];
      sy += y;
      sxy+= x*y;
      s += 1.;
      sx += x;
      sxx += x*x;
    }
    // protect against a vertical line???

    double d = s*sxx - sx*sx;
    b = (sxx*sy- sx*sxy)/ d;
    a = (s*sxy - sx*sy) / d;
  }
}



// ------------ method called once each job just before starting event loop  ------------
void 
BetaFromTOF::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
BetaFromTOF::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(BetaFromTOF);
