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
// $Id: BetaFromTOF.cc,v 1.13 2008/03/14 12:04:50 ptraczyk Exp $
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


#include "AnalysisDataFormats/SUSYBSMObjects/interface/HSCParticle.h"
                   

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
      void rawFit(double &a, double &da, double &b, double &db, const vector<double> hitsx, const vector<double> hitsy);
      void textplot(vector<double> x, vector <double> y, vector <double> side);
      void processSegment( const DTRecSegment2D *si, vector<double> &dstnc, vector<double> &dsegm, 
                           vector<double> &dtraj, vector<double> &hitWeight, vector<double> &left );

      edm::InputTag TKtrackTags_; 
      edm::InputTag STAtrackTags_; 
      edm::InputTag MuonTags_; 
      edm::InputTag DTSegmentTags_; 

  unsigned int theHitsMin;
  bool debug;

  Handle<reco::TrackCollection> TKTrackCollection;
  Handle<reco::TrackCollection> STATrackCollection;
  Handle<reco::TrackCollection> GLBTrackCollection;
  ESHandle<GlobalTrackingGeometry> theTrackingGeometry;
  
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
  theHitsMin(iConfig.getParameter<int>("HitsMin")),
  debug(iConfig.getParameter<bool>("debug")) 
{
  // service parameters
  ParameterSet serviceParameters = iConfig.getParameter<ParameterSet>("ServiceParameters");
  // the services
  theService = new MuonServiceProxy(serviceParameters);
//  produces<std::vector<float> >();
  produces<susybsm::MuonTOFCollection>();
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

  iSetup.get<GlobalTrackingGeometryRecord>().get(theTrackingGeometry);

  ESHandle<DTGeometry> theDTGeometry;
  iSetup.get<MuonGeometryRecord>().get(theDTGeometry);

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

  MuonRefProd muRP(allMuons); 
  using namespace susybsm;
  MuonTOFCollection *outputCollection = new MuonTOFCollection(muRP);

  // Get the DT-Segment collection from the Event
  edm::Handle<DTRecSegment4DCollection> dtRecHits;
  iEvent.getByLabel(DTSegmentTags_, dtRecHits);  
  if (debug) 
    cout << "DT Segment collection size: " << dtRecHits->size() << endl;

  size_t muId=0;
  for(reco::MuonCollection::const_iterator mi = muons.begin(); mi != muons.end() ; mi++,muId++) {
    TrackRef staMuon = mi->standAloneMuon();

    double stationHits[4]={0,0,0,0};
    double invbeta=0;
    DriftTubeTOF tof;
    
    tof.nHits=0;
    tof.nStations=0;

    for (reco::TrackCollection::const_iterator candTrack = staMuons.begin(); candTrack != staMuons.end(); ++candTrack) {

      // find the standalone muon matching the global muon
      if ((staMuon->momentum().unit()-candTrack->momentum().unit()).Mag2()>0.01) continue;
//      const Track* candTrack = staMuon.get();

      vector <double> dstnc, dsegm, dtraj, hitWeight, left;
      int totalWeight=0;
      int nStations=0;

      if (debug) 
        cout << " STA Track:   RecHits: " << (*candTrack).recHitsSize() 
             << " momentum: " << (*candTrack).p() << endl;

      for (trackingRecHit_iterator hi=(*candTrack).recHitsBegin(); hi!=(*candTrack).recHitsEnd(); hi++) {

        // Check that we're in DT
        if ( ((*hi)->geographicalId().subdetId() != MuonSubdetId::DT ) 
          || ((*hi)->geographicalId().det() != 2)) continue;
  
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

	// Check for showers in a cone around the 4D hit                 
        for (DTRecSegment4DCollection::const_iterator rechit = range.first; rechit!=range.second;++rechit)
          if ((rechit->localPosition()-(*hi)->localPosition()).mag()<20.) stationHits[station-1]++;

        for (DTRecSegment4DCollection::const_iterator rechit = range.first; rechit!=range.second;++rechit) {

          // match with the current recHit
          if ((rechit->localPosition()-(*hi)->localPosition()).mag()>0.01) continue;

          // Check if both phi and theta segments exist in the 4D Segment
          if ((!rechit->hasPhi()) || (!rechit->hasZed())) continue;

          nStations++;

          double t0, dist, ibphi=0, ibtheta=0;

	  const DTRecSegment2D* sphi = dynamic_cast<const DTRecSegment2D*>(rechit->phiSegment());
          const GeomDet* geomDetPhi = theTrackingGeometry->idToDet(sphi->geographicalId());
	  const DTRecSegment2D* szed = dynamic_cast<const DTRecSegment2D*>(rechit->zSegment());
          const GeomDet* geomDetZed = theTrackingGeometry->idToDet(szed->geographicalId());

          const vector<DTRecHit1D> hits1dphi = sphi->specificRecHits();
          const vector<DTRecHit1D> hits1dzed = szed->specificRecHits();

          dist = geomDetPhi->toGlobal(sphi->localPosition()).mag();
          t0 = sphi->t0();


  	  for (vector<DTRecHit1D>::const_iterator hiti=hits1dphi.begin(); hiti!=hits1dphi.end(); hiti++) {
  	    const GeomDet* dtcell = theTrackingGeometry->idToDet(hiti->geographicalId());
            TimeMeasurement thisHit;
                  
            thisHit.driftCell = hiti->geographicalId();
            if (hiti->lrSide()==DTEnums::Left) thisHit.isLeft=true; else thisHit.isLeft=false;
            thisHit.isPhi = true;
            thisHit.posInLayer = geomDetPhi->toLocal(dtcell->toGlobal(hiti->localPosition())).x();
            thisHit.distIP = dtcell->toGlobal(hiti->localPosition()).mag();
            thisHit.station = station;
                  
            tof.timeMeasurements.push_back(thisHit);
                  
          }

          if (hits1dphi.size()>=theHitsMin) {
            ibphi=1.+t0/dist*30.;
            ibtheta=ibphi;
            if (debug) 
              cout << " Station " << station << "   Phi 1/beta = " << ibphi << " t0 = " << t0 << endl;
	          
	    processSegment(sphi, dstnc, dsegm, dtraj, hitWeight, left);
	          
            totalWeight+=hits1dphi.size()-2;

  	  }

          dist = geomDetZed->toGlobal(szed->localPosition()).mag();
  	  t0 = szed->t0();

  	  for (vector<DTRecHit1D>::const_iterator hiti=hits1dzed.begin(); hiti!=hits1dzed.end(); hiti++) {
            const GeomDet* dtcell = theTrackingGeometry->idToDet(hiti->geographicalId());
            TimeMeasurement thisHit;
                  
            thisHit.driftCell = hiti->geographicalId();
            if (hiti->lrSide()==DTEnums::Left) thisHit.isLeft=true; else thisHit.isLeft=false;
            thisHit.isPhi = false;
            thisHit.posInLayer = geomDetZed->toLocal(dtcell->toGlobal(hiti->localPosition())).x();
            thisHit.distIP = dtcell->toGlobal(hiti->localPosition()).mag();
            thisHit.station = station;
                  
            tof.timeMeasurements.push_back(thisHit);
          }

          if (hits1dzed.size()>=theHitsMin) {
            ibtheta=1.+t0/dist*30.;
            if (!ibphi) ibphi=ibtheta;
            if (debug) 
              cout << " Station " << station << " Theta 1/beta = " << ibtheta << " t0 = " << t0 << endl;

            processSegment(szed, dstnc, dsegm, dtraj, hitWeight, left);

            totalWeight+=hits1dzed.size()-2;
          }
              	        
        } // rechit
      } // hi

      invbeta=0;
      // calculate the value and error of 1/beta from the complete set of 1D hits
      if (debug)
        cout << " Points for global fit: " << dstnc.size() << endl;

      // inverse beta - weighted average of the contributions from individual hits
      for (unsigned int i=0;i<dstnc.size();i++) {
//        cout << "    Dstnc: " << dstnc.at(i) << "   delta t0(hit-segment): " << dsegm.at(i) << "   weight: " << hitWeight.at(i); 
//        cout << " Local 1/beta: " << 1.+dsegm.at(i)/dstnc.at(i)*30. << endl;
        invbeta+=(1.+dsegm.at(i)/dstnc.at(i)*30.)*hitWeight.at(i)/totalWeight;
      }
    
      // the dispersion of inverse beta
      double invbetaerr=0,diff;
      for (unsigned int i=0;i<dstnc.size();i++) {
        diff=(1.+dsegm.at(i)/dstnc.at(i)*30.)-invbeta;
        invbetaerr+=diff*diff*hitWeight.at(i);
      }
    
      invbetaerr=sqrt(invbetaerr)/totalWeight;

      if (debug)
        cout << " Measured 1/beta: " << invbeta << " +/- " << invbetaerr << endl;

      tof.invBeta=invbeta;
      tof.invBetaErr=invbetaerr;
      
      // unconstrained fit to the full set of points
      vector <double> x,y;
      double freeBeta, freeBetaErr, freeTime, freeTimeErr, vertexTime=0, vertexTimeErr=0;    

      for (unsigned int i=0;i<dstnc.size();i++) {
//        cout << "    Dstnc: " << dstnc.at(i) << "   delta t0(hit-segment): " << dsegm.at(i) << "   weight: " << hitWeight.at(i); 
//        cout << " Local 1/beta: " << 1.+dsegm.at(i)/dstnc.at(i)*30. << endl;
        x.push_back(dstnc.at(i)/30.);
        y.push_back(dsegm.at(i)+dstnc.at(i)/30.);
        vertexTime+=dsegm.at(i)*hitWeight.at(i)/totalWeight;
//        cout << "    x: " << x.at(i) << "   y: " << y.at(i) << " Local 1/beta: " << 1.+dsegm.at(i)/dstnc.at(i)*30. << endl;
      }

      if ((invbetaerr>0.07) || (stationHits[0]>1)) {
        cout << " *** Beta: " << invbeta << " +/- " << invbetaerr << "   All Hits: " << dstnc.size() << endl;
        for (int a=0;a<4;a++) cout << "     St: " << a+1 << "   4d hits: " << stationHits[a] << endl;
        for (unsigned int i=0;i<dstnc.size();i++)
          cout << "    x: " << dstnc.at(i) << "   y: " << dsegm.at(i) << " Local 1/beta: " << 1.+dsegm.at(i)/dstnc.at(i)*30. << endl;
      }

      for (unsigned int i=0;i<dstnc.size();i++) {
        diff=dsegm.at(i)-vertexTime;
        vertexTimeErr+=diff*diff*hitWeight.at(i);
      }
      vertexTimeErr=sqrt(vertexTimeErr)/totalWeight;
      
      rawFit(freeBeta, freeBetaErr, freeTime, freeTimeErr, x, y);
//      textplot(x,y,left);
    
      tof.invBetaFree=freeBeta;
      tof.invBetaFreeErr=freeBetaErr;
      tof.vertexTime=vertexTime;
      tof.vertexTimeErr=vertexTimeErr;
      
      tof.nHits = dstnc.size();
      tof.nStations=nStations;

      if (debug) {
        cout << " Free 1/beta: " << freeBeta << " +/- " << freeBetaErr << endl;   
        cout << "   Free time: " << freeTime << " +/- " << freeTimeErr << endl;   
        cout << " Vertex time: " << vertexTime << " +/- " << freeTimeErr << endl;   
      }  

      // End the loop over STA muons - since we already found the matching one
      break;
    }  //candTrack

    outputCollection->setValue(muId,tof); 

  } // mi

  std::auto_ptr<MuonTOFCollection> res(outputCollection);
  iEvent.put(res);

  if (debug) 
    cout << " *** Beta from TOF End ***" << endl;

}


void BetaFromTOF::processSegment( const DTRecSegment2D* si, vector<double> &dstnc, vector<double> &dsegm, 
                                  vector<double> &dtraj, vector<double> &hitWeight, vector<double> &left ) {

  const GeomDet* geomDet = theTrackingGeometry->idToDet(si->geographicalId());
//  double  dist = geomDet->toGlobal(si->localPosition()).mag();
//  double  t0 = si->t0();

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
  
  double t0 = fitT0(a,b,hitxl,hityl,hitxr,hityr);
    	        
  if (debug)
    cout << "             t0 from fit: " << t0 << endl;  
  	        
  for (vector<DTRecHit1D>::const_iterator hiti=hits1d.begin(); hiti!=hits1d.end(); hiti++) {
    const GeomDet* dtcell = theTrackingGeometry->idToDet(hiti->geographicalId());

    double dist = dtcell->toGlobal(hiti->localPosition()).mag();
    double layerZ  = geomDet->toLocal(dtcell->toGlobal(hiti->localPosition())).z();
    double segmLocalPos = b+layerZ*a;
    double hitLocalPos = geomDet->toLocal(dtcell->toGlobal(hiti->localPosition())).x();
    int hitSide;
    if (hiti->lrSide()==DTEnums::Left) hitSide=-1; else hitSide=1;
    double t0_segm = (-(hitSide*segmLocalPos)+(hitSide*hitLocalPos))/0.00543;
                  
    dstnc.push_back(dist);
    dsegm.push_back(t0_segm);
    left.push_back(hitSide);
    hitWeight.push_back(((double)hits1d.size()-2.)/(double)hits1d.size());

    if (debug) cout << "             dist: " << dist << " pos: " << hitLocalPos 
                    << " Z: " << layerZ << " Segm: " << segmLocalPos
                    << " t0: " << t0_segm << " 1/beta: " << 1.+t0_segm/dist*30. <<
                      endl;
  }

}


double
BetaFromTOF::fitT0(double &a, double &b, vector<double> xl, vector<double> yl, vector<double> xr, vector<double> yr ) {

  double sx=0,sy=0,sxy=0,sxx=0,ssx=0,ssy=0,s=0,ss=0;

  for (unsigned int i=0; i<xl.size(); i++) {
    sx+=xl[i];
    sy+=yl[i];
    sxy+=xl[i]*yl[i];
    sxx+=xl[i]*xl[i];
    s++;
    ssx+=xl[i];
    ssy+=yl[i];
    ss++;
  } 

  for (unsigned int i=0; i<xr.size(); i++) {
    sx+=xr[i];
    sy+=yr[i];
    sxy+=xr[i]*yr[i];
    sxx+=xr[i]*xr[i];
    s++;
    ssx-=xr[i];
    ssy-=yr[i];
    ss--;
  } 

  double delta = ss*ss*sxx+s*sx*sx+s*ssx*ssx-s*s*sxx-2*ss*sx*ssx;
  
  double t0_corr=0.;

  if (delta) {
    a=(ssy*s*ssx+sxy*ss*ss+sy*sx*s-sy*ss*ssx-ssy*sx*ss-sxy*s*s)/delta;
    b=(ssx*sy*ssx+sxx*ssy*ss+sx*sxy*s-sxx*sy*s-ssx*sxy*ss-sx*ssy*ssx)/delta;
    t0_corr=(ssx*s*sxy+sxx*ss*sy+sx*sx*ssy-sxx*s*ssy-sx*ss*sxy-ssx*sx*sy)/delta;
  }

  // convert drift distance to time
  t0_corr/=-0.00543;

  return t0_corr;
}


void 
BetaFromTOF::rawFit(double &a, double &da, double &b, double &db, const vector<double> hitsx, const vector<double> hitsy) {

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
    da = sqrt(sxx/d);
    db = sqrt(s/d);
  }
}

void 
BetaFromTOF::textplot(vector<double> x, vector <double> y, vector <double> side)
{

  int data[82][42];
  double xmax=0,ymax=0,xmin=999,ymin=999;
  
  for (unsigned int i=0;i<x.size(); i++) {
    if (x.at(i)<xmin) xmin=x.at(i);    
    if (y.at(i)<ymin) ymin=y.at(i);    
    if (x.at(i)>xmax) xmax=x.at(i);    
    if (y.at(i)>ymax) ymax=y.at(i);    
  }
  
  double xfact=(xmax-xmin+1)/80.;
  double yfact=(ymax-ymin+1)/30.;

  for (int ix=0;ix<82;ix++)
    for (int iy=0;iy<32;iy++)
      data[ix][iy]=0;

//  cout << xmin << " " << xmax << " " << ymin << " " << ymax << endl;

  for (unsigned int i=0;i<x.size(); i++) {
    int xloc = (int)((x.at(i)-xmin)/xfact);
    int yloc = (int)((y.at(i)-ymin)/yfact);
    if ((xloc>=0) && (xloc<82) && (yloc>=0) && (yloc<32)) data[xloc][yloc]=(int)side.at(i); 
      else cout << "ERROR! " << x.at(i) << " " << xloc << " " << yloc << endl;
  }

  for (int iy=31;iy!=0;iy--) {
    cout << setw(4) << (iy*yfact+ymin) << " ";
    for (int ix=0;ix<82;ix++) {
      if (data[ix][iy]) {
        if (data[ix][iy]==1) cout << "P"; 
        if (data[ix][iy]==-1) cout << "L"; 
      } else {
       if (fabs((iy*yfact+ymin)-(ix*xfact+xmin))<.5) cout << "*"; 
        else cout << " ";
      }
    }
    cout << endl;
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
