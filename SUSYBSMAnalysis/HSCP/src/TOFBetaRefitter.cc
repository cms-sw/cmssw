// -*- C++ -*-
//
// Package:    TOFBetaRefitter
// Class:      TOFBetaRefitter
// 
/**\class TOFBetaRefitter TOFBetaRefitter.cc SUSYBSMAnalysis/HSCP/src/TOFBetaRefitter.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Traczyk Piotr
//         Created:  Thu Oct 11 15:01:28 CEST 2007
// $Id: TOFBetaRefitter.cc,v 1.1 2008/03/17 23:24:07 ptraczyk Exp $
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

class TOFBetaRefitter : public edm::EDProducer {
  public:
      explicit TOFBetaRefitter(const edm::ParameterSet&);
      ~TOFBetaRefitter();

  private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      double fitT0(double &a, double &b, vector<double> xl, vector<double> yl, vector<double> xr, vector<double> yr );
      void textplot(vector<double> x, vector <double> y, vector <double> side);

      edm::InputTag MuonTags_; 
      edm::InputTag DTSegmentTags_; 

  unsigned int theHitsMinTheta, theHitsMinPhi;
  bool debug;
  bool requireLR;
  bool correctDist;
  double thePruneCut;

  ESHandle<GlobalTrackingGeometry> theTrackingGeometry;
  
  MuonServiceProxy* theService;
};

//
// constructors and destructor
//
TOFBetaRefitter::TOFBetaRefitter(const edm::ParameterSet& iConfig)
  :
  MuonTags_(iConfig.getUntrackedParameter<edm::InputTag>("Muons")),
  DTSegmentTags_(iConfig.getUntrackedParameter<edm::InputTag>("DTsegments")),
  theHitsMinTheta(iConfig.getParameter<int>("HitsMinTheta")),
  theHitsMinPhi(iConfig.getParameter<int>("HitsMinPhi")),
  correctDist(iConfig.getParameter<bool>("correctDist")),
  requireLR(iConfig.getParameter<bool>("requireLR")),
  debug(iConfig.getParameter<bool>("debug")),
  thePruneCut(iConfig.getParameter<double>("pruneCut"))
{
  // service parameters
  ParameterSet serviceParameters = iConfig.getParameter<ParameterSet>("ServiceParameters");

  // the services
  theService = new MuonServiceProxy(serviceParameters);

  produces<susybsm::MuonTOFCollection>();
}


TOFBetaRefitter::~TOFBetaRefitter()
{
  if (theService) delete theService;
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
TOFBetaRefitter::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace susybsm;

  if (debug) 
    cout << "*** Refit beta from TOF Start ***" << endl;

  theService->update(iSetup);

  iSetup.get<GlobalTrackingGeometryRecord>().get(theTrackingGeometry);

  ESHandle<DTGeometry> theDTGeometry;
  iSetup.get<MuonGeometryRecord>().get(theDTGeometry);

  Handle< MuonTOFCollection >  betaRecoH;
  iEvent.getByLabel("betaFromTOF",betaRecoH);
  const MuonTOFCollection & betaReco = *betaRecoH.product();

  Handle<reco::MuonCollection> allMuons;
  iEvent.getByLabel(MuonTags_,allMuons);
  const reco::MuonCollection & muons = * allMuons.product();
  if (debug) 
    cout << "     Muon collection size: " << muons.size() << endl;

  MuonRefProd muRP(allMuons); 
  MuonTOFCollection *outputCollection = new MuonTOFCollection(muRP);

  Propagator *propag = &*theService->propagator("SteppingHelixPropagatorAny")->clone();

  // Get the DT-Segment collection from the Event
  edm::Handle<DTRecSegment4DCollection> dtRecHits;
  iEvent.getByLabel(DTSegmentTags_, dtRecHits);  
  if (debug) 
    cout << "DT Segment collection size: " << dtRecHits->size() << endl;

  size_t muId = 0;

  for(MuonTOFCollection::const_iterator mi = betaReco.begin(); mi != betaReco.end() ; mi++, muId++) {

    //the associated muon track
    TrackRef muonTrack;
    if ((*mi).first->combinedMuon().isNonnull()) muonTrack = (*mi).first->combinedMuon();
      else
        if ((*mi).first->standAloneMuon().isNonnull()) muonTrack = (*mi).first->standAloneMuon();
          else continue;
          
    math::XYZPoint  pos=muonTrack->innerPosition();
    math::XYZVector mom=muonTrack->innerMomentum();

    GlobalPoint  posp(pos.x(), pos.y(), pos.z());
    GlobalVector momv(mom.x(), mom.y(), mom.z());

    FreeTrajectoryState muonFTS(posp, momv, (TrackCharge)muonTrack->charge(), theService->magneticField().product());

    double invbeta = (*mi).second.invBeta;
    double invbetaerr = (*mi).second.invBetaErr;
    
    vector<TimeMeasurement> tms = (*mi).second.timeMeasurements;
      
    vector <double> dstnc, dsegm, dtraj, hitWeight, left;
    int totalWeight=0;
    float invbeta_hits=0;
    int nStations=0;
      
    unsigned int minHits[2]={theHitsMinPhi,theHitsMinTheta}; // Phi, Zed

    // Rebuild segments
    for (int sta=1;sta<5;sta++)
        for (int phi=0;phi<2;phi++) {
          vector <TimeMeasurement> seg;
          for (vector<TimeMeasurement>::iterator tm=tms.begin(); tm!=tms.end(); ++tm) 
            if ((tm->station==sta) && (tm->isPhi==phi)) seg.push_back(*tm);
          
          unsigned int segsize = seg.size();
          
          if (segsize<minHits[phi]) continue;

          double a=0, b=0, celly, chi2, chi2max, t0=0.;
          vector <double> hitxl,hitxr,hityl,hityr;
          vector<TimeMeasurement>::iterator tmmax;
          bool firstpass=true;

          if (debug) 
            cout << endl << " *** New segment" << endl;

          do {
            hitxl.clear();
            hityl.clear();
            hitxr.clear();
            hityr.clear();
            tmmax=seg.begin();
            chi2max=-1.;
            segsize=seg.size();

            for (vector<TimeMeasurement>::iterator tm=seg.begin(); tm!=seg.end(); ++tm) {
 
              DetId id = tm->driftCell;
              const GeomDet* dtcell = theTrackingGeometry->idToDet(id);
              DTChamberId chamberId(id.rawId());
              const GeomDet* dtcham = theTrackingGeometry->idToDet(chamberId);

              celly=dtcham->toLocal(dtcell->position()).z();
            
              if (tm->isLeft) {
                hitxl.push_back(celly);
                hityl.push_back(tm->posInLayer);
              } else {
                hitxr.push_back(celly);
                hityr.push_back(tm->posInLayer);
              }    
            }

            t0 = fitT0(a,b,hitxl,hityl,hitxr,hityr);

            for (vector<TimeMeasurement>::iterator tm=seg.begin(); tm!=seg.end(); ++tm) {
 
              DetId id = tm->driftCell;
              const GeomDet* dtcell = theTrackingGeometry->idToDet(id);
              DTChamberId chamberId(id.rawId());
              const GeomDet* dtcham = theTrackingGeometry->idToDet(chamberId);

              celly=dtcham->toLocal(dtcell->position()).z();
              chi2 = (a * celly + b - tm->posInLayer)/0.02;
              chi2*= chi2;
            
              // calculate chi2 if this is not the 1-st pass
              if ((chi2>chi2max) && (a!=0)) {
                chi2max=chi2;
                tmmax=tm;
              }
            
              if (debug) 
                cout << " Hit x= " << tm->posInLayer << "  z= " << celly << "  chi2= " << chi2 << endl;
            
            }
            
            if (chi2max>thePruneCut) seg.erase(tmmax); 
            firstpass=false;

            if (debug)     	        
              cout << "     Segment size: " << seg.size() << "  t0 from fit: " << t0 << endl;  
//              cout << "     Left size: " << hitxl.size() << "  Right size: " << hitxr.size() <<  endl;  
//              cout << "   a= " << a << "   b= " << b << "   chi2max " << chi2max << endl;
            
          } while ((segsize!=seg.size()) && (seg.size()>2) && (a!=0));

          if ((t0==0) && (requireLR)) {
            if (debug)
              cout << "     t0 = zero, Left hits: " << hitxl.size() << " Right hits: " << hitxr.size() << endl;
            continue;
          }

          if (seg.size()<minHits[phi]) continue;

          nStations++;

          for (vector<TimeMeasurement>::const_iterator tm=seg.begin(); tm!=seg.end(); ++tm) {

            DetId id = tm->driftCell;
            const GeomDet* dtcell = theTrackingGeometry->idToDet(id);
            DTChamberId chamberId(id.rawId());
            const GeomDet* dtcham = theTrackingGeometry->idToDet(chamberId);

            celly=dtcham->toLocal(dtcell->position()).z();

            double dist = tm->distIP;
            double layerZ  = dtcham->toLocal(dtcell->position()).z();
            double segmLocalPos = b+layerZ*a;
            double hitLocalPos = tm->posInLayer;
            int hitSide = -tm->isLeft*2+1;
            double t0_segm = (-(hitSide*segmLocalPos)+(hitSide*hitLocalPos))/0.00543;
            
            std::pair< TrajectoryStateOnSurface, double> tsos;
            tsos=propag->propagateWithPath(muonFTS,dtcell->surface());
            
            if (tsos.first.isValid() && correctDist) dist = tsos.second+posp.mag();
            
            if ((debug) || (fabs(dist)>100.)) {
              cout << " Dist: " << dist << "   segm: " << segmLocalPos << "   hit: " << hitLocalPos;
              if (tsos.first.isValid()) cout << " traj: " << dtcham->toLocal(tsos.first.globalPosition()) << " path: " << tsos.second+posp.mag();
              cout << " Start: " << posp;
              cout << endl;
            }

            dstnc.push_back(dist);
            dsegm.push_back(t0_segm);
            left.push_back(hitSide);
            hitWeight.push_back(((double)seg.size()-2.)/(double)seg.size());

          }
          
          totalWeight+=seg.size()-2;

        }
     
    // calculate the value and error of 1/beta from the complete set of 1D hits
    if (debug) 
      cout << "     Points for global fit: " << dstnc.size() << endl;

    // inverse beta - weighted average of the contributions from individual hits
    for (unsigned int i=0;i<dstnc.size();i++) {
      invbeta_hits+=(1.+dsegm.at(i)/dstnc.at(i)*30.)*hitWeight.at(i)/totalWeight;
//      cout << "    Dstnc: " << dstnc.at(i) << "   delta t0(hit-segment): " << dsegm.at(i) << "   weight: " << hitWeight.at(i); 
//      cout << " Local 1/beta: " << 1.+dsegm.at(i)/dstnc.at(i)*30. << endl;
    }
    
    // the dispersion of inverse beta
    double invbetaerr_hits=0,diff;
    for (unsigned int j=0;j<dstnc.size();j++) {
      diff=(1.+dsegm.at(j)/dstnc.at(j)*30.)-invbeta_hits;
      invbetaerr_hits+=diff*diff*hitWeight.at(j);
    }
    
    invbetaerr_hits=sqrt(invbetaerr_hits)/totalWeight;

    if (fabs(invbeta_hits-1.)>1.) {
      for (unsigned int i=0;i<dstnc.size();i++) {
        cout << "    Dstnc: " << dstnc.at(i) << "   delta t0(hit-segment): " << dsegm.at(i) << "   weight: " << hitWeight.at(i); 
        cout << " Local 1/beta: " << 1.+dsegm.at(i)/dstnc.at(i)*30. << endl;
      }
    }

    if ((debug) || (fabs(invbeta-invbeta_hits)>0.3)) {
      cout << " New 1/beta: " << invbeta_hits << " +/- " << invbetaerr_hits << endl;
      cout << " Old 1/beta: " << invbeta << " +/- " << invbetaerr << endl;
      cout << " New nStations: " << nStations << " nHits: " << dstnc.size() << endl;
      cout << " Old nStations: " << (*mi).second.nStations << " nHits: " << (*mi).second.nHits << endl;
    }
          
    DriftTubeTOF tof;
    
    tof.invBeta=invbeta_hits;
    tof.invBetaErr=invbetaerr_hits;
    tof.invBetaFree=(*mi).second.invBetaFree;
    tof.invBetaFreeErr=(*mi).second.invBetaFreeErr;
    tof.vertexTime=(*mi).second.vertexTime;
    tof.vertexTimeErr=(*mi).second.vertexTimeErr;
    tof.nHits = dstnc.size();
    tof.nStations=nStations;

    outputCollection->setValue(muId,tof); 

  } // mi

  std::auto_ptr<MuonTOFCollection> res(outputCollection);
  iEvent.put(res);

  if (debug) 
    cout << "*** Refit beta from TOF done ***" << endl;

}


double
TOFBetaRefitter::fitT0(double &a, double &b, vector<double> xl, vector<double> yl, vector<double> xr, vector<double> yr ) {

  double sx=0,sy=0,sxy=0,sxx=0,ssx=0,ssy=0,s=0,ss=0;

  if ((xl.size()==0) || (xr.size()==0)) return 0;

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
TOFBetaRefitter::textplot(vector<double> x, vector <double> y, vector <double> side)
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
TOFBetaRefitter::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
TOFBetaRefitter::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(TOFBetaRefitter);
