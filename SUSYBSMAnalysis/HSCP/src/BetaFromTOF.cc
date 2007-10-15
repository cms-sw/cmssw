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
// $Id: BetaFromTOF.cc,v 1.1 2007/10/11 15:14:20 ptraczyk Exp $
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
      edm::InputTag TKtrackTags_; 
      edm::InputTag STAtrackTags_; 
      edm::InputTag MuonTags_; 
      edm::InputTag DTSegmentTags_; 

  int thePartID;
  bool debug;

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
  TKtrackTags_(iConfig.getUntrackedParameter<edm::InputTag>("TKtracks")),
  STAtrackTags_(iConfig.getUntrackedParameter<edm::InputTag>("STAtracks")),
  MuonTags_(iConfig.getUntrackedParameter<edm::InputTag>("Muons")),
  DTSegmentTags_(iConfig.getUntrackedParameter<edm::InputTag>("DTsegments")),
  debug(iConfig.getParameter<bool>("debug"))
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

//  iEvent.getByLabel( TKtrackTags_, TKTrackCollection);
//  const reco::TrackCollection tkTC = *(TKTrackCollection.product());

  iEvent.getByLabel( STAtrackTags_, STATrackCollection);
  const reco::TrackCollection staTC = *(STATrackCollection.product());

  iEvent.getByLabel( MuonTags_, GLBTrackCollection);
  const reco::TrackCollection glbTC = *(GLBTrackCollection.product());
  
  // Get the DT-Segment collection from the Event
  edm::Handle<DTRecSegment4DCollection> dtRecHits;
  iEvent.getByLabel(DTSegmentTags_, dtRecHits);  

  if (debug) 
    cout << "DT Segment collection size: " << dtRecHits->size() << endl;

  ESHandle<GlobalTrackingGeometry> theTrackingGeometry;
  iSetup.get<GlobalTrackingGeometryRecord>().get(theTrackingGeometry);

  ESHandle<DTGeometry> theDTGeometry;
  iSetup.get<MuonGeometryRecord>().get(theDTGeometry);

  vector<float> *outputCollection = new vector<float>;

  cout << " *** Analyzing HSCP *** " << endl;

//  for (TrackCollection::const_iterator candTrack = glbTC.begin(); candTrack != glbTC.end(); ++candTrack) {
  for (TrackCollection::const_iterator candTrack = staTC.begin(); candTrack != staTC.end(); ++candTrack) {

    segmVect dtSegments;

    cout << " STA Track:   RecHits: " << (*candTrack).recHitsSize() << " momentum: " << (*candTrack).p() << endl;

    for (trackingRecHit_iterator hi=(*candTrack).recHitsBegin(); hi!=(*candTrack).recHitsEnd(); hi++)

      if (( (*hi)->geographicalId().subdetId() == MuonSubdetId::DT ) && ((*hi)->geographicalId().det() == 2)) {
  
        cout << "Hit dim: " << (*hi)->dimension();

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
            double t0, dist, invbeta;
	    bool matched = rechit->hasPhi() && rechit->hasZed();

	    if (rechit->hasPhi()) {
	      const DTRecSegment2D* si = dynamic_cast<const DTRecSegment2D*>(rechit->phiSegment());
              const GeomDet* geomDet = theTrackingGeometry->idToDet(si->geographicalId());
              dist = geomDet->toGlobal(si->localPosition()).mag();
	      t0 = si->t0();
	      
	      invbeta=1.+t0/dist*30.;
	      cout << " Station " << station << "   Phi 1/beta = " << invbeta << endl;
	    }
	    
	    if (rechit->hasZed()) {
	      const DTRecSegment2D* si = dynamic_cast<const DTRecSegment2D*>(rechit->zSegment());
              const GeomDet* geomDet = theTrackingGeometry->idToDet(si->geographicalId());
              dist = geomDet->toGlobal(si->localPosition()).mag();
	      t0 = si->t0();

	      invbeta=1.+t0/dist*30.;
	      cout << " Station " << station << " Theta 1/beta = " << invbeta << endl;
	    }  
          }
        } // rechit
    } // hi
    
    
  }  //candTrack


  std::auto_ptr<vector<float> > estimator(outputCollection);
  iEvent.put(estimator);

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
