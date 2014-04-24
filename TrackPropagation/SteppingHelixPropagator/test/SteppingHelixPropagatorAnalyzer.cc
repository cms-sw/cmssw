// -*- C++ -*-
//
// Package:    TrackPropagation/SteppingHelixPropagator
// Class:      SteppingHelixPropagatorAnalyzer
// 
/**\class SteppingHelixPropagatorAnalyzer 
   Description: Analyzer of SteppingHelixPropagator performance

   Implementation:
   Use simTracks and simVertices as initial points. For all  muon PSimHits in the event 
   extrapolate/propagate from the previous point (starting from a vertex) 
   to the hit position (detector surface).
   Fill an nTuple (could've been an EventProduct) with expected (given by the propagator) 
   and actual (PSimHits)
   positions of a muon in the detector.
*/
//
// Original Author:  Vyacheslav Krutelyov
//         Created:  Fri Mar  3 16:01:24 CST 2006
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/GeometrySurface/interface/Cylinder.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"

#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"

#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
//#include "Geometry/RPCGeometry/interface/RPCRoll.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"


#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"

#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
//#include "CLHEP/Matrix/DiagMatrix.h"

#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagator.h"
#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixStateInfo.h"

#include "TFile.h"
#include "TTree.h"

#include <map>


//
// class decleration
//

class SteppingHelixPropagatorAnalyzer : public edm::EDAnalyzer {
public:
  explicit SteppingHelixPropagatorAnalyzer(const edm::ParameterSet&);
  ~SteppingHelixPropagatorAnalyzer();


  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob();
  void beginJob();

protected:
  struct GlobalSimHit {
    const PSimHit* hit;
    const Surface* surf;
    DetId id;
    CLHEP::Hep3Vector r3;
    CLHEP::Hep3Vector p3;
  };


  void loadNtVars(int ind, int eType,  int pStatus, 
		  int id,//defs offset: 0 for R, 1*3 for Z and, 2*3 for P
		  const CLHEP::Hep3Vector& p3, const CLHEP::Hep3Vector& r3, 
		  const CLHEP::Hep3Vector& p3R, const CLHEP::Hep3Vector& r3R, 
		  int charge, const AlgebraicSymMatrix66& cov);

  FreeTrajectoryState getFromCLHEP(const CLHEP::Hep3Vector& p3, const CLHEP::Hep3Vector& r3, 
				   int charge, const AlgebraicSymMatrix66& cov,
				   const MagneticField* field);
  void getFromFTS(const FreeTrajectoryState& fts,
		  CLHEP::Hep3Vector& p3, CLHEP::Hep3Vector& r3, 
		  int& charge, AlgebraicSymMatrix66& cov);

  void addPSimHits(const edm::Event& iEvent,
		   const std::string instanceName, 
		   const edm::ESHandle<GlobalTrackingGeometry>& geom,
		   std::vector<SteppingHelixPropagatorAnalyzer::GlobalSimHit>& hits) const;

private:
  // ----------member data ---------------------------
  TFile* ntFile_;
  TTree* tr_;

  int nPoints_;
  int q_[1000];
  int eType_[1000];
  int pStatus_[1000][3];
  float p3_[1000][9];
  float r3_[1000][9];
  int id_[1000];
  float p3R_[1000][3];
  float r3R_[1000][3];
  float covFlat_[1000][21];

  bool debug_;
  int run_;
  int event_;

  int trkIndOffset_;

  bool doneMapping_;

  bool radX0CorrectionMode_;

  bool testPCAPropagation_;

  bool ntupleTkHits_;

  bool startFromPrevHit_;

  std::string g4SimName_;
  edm::InputTag simTracksTag_;
  edm::InputTag simVertexesTag_;
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
SteppingHelixPropagatorAnalyzer::SteppingHelixPropagatorAnalyzer(const edm::ParameterSet& iConfig) :
  simTracksTag_(iConfig.getParameter<edm::InputTag>("simTracksTag")),
  simVertexesTag_(iConfig.getParameter<edm::InputTag>("simVertexesTag"))
{
  //now do what ever initialization is needed

  ntFile_ = new TFile(iConfig.getParameter<std::string>("NtFile").c_str(), "recreate");
  tr_ = new TTree("MuProp", "MuProp");
  tr_->Branch("nPoints", &nPoints_, "nPoints/I");
  tr_->Branch("q", q_, "q[nPoints]/I");
  tr_->Branch("pStatus", pStatus_, "pStatus[nPoints][3]/I");
  tr_->Branch("p3", p3_, "p3[nPoints][9]/F");
  tr_->Branch("r3", r3_, "r3[nPoints][9]/F");
  tr_->Branch("id", id_, "id[nPoints]/I");
  tr_->Branch("p3R", p3R_, "p3R[nPoints][3]/F");
  tr_->Branch("r3R", r3R_, "r3R[nPoints][3]/F");
  tr_->Branch("covFlat", covFlat_, "covFlat[nPoints][21]/F");
  tr_->Branch("run", &run_, "run/I");
  tr_->Branch("event_", &event_, "event/I");

  trkIndOffset_ = iConfig.getParameter<int>("trkIndOffset");
  debug_ = iConfig.getParameter<bool>("debug");
  radX0CorrectionMode_ = iConfig.getParameter<bool>("radX0CorrectionMode");

  testPCAPropagation_ = iConfig.getParameter<bool>("testPCAPropagation");

  ntupleTkHits_ = iConfig.getParameter<bool>("ntupleTkHits");

  startFromPrevHit_ = iConfig.getParameter<bool>("startFromPrevHit");

  g4SimName_ = iConfig.getParameter<std::string>("g4SimName");


  //initialize the variables
  for(int i=0;i<1000;++i){
    q_[i] = 0;
    eType_[i] = 0;
    for(int j=0;j<3;++j) pStatus_[i][j] = 0;
    for(int j=0;j<9;++j) p3_[i][j] = 0;
    for(int j=0;j<9;++j) r3_[i][j] = 0;
    id_[i] = 0;
    for(int j=0;j<3;++j)p3R_[i][j] = 0;
    for(int j=0;j<3;++j) r3R_[i][j] = 0;
    for(int j=0;j<21;++j) covFlat_[i][j] = 0;
  }

}

void SteppingHelixPropagatorAnalyzer::beginJob(){
}


SteppingHelixPropagatorAnalyzer::~SteppingHelixPropagatorAnalyzer()
{
 
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
SteppingHelixPropagatorAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  constexpr char const*metname = "SteppingHelixPropagatorAnalyzer";
  (void)metname;
  using namespace edm;
  ESHandle<MagneticField> bField;
  iSetup.get<IdealMagneticFieldRecord>().get(bField);

  ESHandle<Propagator> shProp;
  iSetup.get<TrackingComponentsRecord>().get("SteppingHelixPropagatorAny", shProp);
  ESHandle<Propagator> shPropAny;
  iSetup.get<TrackingComponentsRecord>().get("SteppingHelixPropagatorAny", shPropAny);

  ESHandle<GlobalTrackingGeometry> geomESH;
  iSetup.get<GlobalTrackingGeometryRecord>().get(geomESH);
  if (debug_){
    LogTrace(metname)<<"Got GlobalTrackingGeometry "<<std::endl;
  }

  //   ESHandle<CSCGeometry> cscGeomESH;
  //   iSetup.get<MuonGeometryRecord>().get(cscGeomESH);
  //   if (debug_){
  //     std::cout<<"Got CSCGeometry "<<std::endl;
  //   }

  //   ESHandle<RPCGeometry> rpcGeomESH;
  //   iSetup.get<MuonGeometryRecord>().get(rpcGeomESH);
  //   if (debug_){
  //     std::cout<<"Got RPCGeometry "<<std::endl;
  //   }

  run_ = (int)iEvent.id().run();
  event_ = (int)iEvent.id().event();
  if (debug_){
    LogTrace(metname)<<"Begin for run:event =="<<run_<<":"<<event_<<std::endl;
  }


  const double FPRP_MISMATCH = 150.;
  int pStatus = 0; //1 will be bad

  Handle<SimTrackContainer> simTracks;
  iEvent.getByLabel<SimTrackContainer>(simTracksTag_, simTracks);
  if (! simTracks.isValid() ){
    std::cout<<"No tracks found"<<std::endl;
    return;
  }
  if (debug_){
    LogTrace(metname)<<"Got simTracks of size "<< simTracks->size()<<std::endl;
  }

  Handle<SimVertexContainer> simVertices;
  iEvent.getByLabel<SimVertexContainer>(simVertexesTag_, simVertices);
  if (! simVertices.isValid() ){
    std::cout<<"No tracks found"<<std::endl;
    return;
  }
  if (debug_){
    LogTrace(metname)<<"Got simVertices of size "<< simVertices->size()<<std::endl;
  }


  std::vector<GlobalSimHit> allSimHits; allSimHits.clear();

  addPSimHits(iEvent, "MuonDTHits", geomESH, allSimHits);
  addPSimHits(iEvent, "MuonCSCHits", geomESH, allSimHits);
  addPSimHits(iEvent, "MuonRPCHits", geomESH, allSimHits);
  addPSimHits(iEvent, "TrackerHitsPixelBarrelLowTof", geomESH, allSimHits);
  addPSimHits(iEvent, "TrackerHitsPixelEndcapLowTof", geomESH, allSimHits);
  addPSimHits(iEvent, "TrackerHitsTIBLowTof", geomESH, allSimHits);
  addPSimHits(iEvent, "TrackerHitsTIDLowTof", geomESH, allSimHits);
  addPSimHits(iEvent, "TrackerHitsTOBLowTof", geomESH, allSimHits);
  addPSimHits(iEvent, "TrackerHitsTECLowTof", geomESH, allSimHits);



  SimTrackContainer::const_iterator tracksCI = simTracks->begin();
  for(; tracksCI != simTracks->end(); tracksCI++){
    
    int trkPDG = tracksCI->type();
    if (abs(trkPDG) != 13 ) {
      if (debug_){
	LogTrace(metname)<<"Skip "<<trkPDG<<std::endl;
      }
      continue;
    }
    CLHEP::Hep3Vector p3T(tracksCI->momentum().x(), tracksCI->momentum().y(), tracksCI->momentum().z());
    if (p3T.mag()< 2.) continue;

    int vtxInd = tracksCI->vertIndex();
    unsigned int trkInd = tracksCI->genpartIndex() - trkIndOffset_;
    CLHEP::Hep3Vector r3T(0.,0.,0.);
    if (vtxInd < 0){
      std::cout<<"Track with no vertex, defaulting to (0,0,0)"<<std::endl;      
    } else {
      r3T = CLHEP::Hep3Vector((*simVertices)[vtxInd].position().x(),
		       (*simVertices)[vtxInd].position().y(),
		       (*simVertices)[vtxInd].position().z());

    }
    AlgebraicSymMatrix66 covT = AlgebraicMatrixID(); 
    covT *= 1e-20; // initialize to sigma=1e-10 .. should get overwhelmed by MULS

    CLHEP::Hep3Vector p3F,r3F; //propagated state
    AlgebraicSymMatrix66 covF;
    int charge = trkPDG > 0 ? -1 : 1; //works for muons

    nPoints_ = 0;
    pStatus = 0;
    loadNtVars(nPoints_, 0, pStatus, 0, p3T, r3T,  p3T, r3T, charge, covT); nPoints_++;
    FreeTrajectoryState ftsTrack = getFromCLHEP(p3T, r3T, charge, covT, &*bField);
    FreeTrajectoryState ftsStart = ftsTrack;
    SteppingHelixStateInfo siStart(ftsStart);
    SteppingHelixStateInfo siDest;
    TrajectoryStateOnSurface tSOSDest;

    std::map<double, const GlobalSimHit*> simHitsByDistance;
    std::map<double, const GlobalSimHit*> simHitsByTof;
    for (std::vector<GlobalSimHit>::const_iterator allHitsCI = allSimHits.begin();
	 allHitsCI != allSimHits.end(); allHitsCI++){
      if (allHitsCI->hit->trackId() != trkInd ) continue;
      if (abs(allHitsCI->hit->particleType()) != 13) continue;
      if (allHitsCI->p3.mag() < 0.5 ) continue;

      double distance = (allHitsCI->r3 - r3T).mag();
      double tof = allHitsCI->hit->timeOfFlight();
      simHitsByDistance[distance] = &*allHitsCI;
      simHitsByTof[tof] = &*allHitsCI;
    }

    {//new scope for timing purposes only
      if (testPCAPropagation_){
	FreeTrajectoryState ftsDest;
	GlobalPoint pDest1(10., 10., 0.);
	GlobalPoint pDest2(10., 10., 10.);
	const SteppingHelixPropagator* shPropAnyCPtr = 
	  dynamic_cast<const SteppingHelixPropagator*>(&*shPropAny);

	ftsDest = shPropAnyCPtr->propagate(ftsStart, pDest1);
	std::cout<<"----------------------------------------------"<<std::endl;
	ftsDest = shPropAnyCPtr->propagate(ftsStart, pDest1, pDest2);
	std::cout<<"----------------------------------------------"<<std::endl;
      }

      //now we are supposed to have a sorted list of hits
      std::map<double, const GlobalSimHit*>::const_iterator simHitsCI 
	= simHitsByDistance.begin();
      for (; simHitsCI != simHitsByDistance.end(); simHitsCI++){
	const GlobalSimHit* igHit = simHitsCI->second;
	const PSimHit* iHit = simHitsCI->second->hit;

	if (debug_){
	  LogTrace(metname)<< igHit->id.rawId()
			   <<" r3L:"<<iHit->localPosition()
			   <<" r3G:"<<igHit->r3
			   <<" p3L:"<<iHit->momentumAtEntry()
			   <<" p3G:"<<igHit->p3
			   <<" pId:"<<iHit->particleType()
			   <<" tId:"<<iHit->trackId()
			   <<std::endl;
	}

	if (debug_){
	  LogTrace(metname)<<"Will propagate to surface: "
			   <<igHit->surf->position()
			   <<" "<<igHit->surf->rotation()<<std::endl;
	}
	pStatus = 0;
	if (startFromPrevHit_){
	  if (simHitsCI != simHitsByDistance.begin()){
	    std::map<double, const GlobalSimHit*>::const_iterator simHitPrevCI = simHitsCI;
	    simHitPrevCI--;
	    const GlobalSimHit* gpHit = simHitPrevCI->second;
	    ftsStart = getFromCLHEP(gpHit->p3, gpHit->r3, charge, covT, &*bField);
	    siStart = SteppingHelixStateInfo(ftsTrack);
	  }
	}
	
	if (radX0CorrectionMode_ ){
	  const SteppingHelixPropagator* shPropCPtr = 
	    dynamic_cast<const SteppingHelixPropagator*>(&*shProp);
	  shPropCPtr->propagate(siStart, *igHit->surf,siDest);
	  if (siDest.isValid()){
	    siStart = siDest;
	    siStart.getFreeState(ftsStart);
	    getFromFTS(ftsStart, p3F, r3F, charge, covF);
	    pStatus = 0;
	  } else pStatus = 1;
	  if ( pStatus == 1 || (r3F- igHit->r3).mag() > FPRP_MISMATCH){ 
	    //start from the beginning if failed with previous
	    siStart = SteppingHelixStateInfo(ftsTrack);
	    pStatus = 1;
	  }
	} else {
	  tSOSDest = shProp->propagate(ftsStart, *igHit->surf);
	  if (tSOSDest.isValid()){
	    ftsStart = *tSOSDest.freeState();
	    getFromFTS(ftsStart, p3F, r3F, charge, covF);
	    pStatus = 0;
	  } else pStatus = 1;
	  if ( pStatus == 1 || (r3F- igHit->r3).mag() > FPRP_MISMATCH){ 
	    //start from the beginning if failed with previous
	    ftsStart = ftsTrack;
	    pStatus = 1;
	  }
	}
	

	if (debug_){
	  LogTrace(metname)<<"Got to "
			   <<" r3Prp:"<<r3F
			   <<" r3Hit:"<<igHit->r3
			   <<" p3Prp:"<<p3F
			   <<" p3Hit:"<<igHit->p3
			   <<" pPrp:"<<p3F.mag()
			   <<" pHit:"<<igHit->p3.mag()
			   <<std::endl;
	}
	loadNtVars(nPoints_, 0, pStatus, igHit->id.rawId(), 
		   p3F, r3F, igHit->p3, igHit->r3, charge, covF); nPoints_++;
      }

    }
    if (tr_) tr_->Fill(); //fill this track prop info
  }
    
  
}

// "endJob" is an inherited method that you may implement to do post-EOF processing
// and produce final output.
//
void SteppingHelixPropagatorAnalyzer::endJob() {
  ntFile_->cd();
  tr_->Write();
  delete ntFile_; ntFile_ = 0;
}

void SteppingHelixPropagatorAnalyzer::loadNtVars(int ind, int eType, int pStatus, int id,
						 const CLHEP::Hep3Vector& p3, const CLHEP::Hep3Vector& r3, 
						 const CLHEP::Hep3Vector& p3R, const CLHEP::Hep3Vector& r3R, 
						 int charge, const AlgebraicSymMatrix66& cov){
  p3_[ind][eType*3+0] = p3.x();  p3_[ind][eType*3+1] = p3.y();  p3_[ind][eType*3+2] = p3.z();
  r3_[ind][eType*3+0] = r3.x();  r3_[ind][eType*3+1] = r3.y();  r3_[ind][eType*3+2] = r3.z();
  id_[ind] = id;
  p3R_[ind][0] = p3R.x();  p3R_[ind][1] = p3R.y();  p3R_[ind][2] = p3R.z();
  r3R_[ind][0] = r3R.x();  r3R_[ind][1] = r3R.y();  r3R_[ind][2] = r3R.z();
  int flatInd = 0;
  for (int i =1; i <= cov.kRows; i++) 
    for (int j=1; j<=i;j++){
      covFlat_[ind][flatInd] = cov(i-1,j-1);
      flatInd++;
    }
  q_[ind] = charge;
  eType_[ind] = eType;
  pStatus_[ind][eType] = pStatus;

}

FreeTrajectoryState
SteppingHelixPropagatorAnalyzer::getFromCLHEP(const CLHEP::Hep3Vector& p3, const CLHEP::Hep3Vector& r3, 
					      int charge, const AlgebraicSymMatrix66& cov,
					      const MagneticField* field){

  GlobalVector p3GV(p3.x(), p3.y(), p3.z());
  GlobalPoint r3GP(r3.x(), r3.y(), r3.z());
  GlobalTrajectoryParameters tPars(r3GP, p3GV, charge, field);

  CartesianTrajectoryError tCov(cov);
  
  return cov.kRows == 6 ? FreeTrajectoryState(tPars, tCov) : FreeTrajectoryState(tPars) ;
}

void SteppingHelixPropagatorAnalyzer::getFromFTS(const FreeTrajectoryState& fts,
						 CLHEP::Hep3Vector& p3, CLHEP::Hep3Vector& r3, 
						 int& charge, AlgebraicSymMatrix66& cov){
  GlobalVector p3GV = fts.momentum();
  GlobalPoint r3GP = fts.position();

  p3.set(p3GV.x(), p3GV.y(), p3GV.z());
  r3.set(r3GP.x(), r3GP.y(), r3GP.z());
  
  charge = fts.charge();
  cov = fts.hasError() ? fts.cartesianError().matrix() : AlgebraicSymMatrix66();

}

void SteppingHelixPropagatorAnalyzer
::addPSimHits(const edm::Event& iEvent,
	      const std::string instanceName, 
	      const edm::ESHandle<GlobalTrackingGeometry>& geom,
	      std::vector<SteppingHelixPropagatorAnalyzer::GlobalSimHit>& hits) const {
  static std::string metname = "SteppingHelixPropagatorAnalyzer";
  edm::Handle<edm::PSimHitContainer> handle;
  iEvent.getByLabel(g4SimName_, instanceName, handle);
  if (! handle.isValid() ){
    std::cout<<"No hits found"<<std::endl;
    return;
  }
  if (debug_){
    LogTrace(metname)<<"Got "<<instanceName<<" of size "<< handle->size()<<std::endl;
  }  

  edm::PSimHitContainer::const_iterator pHits_CI = handle->begin();
  for (; pHits_CI != handle->end(); pHits_CI++){
    int dtId = pHits_CI->detUnitId(); 
    DetId wId(dtId);

    const GeomDet* layer = geom->idToDet(wId);

    GlobalSimHit gHit;
    gHit.hit = &*pHits_CI;
    gHit.surf = &layer->surface();
    gHit.id = wId;

    //place the hit onto the surface
    float dzFrac = gHit.hit->entryPoint().z()/(gHit.hit->exitPoint().z()-gHit.hit->entryPoint().z());
    GlobalPoint r3Hit = gHit.surf->toGlobal(gHit.hit->entryPoint()-dzFrac*(gHit.hit->exitPoint()-gHit.hit->entryPoint()));
    gHit.r3.set(r3Hit.x(), r3Hit.y(), r3Hit.z());
    GlobalVector p3Hit = gHit.surf->toGlobal(gHit.hit->momentumAtEntry());
    gHit.p3.set(p3Hit.x(), p3Hit.y(), p3Hit.z());
    hits.push_back(gHit);
  }
}


//define this as a plug-in
DEFINE_FWK_MODULE(SteppingHelixPropagatorAnalyzer);
