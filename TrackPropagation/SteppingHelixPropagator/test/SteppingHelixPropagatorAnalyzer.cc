// -*- C++ -*-
//
// Package:    SteppingHelixPropagatorAnalyzer
// Class:      SteppingHelixPropagatorAnalyzer
// 
/**\class SteppingHelixPropagatorAnalyzer SteppingHelixPropagatorAnalyzer.cc TrackPropagation/SteppingHelixPropagator/src/SteppingHelixPropagatorAnalyzer.cc

Description: <one line class summary>

Implementation:
<Notes on implementation>
*/
//
// Original Author:  Vyacheslav Krutelyov
//         Created:  Fri Mar  3 16:01:24 CST 2006
// $Id$
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Utilities/Timing/interface/TimingReport.h"

#include "Geometry/Surface/interface/Cylinder.h"
#include "Geometry/Surface/interface/Plane.h"

#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
//#include "Geometry/RPCGeometry/interface/RPCRoll.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"


#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/Track/interface/EmbdSimTrack.h"
#include "SimDataFormats/Track/interface/EmbdSimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/EmbdSimVertex.h"
#include "SimDataFormats/Vertex/interface/EmbdSimVertexContainer.h"

#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"

#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Units/PhysicalConstants.h"
#include "CLHEP/Matrix/DiagMatrix.h"


#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagator.h"

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
  void beginJob(edm::EventSetup const&);

 protected:
  void loadNtVars(int ind, int eType,  int pStatus, int id,//defs offset: 0 for R, 1*3 for Z and, 2*3 for P
		  const Hep3Vector& p3, const Hep3Vector& r3, 
		  const Hep3Vector& p3R, const Hep3Vector& r3R, 
		  int charge, const HepSymMatrix& cov);

  FreeTrajectoryState getFromCLHEP(const Hep3Vector& p3, const Hep3Vector& r3, 
				    int charge, const HepSymMatrix& cov,
				    const MagneticField* field);
  void getFromFTS(const FreeTrajectoryState& fts,
		  Hep3Vector& p3, Hep3Vector& r3, 
		  int& charge, HepSymMatrix& cov);

 private:
// ----------member data ---------------------------
  SteppingHelixPropagator* ivProp_;
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

  bool doneMapping_;

  bool noMaterialMode_;
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
SteppingHelixPropagatorAnalyzer::SteppingHelixPropagatorAnalyzer(const edm::ParameterSet& iConfig)
{
  //now do what ever initialization is needed
  ivProp_ = 0;

  ntFile_ = new TFile(iConfig.getParameter<std::string>("NtFile").c_str(), "recreate");
  tr_ = new TTree("MuProp", "MuProp");
  tr_->Branch("nPoints", &nPoints_, "nPoints/I");
  tr_->Branch("q", q_, "q[nPoints]/I");
  tr_->Branch("pStatus", pStatus_, "pStatus[nPoints][3]/I");
  tr_->Branch("p3", p3_, "p3[nPoints][9]/F");
  tr_->Branch("r3", r3_, "r3[nPoints][9]/F");
  tr_->Branch("id", id_, "id[nPoints][3]/I");
  tr_->Branch("p3R", p3R_, "p3R[nPoints][3]/F");
  tr_->Branch("r3R", r3R_, "r3R[nPoints][3]/F");
  tr_->Branch("covFlat", covFlat_, "covFlat[nPoints][21]/F");
  tr_->Branch("run", &run_, "run/I");
  tr_->Branch("event_", &event_, "event/I");

  debug_ = iConfig.getParameter<bool>("debug");
  noMaterialMode_ = iConfig.getParameter<bool>("noMaterialMode");
  
}

void SteppingHelixPropagatorAnalyzer::beginJob(const edm::EventSetup& es){
}


SteppingHelixPropagatorAnalyzer::~SteppingHelixPropagatorAnalyzer()
{
 
  if (ivProp_) delete ivProp_;

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
SteppingHelixPropagatorAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  ESHandle<MagneticField> bField;
  iSetup.get<IdealMagneticFieldRecord>().get(bField);

  ESHandle<DTGeometry> dtGeomESH;
  iSetup.get<MuonGeometryRecord>().get(dtGeomESH);
  if (debug_){
    std::cout<<"Got DTGeometry "<<std::endl;
  }

  ESHandle<CSCGeometry> cscGeomESH;
  iSetup.get<MuonGeometryRecord>().get(cscGeomESH);
  if (debug_){
    std::cout<<"Got CSCGeometry "<<std::endl;
  }

  ESHandle<RPCGeometry> rpcGeomESH;
  iSetup.get<MuonGeometryRecord>().get(rpcGeomESH);
  if (debug_){
    std::cout<<"Got RPCGeometry "<<std::endl;
  }

  run_ = (int)iEvent.id().run();
  event_ = (int)iEvent.id().event();
  if (debug_){
    std::cout<<"Begin for run:event =="<<run_<<":"<<event_<<std::endl;
  }


  if (! ivProp_ ) ivProp_ = new SteppingHelixPropagator(&*bField);
  ivProp_->setDebug(debug_);

  ivProp_->setMaterialMode(noMaterialMode_);
  const double FPRP_MISMATCH = 150.;
  int pStatus = 0; //1 will be bad

  Handle<EmbdSimTrackContainer> simTracks;
  iEvent.getByType<EmbdSimTrackContainer>(simTracks);
  if (! simTracks.isValid() ){
    std::cout<<"No tracks found"<<std::endl;
    return;
  }
  if (debug_){
    std::cout<<"Got simTracks of size "<< simTracks->size()<<std::endl;
  }

  Handle<EmbdSimVertexContainer> simVertices;
  iEvent.getByType<EmbdSimVertexContainer>(simVertices);
  if (! simVertices.isValid() ){
    std::cout<<"No tracks found"<<std::endl;
    return;
  }
  if (debug_){
    std::cout<<"Got simVertices of size "<< simVertices->size()<<std::endl;
  }


  Handle<PSimHitContainer> simHitsDT;
  iEvent.getByLabel("SimG4Object", "MuonDTHits", simHitsDT);
  if (! simHitsDT.isValid() ){
    std::cout<<"No hits found"<<std::endl;
    return;
  }
  if (debug_){
    std::cout<<"Got MuonDTHits of size "<< simHitsDT->size()<<std::endl;
  }
  Handle<PSimHitContainer> simHitsCSC;
  iEvent.getByLabel("SimG4Object", "MuonCSCHits", simHitsCSC);
  if (! simHitsCSC.isValid() ){
    std::cout<<"No hits found"<<std::endl;
    return;
  }
  if (debug_){
    std::cout<<"Got MuonCSCHits of size "<< simHitsCSC->size()<<std::endl;
  }
  Handle<PSimHitContainer> simHitsRPC;
  iEvent.getByLabel("SimG4Object", "MuonRPCHits", simHitsRPC);
  if (! simHitsRPC.isValid() ){
    std::cout<<"No hits found"<<std::endl;
    return;
  }
  if (debug_){
    std::cout<<"Got MuonRPCHits of size "<< simHitsRPC->size()<<std::endl;
  }

  EmbdSimTrackContainer::const_iterator tracksCI = simTracks->begin();
  for(; tracksCI != simTracks->end(); tracksCI++){
    
    int trkPDG = tracksCI->type();
    if (abs(trkPDG) != 13 ) {
      if (debug_){
	std::cout<<"Skip "<<trkPDG<<std::endl;
      }
      continue;
    }
    Hep3Vector p3T = tracksCI->momentum().vect();
    if (p3T.mag()< 2.) continue;

    TimeMe tProp("SteppingHelixPropagatorAnalyzer::analyze::propagate");
    int vtxInd = tracksCI->vertIndex();
    uint trkInd = tracksCI->genpartIndex();
    Hep3Vector r3T(0.,0.,0.);
    if (vtxInd < 0){
      std::cout<<"Track with no vertex, defaulting to (0,0,0)"<<std::endl;      
    } else {
      r3T = (*simVertices)[vtxInd].position().vect()*0.1; //seems to be stored in mm --> convert to cm
    }
    HepSymMatrix covT(6,1); covT *= 1e-6; // initialize to sigma=1e-3

    Hep3Vector p3F,r3F; //propagated state
    Hep3Vector p3R,r3R; //reference (hit) state
    HepSymMatrix covF(6,0);
    int charge = trkPDG > 0 ? -1 : 1;

    nPoints_ = 0;
    pStatus = 0;
    loadNtVars(nPoints_, 0, pStatus, 0, p3T, r3T,  p3T, r3T, charge, covT); nPoints_++;
    FreeTrajectoryState ftsTrack = getFromCLHEP(p3T, r3T, charge, covT, &*bField);
    FreeTrajectoryState ftsStart = ftsTrack;
    TrajectoryStateOnSurface tSOSDest;

    PSimHitContainer::const_iterator muHitsDT_CI = simHitsDT->begin();
    for (; muHitsDT_CI != simHitsDT->end(); muHitsDT_CI++){
      if (muHitsDT_CI->trackId() != trkInd ) continue;
      DTWireId wId(muHitsDT_CI->detUnitId());
      const DTLayer* layer = dtGeomESH->layer(wId);
      if (layer == 0){
	std::cout<<"Failed to get detector unit"<<std::endl;
	continue;
      }
      const Surface& surf = layer->surface();
      GlobalPoint r3Hit = surf.toGlobal(muHitsDT_CI->localPosition());
      r3R.set(r3Hit.x(), r3Hit.y(), r3Hit.z());
      GlobalVector p3Hit = surf.toGlobal(muHitsDT_CI->momentumAtEntry());
      p3R.set(p3Hit.x(), p3Hit.y(), p3Hit.z());

      if (p3Hit.mag() < 0.5 ) continue;
      if (abs(muHitsDT_CI->particleType()) != 13) continue;
      if (debug_){
	std::cout<< wId
		 <<" r3L:"<<muHitsDT_CI->localPosition()
		 <<" r3G:"<<r3Hit
		 <<" p3L:"<<muHitsDT_CI->momentumAtEntry()
		 <<" p3G:"<<p3Hit
		 <<" pId:"<<muHitsDT_CI->particleType()
		 <<" tId:"<<muHitsDT_CI->trackId()
		 <<std::endl;
      }

      if (debug_){
	std::cout<<"Will propagate to surface: "<<surf.position()<<" "<<surf.rotation()<<std::endl;
      }
      tSOSDest = ivProp_->Propagator::propagate(ftsStart, surf);
      ftsStart = *tSOSDest.freeState();
      getFromFTS(ftsStart, p3F, r3F, charge, covF);

      pStatus = tSOSDest.isValid() ? 0 : 1;
      if ( (r3F-r3R).mag() > FPRP_MISMATCH){ 
	//start from the beginning if failed with previous
	ftsStart = ftsTrack;
	pStatus = 1;
      }

      if (debug_){
	std::cout<<"Got to "
		 <<" r3Prp:"<<r3F
		 <<" r3Hit:"<<r3R
		 <<" p3Prp:"<<p3F
		 <<" p3Hit:"<<p3R
		 <<std::endl;
      }
      loadNtVars(nPoints_, 0, pStatus, muHitsDT_CI->detUnitId(), p3F, r3F, p3R, r3R, charge, covF); nPoints_++;
    }


    PSimHitContainer::const_iterator muHitsRPC_CI = simHitsRPC->begin();
    for (; muHitsRPC_CI != simHitsRPC->end(); muHitsRPC_CI++){
      if (muHitsRPC_CI->trackId() != trkInd || 1< 2) continue; // no use, RPCs are not working yet for me
      if (debug_){
	std::cout<<" Doing RPC id "<<muHitsRPC_CI->detUnitId()<<std::endl;
      }
      //      RPCDetId wId; wId.buildfromTrIndex(muHitsRPC_CI->detUnitId());
      RPCDetId wId(muHitsRPC_CI->detUnitId());
      if (debug_){
	std::cout<<" Doing RPC id "<<wId<<std::endl;
      }
      const GeomDet* roll = rpcGeomESH->idToDet(wId);
      if (roll == 0){
	std::cout<<"Failed to get detector unit"<<std::endl;
	continue;
      }
      const Surface& surf = roll->surface();
      GlobalPoint r3Hit = surf.toGlobal(muHitsRPC_CI->localPosition());
      r3R.set(r3Hit.x(), r3Hit.y(), r3Hit.z());
      GlobalVector p3Hit = surf.toGlobal(muHitsRPC_CI->momentumAtEntry());
      p3R.set(p3Hit.x(), p3Hit.y(), p3Hit.z());

      if (p3Hit.mag() < 0.5 ) continue;
      if (abs(muHitsRPC_CI->particleType()) != 13) continue;
      if (debug_){
	std::cout<< wId
		 <<" r3L:"<<muHitsRPC_CI->localPosition()
		 <<" r3G:"<<r3Hit
		 <<" p3L:"<<muHitsRPC_CI->momentumAtEntry()
		 <<" p3G:"<<p3Hit
		 <<" pId:"<<muHitsRPC_CI->particleType()
		 <<" tId:"<<muHitsRPC_CI->trackId()
		 <<std::endl;
      }

      if (debug_){
	std::cout<<"Will propagate to surface:"<<surf.position()<<" "<<surf.rotation()<<std::endl;
      }
      tSOSDest = ivProp_->Propagator::propagate(ftsStart, surf);
      ftsStart = *tSOSDest.freeState();
      getFromFTS(ftsStart, p3F, r3F, charge, covF);

      pStatus = tSOSDest.isValid() ? 0 : 1;
      if ( (r3F-r3R).mag() > FPRP_MISMATCH){ 
	//start from the beginning if failed with previous
	ftsStart = ftsTrack;
	pStatus = 1;
      }

      if (debug_){
	std::cout<<"Got to "
		 <<" r3Prp:"<<r3F
		 <<" r3Hit:"<<r3R
		 <<" p3Prp:"<<p3F
		 <<" p3Hit:"<<p3R
		 <<std::endl;
      }
      loadNtVars(nPoints_, 0, pStatus, muHitsRPC_CI->detUnitId(), p3F, r3F, p3R, r3R, charge, covF); nPoints_++;
    }


    PSimHitContainer::const_iterator muHitsCSC_CI = simHitsCSC->begin();
    for (; muHitsCSC_CI != simHitsCSC->end(); muHitsCSC_CI++){
      if (muHitsCSC_CI->trackId() != trkInd ) continue;
      CSCDetId wId(muHitsCSC_CI->detUnitId());
      const GeomDet* layer = cscGeomESH->idToDet(wId);
      if (layer == 0){
	std::cout<<"Failed to get CSC detector unit"<<std::endl;
	continue;
      }
      const Surface& surf = layer->surface();
      GlobalPoint r3Hit = surf.toGlobal(muHitsCSC_CI->localPosition());
      r3R.set(r3Hit.x(), r3Hit.y(), r3Hit.z());
      GlobalVector p3Hit = surf.toGlobal(muHitsCSC_CI->momentumAtEntry());
      p3R.set(p3Hit.x(), p3Hit.y(), p3Hit.z());

      if (p3Hit.mag() < 0.5 ) continue;
      if (abs(muHitsCSC_CI->particleType()) != 13) continue;
      if (debug_){
	std::cout<< wId
		 <<" r3L:"<<muHitsCSC_CI->localPosition()
		 <<" r3G:"<<r3Hit
		 <<" p3L:"<<muHitsCSC_CI->momentumAtEntry()
		 <<" p3G:"<<p3Hit
		 <<" pId:"<<muHitsCSC_CI->particleType()
		 <<" tId:"<<muHitsCSC_CI->trackId()
		 <<std::endl;
      }

      if (debug_){
	std::cout<<"Will propagate to surface: "<<surf.position()<<" "<<surf.rotation()<<std::endl;
      }
      tSOSDest = ivProp_->Propagator::propagate(ftsStart, surf);
      ftsStart = *tSOSDest.freeState();
      getFromFTS(ftsStart, p3F, r3F, charge, covF);

      pStatus = tSOSDest.isValid() ? 0 : 1;
      if ( (r3F-r3R).mag() > FPRP_MISMATCH){ 
	//start from the beginning if failed with previous
	ftsStart = ftsTrack;
	pStatus = 1;
      }

      if (debug_){
	std::cout<<"Got to "
		 <<" r3Prp:"<<r3F
		 <<" r3Hit:"<<r3R
		 <<" p3Prp:"<<p3F
		 <<" p3Hit:"<<p3R
		 <<std::endl;
      }
      loadNtVars(nPoints_, 0, pStatus, muHitsCSC_CI->detUnitId(), p3F, r3F, p3R, r3R, charge, covF); nPoints_++;

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
  TimingReport::current()->dump(std::cout);
}

void SteppingHelixPropagatorAnalyzer::loadNtVars(int ind, int eType, int pStatus, int id,
			    const Hep3Vector& p3, const Hep3Vector& r3, 
			    const Hep3Vector& p3R, const Hep3Vector& r3R, 
			    int charge, const HepSymMatrix& cov){
      p3_[ind][eType*3+0] = p3.x();  p3_[ind][eType*3+1] = p3.y();  p3_[ind][eType*3+2] = p3.z();
      r3_[ind][eType*3+0] = r3.x();  r3_[ind][eType*3+1] = r3.y();  r3_[ind][eType*3+2] = r3.z();
      id_[ind] = id;
      p3R_[ind][0] = p3R.x();  p3R_[ind][1] = p3R.y();  p3R_[ind][2] = p3R.z();
      r3R_[ind][0] = r3R.x();  r3R_[ind][1] = r3R.y();  r3R_[ind][2] = r3R.z();
      int flatInd = 0;
      for (int i =1; i <= 6; i++) 
	for (int j=1; j<=i;j++){
	  covFlat_[ind][flatInd] = cov.fast(i,j);
	  flatInd++;
	}
      q_[ind] = charge;
      eType_[ind] = eType;
      pStatus_[ind][eType] = pStatus;

}

FreeTrajectoryState
SteppingHelixPropagatorAnalyzer::getFromCLHEP(const Hep3Vector& p3, const Hep3Vector& r3, 
					      int charge, const HepSymMatrix& cov,
					      const MagneticField* field){

  GlobalVector p3GV(p3.x(), p3.y(), p3.z());
  GlobalPoint r3GP(r3.x(), r3.y(), r3.z());
  GlobalTrajectoryParameters tPars(r3GP, p3GV, charge, field);

  CartesianTrajectoryError tCov(cov);
  
  return FreeTrajectoryState(tPars, tCov);
}

void SteppingHelixPropagatorAnalyzer::getFromFTS(const FreeTrajectoryState& fts,
						 Hep3Vector& p3, Hep3Vector& r3, 
						 int& charge, HepSymMatrix& cov){
  GlobalVector p3GV = fts.momentum();
  GlobalPoint r3GP = fts.position();

  p3.set(p3GV.x(), p3GV.y(), p3GV.z());
  r3.set(r3GP.x(), r3GP.y(), r3GP.z());
  
  charge = fts.charge();
  cov = fts.cartesianError().matrix();

}


//define this as a plug-in
DEFINE_FWK_MODULE(SteppingHelixPropagatorAnalyzer)
