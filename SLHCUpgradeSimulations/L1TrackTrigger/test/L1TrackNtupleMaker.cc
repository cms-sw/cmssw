//////////////////////////////////////////////////////////////////////
//                                                                  //
//  Analyzer for making mini-ntuple for L1 track performance plots  //
//                                                                  //
//////////////////////////////////////////////////////////////////////


////////////////////
// FRAMEWORK HEADERS
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

///////////////////////
// DATA FORMATS HEADERS
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "DataFormats/Common/interface/Ref.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h" 
#include "DataFormats/Common/interface/DetSetVector.h"

#include "SimDataFormats/SLHC/interface/StackedTrackerTypes.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/Vector3D.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/Candidate/interface/Candidate.h"

#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementPoint.h"
#include "TrackingTools/GeomPropagators/interface/HelixArbitraryPlaneCrossing.h"

////////////////////////////
// DETECTOR GEOMETRY HEADERS
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetType.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelTopologyBuilder.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/RectangularPixelTopology.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"

#include "Geometry/Records/interface/StackedTrackerGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/StackedTrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/StackedTrackerDetUnit.h"
#include "DataFormats/SiPixelDetId/interface/StackedTrackerDetId.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"


////////////////
// PHYSICS TOOLS
#include "CommonTools/UtilAlgos/interface/TFileService.h"

///////////////
// ROOT HEADERS
#include <TROOT.h>
#include <TCanvas.h>
#include <TTree.h>
#include <TFile.h>
#include <TF1.h>
#include <TH2F.h>
#include <TH1F.h>

//////////////
// STD HEADERS
#include <memory>
#include <string>
#include <iostream>

//////////////
// NAMESPACES
using namespace std;
using namespace edm;


//////////////////////////////
//                          //
//     CLASS DEFINITION     //
//                          //
//////////////////////////////

class L1TrackNtupleMaker : public edm::EDAnalyzer
{
public:

  // Constructor/destructor
  explicit L1TrackNtupleMaker(const edm::ParameterSet& iConfig);
  virtual ~L1TrackNtupleMaker();

  // Mandatory methods
  virtual void beginJob();
  virtual void endJob();
  virtual void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup);
  
protected:
  
private:
  
  // Containers of parameters passed by python configuration file
  edm::ParameterSet config; 
  

  //-----------------------------------------------------------------------------------------------
  // tree & branches for mini-ntuple

  TTree* eventTree;

  // basic track properties, filled for *all* tracks, regardless of type
  std::vector<float>* m_trk_pt;
  std::vector<float>* m_trk_eta;
  std::vector<float>* m_trk_phi;
  std::vector<float>* m_trk_z0;
  std::vector<int>*   m_trk_chi2;
  std::vector<int>*   m_trk_charge;
  std::vector<int>*   m_trk_nstub;

  // sim track properties, filled for *all* sim tracks
  std::vector<float>* m_simtrk_pt;
  std::vector<float>* m_simtrk_eta;
  std::vector<float>* m_simtrk_phi;
  std::vector<float>* m_simtrk_z0;
  std::vector<int>*   m_simtrk_id;
  std::vector<int>*   m_simtrk_type;

  // *sim track* properties, for sim tracks that are matched to a L1 track using simtrackID
  std::vector<float>* m_matchID_simtrk_pt;
  std::vector<float>* m_matchID_simtrk_eta;
  std::vector<float>* m_matchID_simtrk_phi;
  std::vector<float>* m_matchID_simtrk_z0;
  std::vector<int>*   m_matchID_simtrk_id;
  std::vector<int>*   m_matchID_simtrk_type;

  // *L1 track* properties, for sim tracks that are matched to an L1 track using simtrackID
  std::vector<float>* m_matchID_trk_pt;
  std::vector<float>* m_matchID_trk_eta;
  std::vector<float>* m_matchID_trk_phi;
  std::vector<float>* m_matchID_trk_z0;
  std::vector<int>*   m_matchID_trk_chi2; 
  std::vector<int>*   m_matchID_trk_charge;
  std::vector<int>*   m_matchID_trk_nstub;
  std::vector<int>*   m_matchID_trk_nmatch;

  // *sim track* properties, for sim tracks that are matched to an L1 track using dR<0.1
  std::vector<float>* m_matchDR_simtrk_pt;
  std::vector<float>* m_matchDR_simtrk_eta;
  std::vector<float>* m_matchDR_simtrk_phi;
  std::vector<float>* m_matchDR_simtrk_z0;
  std::vector<int>*   m_matchDR_simtrk_id;
  std::vector<int>*   m_matchDR_simtrk_type;

  // *L1 track* properties, for sim tracks that are matched to an L1 track using dR<0.1
  std::vector<float>* m_matchDR_trk_pt;
  std::vector<float>* m_matchDR_trk_eta;
  std::vector<float>* m_matchDR_trk_phi;
  std::vector<float>* m_matchDR_trk_z0;
  std::vector<int>*   m_matchDR_trk_chi2; 
  std::vector<int>*   m_matchDR_trk_charge;
  std::vector<int>*   m_matchDR_trk_nstub;
  std::vector<int>*   m_matchDR_trk_nmatch;


};


//////////////////////////////////
//                              //
//     CLASS IMPLEMENTATION     //
//                              //
//////////////////////////////////

//////////////
// CONSTRUCTOR
L1TrackNtupleMaker::L1TrackNtupleMaker(edm::ParameterSet const& iConfig) : 
  config(iConfig)
{
}

/////////////
// DESTRUCTOR
L1TrackNtupleMaker::~L1TrackNtupleMaker()
{
}  

//////////
// END JOB
void L1TrackNtupleMaker::endJob()
{
  // things to be done at the exit of the event Loop
  cerr << "L1TrackNtupleMaker::endJob" << endl;

}

////////////
// BEGIN JOB
void L1TrackNtupleMaker::beginJob()
{

  // things to be done before entering the event Loop
  cerr << "L1TrackNtupleMaker::beginJob" << endl;


  //-----------------------------------------------------------------------------------------------
  // book histograms / make ntuple
  edm::Service<TFileService> fs;


  // initilize
  m_trk_pt     = new std::vector<float>;
  m_trk_eta    = new std::vector<float>;
  m_trk_phi    = new std::vector<float>;
  m_trk_z0     = new std::vector<float>;
  m_trk_chi2   = new std::vector<int>;
  m_trk_charge = new std::vector<int>;
  m_trk_nstub  = new std::vector<int>;

  m_simtrk_pt   = new std::vector<float>;
  m_simtrk_eta  = new std::vector<float>;
  m_simtrk_phi  = new std::vector<float>;
  m_simtrk_z0   = new std::vector<float>;
  m_simtrk_id   = new std::vector<int>;
  m_simtrk_type = new std::vector<int>;

  m_matchID_simtrk_pt   = new std::vector<float>;
  m_matchID_simtrk_eta  = new std::vector<float>;
  m_matchID_simtrk_phi  = new std::vector<float>;
  m_matchID_simtrk_z0   = new std::vector<float>;
  m_matchID_simtrk_id   = new std::vector<int>;
  m_matchID_simtrk_type = new std::vector<int>;

  m_matchID_trk_pt     = new std::vector<float>;
  m_matchID_trk_eta    = new std::vector<float>;
  m_matchID_trk_phi    = new std::vector<float>;
  m_matchID_trk_z0     = new std::vector<float>;
  m_matchID_trk_chi2   = new std::vector<int>;
  m_matchID_trk_charge = new std::vector<int>;
  m_matchID_trk_nstub  = new std::vector<int>;
  m_matchID_trk_nmatch = new std::vector<int>;

  m_matchDR_simtrk_pt   = new std::vector<float>;
  m_matchDR_simtrk_eta  = new std::vector<float>;
  m_matchDR_simtrk_phi  = new std::vector<float>;
  m_matchDR_simtrk_z0   = new std::vector<float>;
  m_matchDR_simtrk_id   = new std::vector<int>;
  m_matchDR_simtrk_type = new std::vector<int>;

  m_matchDR_trk_pt     = new std::vector<float>;
  m_matchDR_trk_eta    = new std::vector<float>;
  m_matchDR_trk_phi    = new std::vector<float>;
  m_matchDR_trk_z0     = new std::vector<float>;
  m_matchDR_trk_chi2   = new std::vector<int>;
  m_matchDR_trk_charge = new std::vector<int>;
  m_matchDR_trk_nstub  = new std::vector<int>;
  m_matchDR_trk_nmatch = new std::vector<int>;


  // ntuple
  eventTree = fs->make<TTree>("eventTree", "Event tree");

  eventTree->Branch("trk_pt",    &m_trk_pt);
  eventTree->Branch("trk_eta",   &m_trk_eta);
  eventTree->Branch("trk_phi",   &m_trk_phi);
  eventTree->Branch("trk_z0",    &m_trk_z0);
  eventTree->Branch("trk_chi2",  &m_trk_chi2);
  eventTree->Branch("trk_charge",&m_trk_charge);
  eventTree->Branch("trk_nstub", &m_trk_nstub);

  eventTree->Branch("simtrk_pt",  &m_simtrk_pt);
  eventTree->Branch("simtrk_eta", &m_simtrk_eta);
  eventTree->Branch("simtrk_phi", &m_simtrk_phi);
  eventTree->Branch("simtrk_z0",  &m_simtrk_z0);
  eventTree->Branch("simtrk_id",  &m_simtrk_id);
  eventTree->Branch("simtrk_type",&m_simtrk_type);

  eventTree->Branch("matchID_simtrk_pt",  &m_matchID_simtrk_pt);
  eventTree->Branch("matchID_simtrk_eta", &m_matchID_simtrk_eta);
  eventTree->Branch("matchID_simtrk_phi", &m_matchID_simtrk_phi);
  eventTree->Branch("matchID_simtrk_z0",  &m_matchID_simtrk_z0);
  eventTree->Branch("matchID_simtrk_id",  &m_matchID_simtrk_id);
  eventTree->Branch("matchID_simtrk_type",&m_matchID_simtrk_type);

  eventTree->Branch("matchID_trk_pt",    &m_matchID_trk_pt);
  eventTree->Branch("matchID_trk_eta",   &m_matchID_trk_eta);
  eventTree->Branch("matchID_trk_phi",   &m_matchID_trk_phi);
  eventTree->Branch("matchID_trk_z0",    &m_matchID_trk_z0);
  eventTree->Branch("matchID_trk_chi2",  &m_matchID_trk_chi2);
  eventTree->Branch("matchID_trk_charge",&m_matchID_trk_charge);
  eventTree->Branch("matchID_trk_nstub", &m_matchID_trk_nstub);
  eventTree->Branch("matchID_trk_nmatch",&m_matchID_trk_nmatch);

  eventTree->Branch("matchDR_simtrk_pt",  &m_matchDR_simtrk_pt);
  eventTree->Branch("matchDR_simtrk_eta", &m_matchDR_simtrk_eta);
  eventTree->Branch("matchDR_simtrk_phi", &m_matchDR_simtrk_phi);
  eventTree->Branch("matchDR_simtrk_z0",  &m_matchDR_simtrk_z0);
  eventTree->Branch("matchDR_simtrk_id",  &m_matchDR_simtrk_id);
  eventTree->Branch("matchDR_simtrk_type",&m_matchDR_simtrk_type);

  eventTree->Branch("matchDR_trk_pt",    &m_matchDR_trk_pt);
  eventTree->Branch("matchDR_trk_eta",   &m_matchDR_trk_eta);
  eventTree->Branch("matchDR_trk_phi",   &m_matchDR_trk_phi);
  eventTree->Branch("matchDR_trk_z0",    &m_matchDR_trk_z0);
  eventTree->Branch("matchDR_trk_chi2",  &m_matchDR_trk_chi2);
  eventTree->Branch("matchDR_trk_charge",&m_matchDR_trk_charge);
  eventTree->Branch("matchDR_trk_nstub", &m_matchDR_trk_nstub);
  eventTree->Branch("matchDR_trk_nmatch",&m_matchDR_trk_nmatch);


}

//////////
// ANALYZE
void L1TrackNtupleMaker::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  cerr << "L1TrackNtupleMaker:  Start in analyze()" << endl;


  // clear variables
  m_trk_pt->clear();
  m_trk_eta->clear();
  m_trk_phi->clear();
  m_trk_z0->clear();
  m_trk_chi2->clear();
  m_trk_charge->clear();
  m_trk_nstub->clear();

  m_simtrk_pt->clear();
  m_simtrk_eta->clear();
  m_simtrk_phi->clear();
  m_simtrk_z0->clear();
  m_simtrk_id->clear();
  m_simtrk_type->clear();

  m_matchID_simtrk_pt->clear();
  m_matchID_simtrk_eta->clear();
  m_matchID_simtrk_phi->clear();
  m_matchID_simtrk_z0->clear();
  m_matchID_simtrk_id->clear();
  m_matchID_simtrk_type->clear();

  m_matchID_trk_pt->clear();
  m_matchID_trk_eta->clear();
  m_matchID_trk_phi->clear();
  m_matchID_trk_z0->clear();
  m_matchID_trk_chi2->clear();
  m_matchID_trk_charge->clear();
  m_matchID_trk_nstub->clear();
  m_matchID_trk_nmatch->clear();
  
  m_matchDR_simtrk_pt->clear();
  m_matchDR_simtrk_eta->clear();
  m_matchDR_simtrk_phi->clear();
  m_matchDR_simtrk_z0->clear();
  m_matchDR_simtrk_id->clear();
  m_matchDR_simtrk_type->clear();

  m_matchDR_trk_pt->clear();
  m_matchDR_trk_eta->clear();
  m_matchDR_trk_phi->clear();
  m_matchDR_trk_z0->clear();
  m_matchDR_trk_chi2->clear();
  m_matchDR_trk_charge->clear();
  m_matchDR_trk_nstub->clear();
  m_matchDR_trk_nstub->clear();



  //-----------------------------------------------------------------------------------------------
  // retrieve various containers
  //-----------------------------------------------------------------------------------------------

  // get sim tracks & sim vertices
  edm::Handle<edm::SimTrackContainer>   simTrackHandle;
  edm::Handle<edm::SimVertexContainer>  simVtxHandle;
  iEvent.getByLabel( "g4SimHits", simTrackHandle );
  iEvent.getByLabel( "g4SimHits", simVtxHandle );

  // get the L1 tracks
  edm::Handle<L1TkTrack_PixelDigi_Collection> L1TrackHandle;
  iEvent.getByLabel("L1Tracks", "Level1TkTracks", L1TrackHandle);



  // ----------------------------------------------------------------------------------------------
  // loop over L1 tracks
  // ----------------------------------------------------------------------------------------------

  L1TkTrack_PixelDigi_Collection::const_iterator iterL1Track;
  for (iterL1Track = L1TrackHandle->begin(); iterL1Track != L1TrackHandle->end(); ++iterL1Track) {
        
    float tmp_trk_pt  = iterL1Track->getMomentum().perp();
    float tmp_trk_eta = iterL1Track->getMomentum().eta();
    float tmp_trk_phi = iterL1Track->getMomentum().phi();
    float tmp_trk_z0  = iterL1Track->getVertex().z();
    float tmp_trk_chi2   = iterL1Track->getChi2RPhi();
    float tmp_trk_charge = iterL1Track->getCharge();

    if (tmp_trk_pt < 2.0) continue;

    m_trk_pt ->push_back(tmp_trk_pt);
    m_trk_eta->push_back(tmp_trk_eta);
    m_trk_phi->push_back(tmp_trk_phi);
    m_trk_z0 ->push_back(tmp_trk_z0);
    m_trk_chi2  ->push_back(tmp_trk_chi2);
    m_trk_charge->push_back(tmp_trk_charge);

    /*
    // ----------------------------------------------------------------------------------------------
    // find matching sim track
    unsigned int simtrackid = iterL1Track->getSimTrackId();

    SimTrackContainer::const_iterator iterSimTracks;
    for (iterSimTracks = simTrackHandle->begin(); iterSimTracks != simTrackHandle->end(); ++iterSimTracks) {  
      
      float tmp_simtrk_pt  = iterSimTracks->momentum().pt();
      float tmp_simtrk_eta = iterSimTracks->momentum().eta();
      float tmp_simtrk_phi = iterSimTracks->momentum().phi();
      unsigned int tmp_simtrk_id = iterSimTracks->trackId();

      if (tmp_simtrk_id == simtrackid) { 
	// this is a match...!!
      }

    }//end simtrk loop
    */


    // ----------------------------------------------------------------------------------------------
    // get pointers to stubs associated to the L1 track
    std::vector< edm::Ptr< L1TkStub_PixelDigi_ > > theStubs = iterL1Track->getStubPtrs();

    int tmp_trk_nstub = (int) theStubs.size();
    m_trk_nstub->push_back(tmp_trk_nstub);

    /*
    // loop over the stubs
    for (unsigned int i=0; i<(unsigned int)theStubs.size(); i++) {
      bool genuine = theStubs.at(i)->isGenuine();
      if (genuine) {
	bool combine = theStubs.at(i)->isCombinatoric();
	int type = theStubs.at(i)->findType();
	unsigned int simid = theStubs.at(i)->findSimTrackId();
      }
    }
    */

  } //end loop L1tracks



  // ----------------------------------------------------------------------------------------------
  // loop over sim tracks
  // ----------------------------------------------------------------------------------------------

  SimTrackContainer::const_iterator iterSimTracks;
  SimVertexContainer::const_iterator iterSimVtx;

  for (iterSimTracks = simTrackHandle->begin(); iterSimTracks != simTrackHandle->end(); ++iterSimTracks) {  
 
    float tmp_simtrk_pt  = iterSimTracks->momentum().pt();
    float tmp_simtrk_eta = iterSimTracks->momentum().eta();
    float tmp_simtrk_phi = iterSimTracks->momentum().phi();
    unsigned int tmp_simtrk_id = iterSimTracks->trackId();
    int tmp_simtrk_type = iterSimTracks->type();

    if (tmp_simtrk_pt < 2.0) continue;

    float matchID_trk_pt  = -9999;
    float matchID_trk_eta = -9999;
    float matchID_trk_phi = -9999;
    float matchID_trk_z0  = -9999;
    float matchID_trk_chi2   = -9999;
    float matchID_trk_charge = -9999;
    int   matchID_trk_nstub  = -9999;
    
    float matchDR_trk_pt  = -9999;
    float matchDR_trk_eta = -9999;
    float matchDR_trk_phi = -9999;
    float matchDR_trk_z0  = -9999;
    float matchDR_trk_chi2   = -9999;
    float matchDR_trk_charge = -9999;
    int   matchDR_trk_nstub  = -9999;

    int nMatchID = 0;
    int nMatchDR = 0;


    // ----------------------------------------------------------------------------------------------
    // get sim vertex
    int sim_vtxid = iterSimTracks->vertIndex();
    float tmp_simtrk_z0 = -999.9;
    if (sim_vtxid > -1) {
      const SimVertex& theSimVertex = (*simVtxHandle)[sim_vtxid];
      math::XYZTLorentzVectorD trkVtxPos = theSimVertex.position();
      tmp_simtrk_z0 = trkVtxPos.z();
    }
    else {
      cout << "MEBUG: warning, cannot acces sim vertex !?" << endl;
    }

    m_simtrk_pt  ->push_back(tmp_simtrk_pt);
    m_simtrk_eta ->push_back(tmp_simtrk_eta);
    m_simtrk_phi ->push_back(tmp_simtrk_phi);
    m_simtrk_z0  ->push_back(tmp_simtrk_z0);
    m_simtrk_id  ->push_back(tmp_simtrk_id);
    m_simtrk_type->push_back(tmp_simtrk_type);
    
    nMatchID = 0; //number of L1 track matches found for each sim track (should be ==1 for matchID, can be >1 for matchDR)
    nMatchDR = 0;


    // ----------------------------------------------------------------------------------------------
    // loop over L1 tracks for match
    L1TkTrack_PixelDigi_Collection::const_iterator iterL1Track;
    for (iterL1Track = L1TrackHandle->begin(); iterL1Track != L1TrackHandle->end(); ++iterL1Track) {
      
      float tmp_trk_pt  = iterL1Track->getMomentum().perp();
      float tmp_trk_eta = iterL1Track->getMomentum().eta();
      float tmp_trk_phi = iterL1Track->getMomentum().phi();
      float tmp_trk_z0  = iterL1Track->getVertex().z();
      float tmp_trk_chi2   = iterL1Track->getChi2RPhi();
      float tmp_trk_charge = iterL1Track->getCharge();
      unsigned int tmp_trk_simtrackid = iterL1Track->getSimTrackId();
      
      if (tmp_trk_pt < 2.0) continue;
      
      // get pointers to stubs associated to the L1 track
      std::vector< edm::Ptr< L1TkStub_PixelDigi_ > > theStubs = iterL1Track->getStubPtrs();
      int tmp_trk_nstub = (int) theStubs.size();

      
      // ----------------------------------------------------------------------------------------------
      // matching based on dR < 0.1
      float dR = sqrt( (tmp_trk_eta-tmp_simtrk_eta)*(tmp_trk_eta-tmp_simtrk_eta) + (tmp_trk_phi-tmp_simtrk_phi)*(tmp_trk_phi-tmp_simtrk_phi) );
      
      if (dR < 0.1) { //match
	nMatchDR++;
	if (tmp_trk_pt > matchDR_trk_pt) {
	  matchDR_trk_pt  = tmp_trk_pt;
	  matchDR_trk_eta = tmp_trk_eta;
	  matchDR_trk_phi = tmp_trk_phi;
	  matchDR_trk_z0  = tmp_trk_z0;
	  matchDR_trk_chi2   = tmp_trk_chi2;
	  matchDR_trk_charge = tmp_trk_charge;
	  matchDR_trk_nstub  = tmp_trk_nstub;
	}
      }// end if match dR<0.1
      

      // ----------------------------------------------------------------------------------------------
      // matching based on sim track ID
      if ( (tmp_simtrk_id == tmp_trk_simtrackid)) {
	nMatchID++;
	if (tmp_trk_pt > matchID_trk_pt) {
	  matchID_trk_pt  = tmp_trk_pt;
	  matchID_trk_eta = tmp_trk_eta;
	  matchID_trk_phi = tmp_trk_phi;
	  matchID_trk_z0  = tmp_trk_z0;
	  matchID_trk_chi2   = tmp_trk_chi2;
	  matchID_trk_charge = tmp_trk_charge;
	  matchID_trk_nstub  = tmp_trk_nstub;
	}
      }
      
      
    }// end loop L1 tracks


    if (matchID_trk_pt > -1) {
      m_matchID_simtrk_pt  ->push_back(tmp_simtrk_pt);
      m_matchID_simtrk_eta ->push_back(tmp_simtrk_eta);
      m_matchID_simtrk_phi ->push_back(tmp_simtrk_phi);
      m_matchID_simtrk_z0  ->push_back(tmp_simtrk_z0);
      m_matchID_simtrk_type->push_back((int)tmp_simtrk_type);
      m_matchID_simtrk_id  ->push_back(tmp_simtrk_id);
      
      m_matchID_trk_pt ->push_back(matchID_trk_pt);
      m_matchID_trk_eta->push_back(matchID_trk_eta);
      m_matchID_trk_phi->push_back(matchID_trk_phi);
      m_matchID_trk_z0 ->push_back(matchID_trk_z0);
      m_matchID_trk_chi2  ->push_back(matchID_trk_chi2);
      m_matchID_trk_charge->push_back(matchID_trk_charge);
      m_matchID_trk_nstub ->push_back(matchID_trk_nstub);
      m_matchID_trk_nmatch->push_back(nMatchID);
    }

    if (matchDR_trk_pt > -1) {
      m_matchDR_simtrk_pt  ->push_back(tmp_simtrk_pt);
      m_matchDR_simtrk_eta ->push_back(tmp_simtrk_eta);
      m_matchDR_simtrk_phi ->push_back(tmp_simtrk_phi);
      m_matchDR_simtrk_z0  ->push_back(tmp_simtrk_z0);
      m_matchDR_simtrk_id  ->push_back(tmp_simtrk_id);
      m_matchDR_simtrk_type->push_back((int)tmp_simtrk_type);
      
      m_matchDR_trk_pt ->push_back(matchDR_trk_pt);
      m_matchDR_trk_eta->push_back(matchDR_trk_eta);
      m_matchDR_trk_phi->push_back(matchDR_trk_phi);
      m_matchDR_trk_z0 ->push_back(matchDR_trk_z0);
      m_matchDR_trk_chi2  ->push_back(matchDR_trk_chi2);
      m_matchDR_trk_charge->push_back(matchDR_trk_charge);
      m_matchDR_trk_nstub ->push_back(matchDR_trk_nstub);
      m_matchDR_trk_nmatch->push_back(nMatchDR);
    }
    
    

  } //end loop simtracks
  
  eventTree->Fill();


} // end of analyze()


///////////////////////////
// DEFINE THIS AS A PLUG-IN
DEFINE_FWK_MODULE(L1TrackNtupleMaker);
