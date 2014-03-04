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

#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/L1TrackTrigger/interface/TTCluster.h"
#include "DataFormats/L1TrackTrigger/interface/TTStub.h"
#include "DataFormats/L1TrackTrigger/interface/TTTrack.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertex.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTClusterAssociationMap.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTStubAssociationMap.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTTrackAssociationMap.h"
#include "Geometry/Records/interface/StackedTrackerGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/StackedTrackerGeometry.h"

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
  
  int MyProcess;
  bool DebugMode;


  //-----------------------------------------------------------------------------------------------
  // tree & branches for mini-ntuple

  TTree* eventTree;

  // all L1 tracks
  std::vector<float>* m_trk_pt;
  std::vector<float>* m_trk_eta;
  std::vector<float>* m_trk_phi;
  std::vector<float>* m_trk_z0;
  std::vector<float>* m_trk_chi2; 
  std::vector<int>*   m_trk_nstub;
  std::vector<int>*   m_trk_genuine;
  std::vector<int>*   m_trk_unknown;
  std::vector<int>*   m_trk_combinatoric;

  // all tracking particles
  std::vector<float>* m_tp_pt;
  std::vector<float>* m_tp_eta;
  std::vector<float>* m_tp_phi;
  std::vector<float>* m_tp_z0;
  std::vector<int>*   m_tp_pdgid;
  std::vector<int>*   m_tp_nmatch;

  // *L1 track* properties if m_tp_nmatch > 0
  std::vector<float>* m_matchtrk_pt;
  std::vector<float>* m_matchtrk_eta;
  std::vector<float>* m_matchtrk_phi;
  std::vector<float>* m_matchtrk_z0;
  std::vector<float>* m_matchtrk_chi2; 
  std::vector<int>*   m_matchtrk_nstub;
  std::vector<int>*   m_matchtrk_genuine;

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

  MyProcess = iConfig.getParameter< int >("MyProcess");
  DebugMode = iConfig.getParameter< bool >("DebugMode");

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
  m_trk_pt    = new std::vector<float>;
  m_trk_eta   = new std::vector<float>;
  m_trk_phi   = new std::vector<float>;
  m_trk_z0    = new std::vector<float>;
  m_trk_chi2  = new std::vector<float>;
  m_trk_nstub = new std::vector<int>;
  m_trk_genuine      = new std::vector<int>;
  m_trk_unknown      = new std::vector<int>;
  m_trk_combinatoric = new std::vector<int>;

  m_tp_pt     = new std::vector<float>;
  m_tp_eta    = new std::vector<float>;
  m_tp_phi    = new std::vector<float>;
  m_tp_z0     = new std::vector<float>;
  m_tp_pdgid  = new std::vector<int>;
  m_tp_nmatch = new std::vector<int>;

  m_matchtrk_pt    = new std::vector<float>;
  m_matchtrk_eta   = new std::vector<float>;
  m_matchtrk_phi   = new std::vector<float>;
  m_matchtrk_z0    = new std::vector<float>;
  m_matchtrk_chi2  = new std::vector<float>;
  m_matchtrk_nstub = new std::vector<int>;
  m_matchtrk_genuine = new std::vector<int>;


  // ntuple
  eventTree = fs->make<TTree>("eventTree", "Event tree");

  eventTree->Branch("trk_pt",    &m_trk_pt);
  eventTree->Branch("trk_eta",   &m_trk_eta);
  eventTree->Branch("trk_phi",   &m_trk_phi);
  eventTree->Branch("trk_z0",    &m_trk_z0);
  eventTree->Branch("trk_chi2",  &m_trk_chi2);
  eventTree->Branch("trk_nstub", &m_trk_nstub);
  eventTree->Branch("trk_genuine",      &m_trk_genuine);
  eventTree->Branch("trk_unknown",      &m_trk_unknown);
  eventTree->Branch("trk_combinatoric", &m_trk_combinatoric);

  eventTree->Branch("tp_pt",     &m_tp_pt);
  eventTree->Branch("tp_eta",    &m_tp_eta);
  eventTree->Branch("tp_phi",    &m_tp_phi);
  eventTree->Branch("tp_z0",     &m_tp_z0);
  eventTree->Branch("tp_pdgid",  &m_tp_pdgid);
  eventTree->Branch("tp_nmatch", &m_tp_nmatch);

  eventTree->Branch("matchtrk_pt",      &m_matchtrk_pt);
  eventTree->Branch("matchtrk_eta",     &m_matchtrk_eta);
  eventTree->Branch("matchtrk_phi",     &m_matchtrk_phi);
  eventTree->Branch("matchtrk_z0",      &m_matchtrk_z0);
  eventTree->Branch("matchtrk_chi2",    &m_matchtrk_chi2);
  eventTree->Branch("matchtrk_nstub",   &m_matchtrk_nstub);
  eventTree->Branch("matchtrk_genuine", &m_matchtrk_genuine);

}


//////////
// ANALYZE
void L1TrackNtupleMaker::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  if (!(MyProcess==13 || MyProcess==11 || MyProcess==211 || MyProcess==6 || MyProcess==15)) {
    cout << "The specified MyProcess is invalid! Exiting..." << endl;
    return;
  }


  // clear variables
  m_trk_pt->clear();
  m_trk_eta->clear();
  m_trk_phi->clear();
  m_trk_z0->clear();
  m_trk_chi2->clear();
  m_trk_nstub->clear();
  m_trk_genuine->clear();
  m_trk_unknown->clear();
  m_trk_combinatoric->clear();

  m_tp_pt->clear();
  m_tp_eta->clear();
  m_tp_phi->clear();
  m_tp_z0->clear();
  m_tp_pdgid->clear();
  m_tp_nmatch->clear();

  m_matchtrk_pt->clear();
  m_matchtrk_eta->clear();
  m_matchtrk_phi->clear();
  m_matchtrk_z0->clear();
  m_matchtrk_chi2->clear();
  m_matchtrk_nstub->clear();
  m_matchtrk_genuine->clear();



  //-----------------------------------------------------------------------------------------------
  // retrieve various containers
  //-----------------------------------------------------------------------------------------------

  // L1 tracks
  edm::Handle< std::vector< TTTrack< Ref_PixelDigi_ > > > TTTrackHandle;
  iEvent.getByLabel("TTTracksFromPixelDigis", "Level1TTTracks", TTTrackHandle);
  
  // MC truth association maps
  edm::Handle< TTTrackAssociationMap< Ref_PixelDigi_ > > MCTruthTTTrackHandle;
  iEvent.getByLabel("TTTrackAssociatorFromPixelDigis", "Level1TTTracks", MCTruthTTTrackHandle);
  edm::Handle< TTClusterAssociationMap< Ref_PixelDigi_ > > MCTruthTTClusterHandle;
  iEvent.getByLabel("TTClusterAssociatorFromPixelDigis", "ClusterAccepted", MCTruthTTClusterHandle);
  edm::Handle< TTStubAssociationMap< Ref_PixelDigi_ > > MCTruthTTStubHandle;
  iEvent.getByLabel("TTStubAssociatorFromPixelDigis", "StubAccepted", MCTruthTTStubHandle);

  // tracking particles
  edm::Handle< std::vector< TrackingParticle > > TrackingParticleHandle;
  edm::Handle< std::vector< TrackingVertex > > TrackingVertexHandle;
  iEvent.getByLabel("mix", "MergedTrackTruth", TrackingParticleHandle);
  iEvent.getByLabel("mix", "MergedTrackTruth", TrackingVertexHandle);


  // ----------------------------------------------------------------------------------------------
  // loop over L1 tracks
  // ----------------------------------------------------------------------------------------------

  if (DebugMode) cout << endl << "Loop over L1 tracks!" << endl;

  int this_l1track = 0;
  std::vector< TTTrack< Ref_PixelDigi_ > >::const_iterator iterL1Track;
  for ( iterL1Track = TTTrackHandle->begin(); iterL1Track != TTTrackHandle->end(); iterL1Track++ ) {
    
    edm::Ptr< TTTrack< Ref_PixelDigi_ > > l1track_ptr(TTTrackHandle, this_l1track);
    this_l1track++;

    float tmp_trk_pt   = iterL1Track->getMomentum().perp();
    float tmp_trk_eta  = iterL1Track->getMomentum().eta();
    float tmp_trk_phi  = iterL1Track->getMomentum().phi();
    float tmp_trk_z0   = iterL1Track->getPOCA().z(); //cm
    float tmp_trk_chi2 = iterL1Track->getChi2();
    int tmp_trk_nstub  = (int) iterL1Track->getStubRefs().size();
    
    if (tmp_trk_pt < 2.0) continue; 

    int tmp_trk_genuine = 0;
    int tmp_trk_unknown = 0;
    int tmp_trk_combinatoric = 0;
    if (MCTruthTTTrackHandle->isGenuine(l1track_ptr)) tmp_trk_genuine = 1;
    if (MCTruthTTTrackHandle->isUnknown(l1track_ptr)) tmp_trk_unknown = 1;
    if (MCTruthTTTrackHandle->isCombinatoric(l1track_ptr)) tmp_trk_combinatoric = 1;

    if (DebugMode) {
      cout << "L1 track, pt: " << tmp_trk_pt << " eta: " << tmp_trk_eta << " phi: " << tmp_trk_phi 
	   << " z0: " << tmp_trk_z0 << " chi2: " << tmp_trk_chi2 << " nstub: " << tmp_trk_nstub;
      if (tmp_trk_genuine) cout << " (is genuine)" << endl; 
      if (tmp_trk_unknown) cout << " (is unknown)" << endl; 
      if (tmp_trk_combinatoric) cout << " (is combinatoric)" << endl; 
    }

    m_trk_pt ->push_back(tmp_trk_pt);
    m_trk_eta->push_back(tmp_trk_eta);
    m_trk_phi->push_back(tmp_trk_phi);
    m_trk_z0 ->push_back(tmp_trk_z0);
    m_trk_chi2 ->push_back(tmp_trk_chi2);
    m_trk_nstub->push_back(tmp_trk_nstub);
    m_trk_genuine->push_back(tmp_trk_genuine);
    m_trk_unknown->push_back(tmp_trk_unknown);
    m_trk_combinatoric->push_back(tmp_trk_combinatoric);

  }



  // ----------------------------------------------------------------------------------------------
  // loop over tracking particles
  // ----------------------------------------------------------------------------------------------

  if (DebugMode) cout << endl << "Loop over tracking particles!" << endl;

  int this_tp = 0;
  std::vector< TrackingParticle >::const_iterator iterTP;
  for (iterTP = TrackingParticleHandle->begin(); iterTP != TrackingParticleHandle->end(); ++iterTP) {
 
    edm::Ptr< TrackingParticle > tp_ptr(TrackingParticleHandle, this_tp);
    this_tp++;

    int tmp_eventid = iterTP->eventId().event();
    if (tmp_eventid > 0) continue; //only care about tracking particles from the primary interaction for now

    float tmp_tp_pt  = iterTP->pt();
    float tmp_tp_eta = iterTP->eta();
    float tmp_tp_phi = iterTP->phi(); 
    float tmp_tp_z0  = iterTP->vz();
    int tmp_tp_pdgid = iterTP->pdgId();
    
    if (MyProcess==13 && abs(tmp_tp_pdgid) != 13) continue;
    if (MyProcess==11 && abs(tmp_tp_pdgid) != 11) continue;
    if ((MyProcess==6 || MyProcess==15 || MyProcess==211) && abs(tmp_tp_pdgid) != 211) continue;
    
    if (tmp_tp_pt < 1.0) continue;
    if (fabs(tmp_tp_eta) > 2.5) continue;
    if (fabs(tmp_tp_z0) > 30.0) continue;
    if ((MyProcess==6 || MyProcess==15) && tmp_tp_pt < 2.0) continue;

    if (DebugMode) cout << "Tracking particle, pt: " << tmp_tp_pt << " eta: " << tmp_tp_eta << " phi: " << tmp_tp_phi 
			<< " z0: " << tmp_tp_z0 << " pdgid: " << tmp_tp_pdgid 
			<< " eventID: " << iterTP->eventId().event()
			<< " ttclusters " << MCTruthTTClusterHandle->findTTClusterRefs(tp_ptr).size() 
			<< " ttstubs " << MCTruthTTStubHandle->findTTStubRefs(tp_ptr).size()
			<< " tttracks " << MCTruthTTTrackHandle->findTTTrackPtrs(tp_ptr).size() << endl;

    if (MCTruthTTClusterHandle->findTTClusterRefs(tp_ptr).size() < 1) {
      if (DebugMode) cout << "No matching TTClusters for TP, continuing..." << endl;
      continue;
    }


    // ----------------------------------------------------------------------------------------------
    // look for L1 tracks matched to the tracking particle
    std::vector< edm::Ptr< TTTrack< Ref_PixelDigi_ > > > matchedTracks = MCTruthTTTrackHandle->findTTTrackPtrs(tp_ptr);
    
    int nMatch = 0;
    int i_track = -1;
    
    if (matchedTracks.size() > 0) { 
    
      if (DebugMode && (matchedTracks.size()>1)) cout << "WARNING: TrackingParticle has more than one matched L1 track!" << endl;

      // loop over matched L1 tracks
      for (int it=0; it<(int)matchedTracks.size(); it++) {

	bool tmp_trk_genuine = false;
	if (MCTruthTTTrackHandle->isGenuine(matchedTracks.at(it))) tmp_trk_genuine = true;
	if (!tmp_trk_genuine) continue;

	if (DebugMode) {
	  if (MCTruthTTTrackHandle->findTrackingParticlePtr(matchedTracks.at(it)).isNull()) {
	    cout << "track matched to TP is NOT uniquely matched to a TP" << endl;
	  }
	  else {
	    edm::Ptr< TrackingParticle > my_tp = MCTruthTTTrackHandle->findTrackingParticlePtr(matchedTracks.at(it));
	    cout << "TP matched to track matched to TP ... tp pt = " << my_tp->p4().pt() << " eta = " << my_tp->momentum().eta() << " phi = " << my_tp->momentum().phi() << " z0 = " << my_tp->vertex().z()/10.0 << endl;
	  }
	  cout << "   ... matched L1 track has pt = " << matchedTracks.at(it)->getMomentum().perp() 
	       << " eta = " << matchedTracks.at(it)->getMomentum().eta()
	       << " phi = " << matchedTracks.at(it)->getMomentum().phi()
	       << " chi2 = " << matchedTracks.at(it)->getChi2() 
	       << " z0 = " << matchedTracks.at(it)->getPOCA().z() 
	       << " nstub = " << matchedTracks.at(it)->getStubRefs().size();
	  if (tmp_trk_genuine) cout << " (genuine!) " << endl;
	  else cout << " (NOT genuine) !!!!!" << endl;
	}

	// require L1 track to be genuine and have at least three stubs for it to be a valid match
	if (matchedTracks.at(it)->getStubRefs().size() < 3) continue;
	
	float dmatch_pt  = 999;
	float dmatch_eta = 999;
	float dmatch_phi = 999;
	int match_id = 999;

	edm::Ptr< TrackingParticle > my_tp = MCTruthTTTrackHandle->findTrackingParticlePtr(matchedTracks.at(it));
	dmatch_pt  = fabs(my_tp->p4().pt() - tmp_tp_pt);
	dmatch_eta = fabs(my_tp->p4().eta() - tmp_tp_eta);
	dmatch_phi = fabs(my_tp->p4().phi() - tmp_tp_phi);
	match_id = my_tp->pdgId();
	
	if (dmatch_pt<0.1 && dmatch_eta<0.1 && dmatch_phi<0.1 && tmp_tp_pdgid==match_id) {
	  nMatch++;
	  if (i_track < 0) i_track = it;
	}

      }//end loop over matched L1 tracks

    }// end has at least 1 matched L1 track
    // ----------------------------------------------------------------------------------------------
        

    float tmp_matchtrk_pt   = -999;
    float tmp_matchtrk_eta  = -999;
    float tmp_matchtrk_phi  = -999;
    float tmp_matchtrk_z0   = -999;
    float tmp_matchtrk_chi2 = -999;
    int tmp_matchtrk_nstub  = -999;
    int tmp_matchtrk_genuine = -999;

    if (nMatch > 1) cout << "WARNING *** 2 or more matches to genuine L1 tracks ***" << endl;

    if (nMatch > 0) {
      tmp_matchtrk_pt   = matchedTracks.at(i_track)->getMomentum().perp();
      tmp_matchtrk_eta  = matchedTracks.at(i_track)->getMomentum().eta();
      tmp_matchtrk_phi  = matchedTracks.at(i_track)->getMomentum().phi();
      tmp_matchtrk_z0   = matchedTracks.at(i_track)->getPOCA().z();
      tmp_matchtrk_chi2 = matchedTracks.at(i_track)->getChi2();
      tmp_matchtrk_nstub  = (int) matchedTracks.at(i_track)->getStubRefs().size();
      tmp_matchtrk_genuine = 1;
    }

    m_tp_pt->push_back(tmp_tp_pt);
    m_tp_eta->push_back(tmp_tp_eta);
    m_tp_phi->push_back(tmp_tp_phi);
    m_tp_z0->push_back(tmp_tp_z0);
    m_tp_pdgid->push_back(tmp_tp_pdgid);
    m_tp_nmatch->push_back(nMatch);

    m_matchtrk_pt ->push_back(tmp_matchtrk_pt);
    m_matchtrk_eta->push_back(tmp_matchtrk_eta);
    m_matchtrk_phi->push_back(tmp_matchtrk_phi);
    m_matchtrk_z0 ->push_back(tmp_matchtrk_z0);
    m_matchtrk_chi2 ->push_back(tmp_matchtrk_chi2);
    m_matchtrk_nstub->push_back(tmp_matchtrk_nstub);
    m_matchtrk_genuine->push_back(tmp_matchtrk_genuine);
    
  } //end loop tracking particles
  

  eventTree->Fill();


} // end of analyze()


///////////////////////////
// DEFINE THIS AS A PLUG-IN
DEFINE_FWK_MODULE(L1TrackNtupleMaker);
