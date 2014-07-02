//////////////////////////////////////////////////////////////////////
//                                                                  //
//  Analyzer for making mini-ntuple for L1 track + EM Tau selection //
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
#include "DataFormats/L1TrackTrigger/interface/L1TkEmParticle.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkEmParticleFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "SimDataFormats/SLHC/interface/L1EGCrystalCluster.h"


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
using namespace l1extra;
using namespace reco;


//////////////////////////////
//                          //
//     CLASS DEFINITION     //
//                          //
//////////////////////////////

class L1TkEmTauNtupleMaker : public edm::EDAnalyzer
{
public:

  // Constructor/destructor
  explicit L1TkEmTauNtupleMaker(const edm::ParameterSet& iConfig);
  virtual ~L1TkEmTauNtupleMaker();

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

  // EM objects
  std::vector<float>* m_em_pt;
  std::vector<float>* m_em_eta;
  std::vector<float>* m_em_phi;

  // tau mc objects
  std::vector<float>* m_tau_pt;
  std::vector<float>* m_tau_eta;
  std::vector<float>* m_tau_phi;
  std::vector<int>* m_tau_mode;

  // tau mc daugther
  std::vector<float>* m_taudaug_pt;
  std::vector<float>* m_taudaug_eta;
  std::vector<float>* m_taudaug_phi;
  std::vector<int>* m_taudaug_id;
  std::vector<int>* m_taudaug_parindex;


  // for L1EmParticle
  edm::InputTag L1EmInputTag;


};


//////////////////////////////////
//                              //
//     CLASS IMPLEMENTATION     //
//                              //
//////////////////////////////////

//////////////
// CONSTRUCTOR
L1TkEmTauNtupleMaker::L1TkEmTauNtupleMaker(edm::ParameterSet const& iConfig) : 
  config(iConfig)
{

  MyProcess = iConfig.getParameter< int >("MyProcess");
  DebugMode = iConfig.getParameter< bool >("DebugMode");
  L1EmInputTag = iConfig.getParameter<edm::InputTag>("L1EmInputTag");

}

/////////////
// DESTRUCTOR
L1TkEmTauNtupleMaker::~L1TkEmTauNtupleMaker()
{
}  

//////////
// END JOB
void L1TkEmTauNtupleMaker::endJob()
{
  // things to be done at the exit of the event Loop
  cerr << "L1TkEmTauNtupleMaker::endJob" << endl;

}

////////////
// BEGIN JOB
void L1TkEmTauNtupleMaker::beginJob()
{

  // things to be done before entering the event Loop
  cerr << "L1TkEmTauNtupleMaker::beginJob" << endl;


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

  m_em_pt    = new std::vector<float>;
  m_em_eta   = new std::vector<float>;
  m_em_phi   = new std::vector<float>;

  m_tau_pt    = new std::vector<float>;
  m_tau_eta   = new std::vector<float>;
  m_tau_phi   = new std::vector<float>;
  m_tau_mode  = new std::vector<int>;

  m_taudaug_pt = new std::vector<float>;
  m_taudaug_eta = new std::vector<float>;
  m_taudaug_phi = new std::vector<float>;
  m_taudaug_id = new std::vector<int>;
  m_taudaug_parindex = new std::vector<int>;


  
  // ntuple
  eventTree = fs->make<TTree>("TkEmTauTree", "Event tree");

  eventTree->Branch("trk_pt",    &m_trk_pt);
  eventTree->Branch("trk_eta",   &m_trk_eta);
  eventTree->Branch("trk_phi",   &m_trk_phi);
  eventTree->Branch("trk_z0",    &m_trk_z0);
  eventTree->Branch("trk_chi2",  &m_trk_chi2);
  eventTree->Branch("trk_nstub", &m_trk_nstub);
  eventTree->Branch("trk_genuine",      &m_trk_genuine);
  eventTree->Branch("trk_unknown",      &m_trk_unknown);
  eventTree->Branch("trk_combinatoric", &m_trk_combinatoric);


  eventTree->Branch("em_pt",     &m_em_pt);
  eventTree->Branch("em_eta",    &m_em_eta);
  eventTree->Branch("em_phi",    &m_em_phi);

  eventTree->Branch("tau_pt",      &m_tau_pt);
  eventTree->Branch("tau_eta",     &m_tau_eta);
  eventTree->Branch("tau_phi",     &m_tau_phi);
  eventTree->Branch("tau_mode",    &m_tau_mode);

  eventTree->Branch("taudaug_pt",      &m_taudaug_pt);
  eventTree->Branch("taudaug_eta",     &m_taudaug_eta);
  eventTree->Branch("taudaug_phi",     &m_taudaug_phi);
  eventTree->Branch("taudaug_id",      &m_taudaug_id);
  eventTree->Branch("taudaug_parindex",      &m_taudaug_parindex);

}


//
// member functions
//

int tauClass(std::vector<const reco::Candidate *>& stabledaughters,
	     double& maxpt) {

  // -999 means not classified
  // 1 means electron
  // 2 means muon
  // 3 means had 1 prong
  // 4 means had 3 prong
  // 5 means had 5 prong
  
  maxpt=-999.9;

  if (stabledaughters[1]->pdgId()==11||stabledaughters[1]->pdgId()==-11) return 1;
  if (stabledaughters[1]->pdgId()==13||stabledaughters[1]->pdgId()==-13) return 2;
    
  int nprong=0;  

  for (unsigned int i=0;i<stabledaughters.size();i++) {
    if (stabledaughters[i]->pdgId()==211||stabledaughters[i]->pdgId()==-211) {
      nprong++;
      if (stabledaughters[i]->pt()>maxpt) maxpt=stabledaughters[i]->pt();
    }
  }

  if (nprong==1) return 3;
  if (nprong==3) return 4;
  if (nprong==5) return 5;

  return -999;

}


void getStableDaughters(const reco::Candidate & p, 
			std::vector<const reco::Candidate *>& stabledaughters){

  int ndaug=p.numberOfDaughters();
  
  for(int j=0;j<ndaug;j++){
    //double vx = p.vx(), vy = p.vy(), vz = p.vz();

    //std::cout << "daug vertex : "<<vx<<" "<<vy<<" "<<vz<<std::endl;

    const reco::Candidate * daug = p.daughter(j);
    if (daug->status()==1) {
      stabledaughters.push_back(daug);
    }
    else {
      getStableDaughters(*daug,stabledaughters);
    }
  }  

}


//////////
// ANALYZE
void L1TkEmTauNtupleMaker::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

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

  m_em_pt->clear();
  m_em_eta->clear();
  m_em_phi->clear();

  m_tau_pt->clear();
  m_tau_eta->clear();
  m_tau_phi->clear();
  m_tau_mode->clear();


  m_taudaug_pt->clear();
  m_taudaug_eta->clear();
  m_taudaug_phi->clear();
  m_taudaug_id->clear();
  m_taudaug_parindex->clear();





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


  //Get generator MC truth

  Handle<GenParticleCollection> genParticles;
  iEvent.getByLabel("genParticles", genParticles);
  int ntau=0;
  for(size_t i = 0; i < genParticles->size(); ++ i) {
    const GenParticle & p = (*genParticles)[i];
    int id = p.pdgId();
    //int st = p.status();
    double eta=p.p4().eta();
    double phi=p.p4().phi();
    double pt=p.p4().pt();
    if (!(id==15||id==-15)) continue;
    if(abs(p.daughter(0)->pdgId())==15) continue;  //skip if daugther is tau
    //int nmother=p.numberOfMothers();
    //if (nmother!=1) continue;
    //if (p.mother(0)->pdgId()!=25) continue;

    //double vx = p.vx(), vy = p.vy(), vz = p.vz();

    //std::cout << "vertex : "<<vx<<" "<<vy<<" "<<vz<<std::endl;

    //std::cout << "i id st eta: "<<i<<" "<<id<<" "<<st<<" "<<eta<<std::endl;

    std::vector<const reco::Candidate *> stabledaughters;

    getStableDaughters(p,stabledaughters);

    double tauplusmaxpt=0.0;

    m_tau_pt->push_back(pt);
    m_tau_eta->push_back(eta);
    m_tau_phi->push_back(phi);
    m_tau_mode->push_back(tauClass(stabledaughters,tauplusmaxpt));

    for (unsigned int j=0;j<stabledaughters.size();j++) {
      m_taudaug_pt->push_back(stabledaughters[j]->pt());
      m_taudaug_eta->push_back(stabledaughters[j]->eta());
      m_taudaug_phi->push_back(stabledaughters[j]->phi());
      m_taudaug_id->push_back(stabledaughters[j]->pdgId());
      m_taudaug_parindex->push_back(ntau);
    }
     
    ntau++;
   
   }



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

  edm::Handle<l1slhc::L1EGCrystalClusterCollection> L1EmHandle;
  iEvent.getByLabel(L1EmInputTag, L1EmHandle);
  std::vector<l1slhc::L1EGCrystalCluster>::const_iterator egIter;

 if ( L1EmHandle.isValid() ) {
   //std::cout << "Found L1EmParticles"<<std::endl;
   for (egIter=L1EmHandle->begin();egIter!=L1EmHandle->end();++egIter) {
     m_em_pt->push_back(egIter->pt());
     m_em_eta->push_back(egIter->eta());
     m_em_phi->push_back(egIter->phi());
   }
 }
 else {
   std::cout << "Did not find L1EmParticles"<<std::endl;
 }



  eventTree->Fill();


} // end of analyze()


///////////////////////////
// DEFINE THIS AS A PLUG-IN
DEFINE_FWK_MODULE(L1TkEmTauNtupleMaker);
