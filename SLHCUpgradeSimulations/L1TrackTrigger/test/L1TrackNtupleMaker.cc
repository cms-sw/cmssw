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
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/L1TrackTrigger/interface/TTCluster.h"
#include "DataFormats/L1TrackTrigger/interface/TTStub.h"
#include "DataFormats/L1TrackTrigger/interface/TTTrack.h"
#include "DataFormats/L1TrackTrigger/interface/TTPixelTrack.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertex.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTClusterAssociationMap.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTStubAssociationMap.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTTrackAssociationMap.h"
#include "Geometry/Records/interface/StackedTrackerGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/StackedTrackerGeometry.h"

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
#include "Geometry/TrackerGeometryBuilder/interface/StackedTrackerDetUnit.h"
#include "DataFormats/SiPixelDetId/interface/StackedTrackerDetId.h"

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
  bool SaveAllTracks;
  bool DoPixelTrack;
  bool SaveStubs;

  edm::InputTag L1TrackInputTag;
  edm::InputTag MCTruthTrackInputTag;


  //-----------------------------------------------------------------------------------------------
  // tree & branches for mini-ntuple

  TTree* eventTree;

  // all L1 tracks
  std::vector<float>* m_trk_pt;
  std::vector<float>* m_trk_eta;
  std::vector<float>* m_trk_phi;
  std::vector<float>* m_trk_z0;
  std::vector<float>* m_trk_chi2; 
  std::vector<float>* m_trk_consistency; 
  std::vector<int>*   m_trk_nstub;
  std::vector<int>*   m_trk_genuine;
  std::vector<int>*   m_trk_unknown;
  std::vector<int>*   m_trk_combinatoric;
  std::vector<int>*   m_trk_fake; //0 fake, 1 track from primary interaction, 2 secondary track
  std::vector<int>*   m_trk_matchtp_pdgid;
  std::vector<float>* m_trk_matchtp_pt;
  std::vector<float>* m_trk_matchtp_eta;
  std::vector<float>* m_trk_matchtp_phi;
  std::vector<float>* m_trk_matchtp_z0;
  std::vector<float>* m_trk_matchtp_dxy;

  // all L1 pixel tracks
  std::vector<float>* m_pixtrk_pt;
  std::vector<float>* m_pixtrk_eta;  
  std::vector<float>* m_pixtrk_phi;
  std::vector<float>* m_pixtrk_z0;
  std::vector<float>* m_pixtrk_d0;
  std::vector<float>* m_pixtrk_chi2;    //this is the chi2 of the combined track with pixel information
  std::vector<float>* m_pixtrk_trkchi2; //this is the chi2 of the non-pixel track part only
  std::vector<int>*   m_pixtrk_nstub; 

  // all tracking particles
  std::vector<float>* m_tp_pt;
  std::vector<float>* m_tp_eta;
  std::vector<float>* m_tp_phi;
  std::vector<float>* m_tp_dxy;
  std::vector<float>* m_tp_d0;
  std::vector<float>* m_tp_z0;
  std::vector<float>* m_tp_d0_prod;
  std::vector<float>* m_tp_z0_prod;
  std::vector<int>*   m_tp_pdgid;
  std::vector<int>*   m_tp_nmatch;
  std::vector<int>*   m_tp_npixmatch;
  std::vector<int>*   m_tp_nstub;

  // *L1 track* properties if m_tp_nmatch > 0
  std::vector<float>* m_matchtrk_pt;
  std::vector<float>* m_matchtrk_eta;
  std::vector<float>* m_matchtrk_phi;
  std::vector<float>* m_matchtrk_z0;
  std::vector<float>* m_matchtrk_chi2; 
  std::vector<float>* m_matchtrk_consistency; 
  std::vector<int>*   m_matchtrk_nstub;
  std::vector<int>*   m_matchtrk_genuine;

  // *L1 track* properties ***for 5-parameter fit*** if m_tp_nmatch > 0
  std::vector<float>* m_matchtrk5p_pt;
  std::vector<float>* m_matchtrk5p_eta;
  std::vector<float>* m_matchtrk5p_phi;
  std::vector<float>* m_matchtrk5p_z0;
  std::vector<float>* m_matchtrk5p_d0;
  std::vector<float>* m_matchtrk5p_chi2; 
  std::vector<float>* m_matchtrk5p_consistency; 

  // *L1 pixel track* properties if m_tp_npixmatch > 0
  std::vector<float>* m_matchpixtrk_pt;
  std::vector<float>* m_matchpixtrk_eta;  
  std::vector<float>* m_matchpixtrk_phi;
  std::vector<float>* m_matchpixtrk_z0;
  std::vector<float>* m_matchpixtrk_d0;
  std::vector<float>* m_matchpixtrk_chi2;    //this is the chi2 of the combined track with pixel information
  std::vector<float>* m_matchpixtrk_trkchi2; //this is the chi2 of the non-pixel track part only
  std::vector<int>*   m_matchpixtrk_nstub; 

  // ALL stubs
  std::vector<float>* m_allstub_x;
  std::vector<float>* m_allstub_y;
  std::vector<float>* m_allstub_z;
  std::vector<float>* m_allstub_pt;
  std::vector<float>* m_allstub_ptsign;
  std::vector<int>*   m_allstub_isBarrel;
  std::vector<int>*   m_allstub_layer;
  std::vector<int>*   m_allstub_isPS;

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
  SaveAllTracks = iConfig.getParameter< bool >("SaveAllTracks");
  DoPixelTrack = iConfig.getParameter< bool >("DoPixelTrack");
  SaveStubs = iConfig.getParameter< bool >("SaveStubs");
  L1TrackInputTag = iConfig.getParameter<edm::InputTag>("L1TrackInputTag");
  MCTruthTrackInputTag = iConfig.getParameter<edm::InputTag>("MCTruthTrackInputTag");

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
  m_trk_consistency  = new std::vector<float>;
  m_trk_genuine      = new std::vector<int>;
  m_trk_unknown      = new std::vector<int>;
  m_trk_combinatoric = new std::vector<int>;
  m_trk_fake = new std::vector<int>;
  m_trk_matchtp_pdgid = new std::vector<int>;
  m_trk_matchtp_pt = new std::vector<float>;
  m_trk_matchtp_eta = new std::vector<float>;
  m_trk_matchtp_phi = new std::vector<float>;
  m_trk_matchtp_z0 = new std::vector<float>;
  m_trk_matchtp_dxy = new std::vector<float>;

  m_pixtrk_pt    = new std::vector<float>;
  m_pixtrk_eta   = new std::vector<float>;
  m_pixtrk_phi   = new std::vector<float>;
  m_pixtrk_z0    = new std::vector<float>;
  m_pixtrk_d0    = new std::vector<float>;
  m_pixtrk_chi2  = new std::vector<float>;
  m_pixtrk_trkchi2  = new std::vector<float>;
  m_pixtrk_nstub = new std::vector<int>;

  m_tp_pt     = new std::vector<float>;
  m_tp_eta    = new std::vector<float>;
  m_tp_phi    = new std::vector<float>;
  m_tp_dxy    = new std::vector<float>;
  m_tp_d0     = new std::vector<float>;
  m_tp_z0     = new std::vector<float>;
  m_tp_d0_prod = new std::vector<float>;
  m_tp_z0_prod = new std::vector<float>;
  m_tp_pdgid  = new std::vector<int>;
  m_tp_nmatch = new std::vector<int>;
  m_tp_npixmatch = new std::vector<int>;
  m_tp_nstub = new std::vector<int>;

  m_matchtrk_pt    = new std::vector<float>;
  m_matchtrk_eta   = new std::vector<float>;
  m_matchtrk_phi   = new std::vector<float>;
  m_matchtrk_z0    = new std::vector<float>;
  m_matchtrk_chi2  = new std::vector<float>;
  m_matchtrk_nstub = new std::vector<int>;
  m_matchtrk_consistency = new std::vector<float>;
  m_matchtrk_genuine     = new std::vector<int>;

  m_matchtrk5p_pt    = new std::vector<float>;
  m_matchtrk5p_eta   = new std::vector<float>;
  m_matchtrk5p_phi   = new std::vector<float>;
  m_matchtrk5p_z0    = new std::vector<float>;
  m_matchtrk5p_d0    = new std::vector<float>;
  m_matchtrk5p_chi2  = new std::vector<float>;
  m_matchtrk5p_consistency = new std::vector<float>;
  
  m_matchpixtrk_pt    = new std::vector<float>;
  m_matchpixtrk_eta   = new std::vector<float>;
  m_matchpixtrk_phi   = new std::vector<float>;
  m_matchpixtrk_z0    = new std::vector<float>;
  m_matchpixtrk_d0    = new std::vector<float>;
  m_matchpixtrk_chi2  = new std::vector<float>;
  m_matchpixtrk_trkchi2  = new std::vector<float>;
  m_matchpixtrk_nstub = new std::vector<int>;

  m_allstub_x = new std::vector<float>;
  m_allstub_y = new std::vector<float>;
  m_allstub_z = new std::vector<float>;
  m_allstub_pt     = new std::vector<float>;
  m_allstub_ptsign = new std::vector<float>;
  m_allstub_isBarrel = new std::vector<int>;
  m_allstub_layer    = new std::vector<int>;
  m_allstub_isPS     = new std::vector<int>;


  // ntuple
  eventTree = fs->make<TTree>("eventTree", "Event tree");

  if (SaveAllTracks) {
    eventTree->Branch("trk_pt",    &m_trk_pt);
    eventTree->Branch("trk_eta",   &m_trk_eta);
    eventTree->Branch("trk_phi",   &m_trk_phi);
    eventTree->Branch("trk_z0",    &m_trk_z0);
    eventTree->Branch("trk_chi2",  &m_trk_chi2);
    eventTree->Branch("trk_nstub", &m_trk_nstub);
    eventTree->Branch("trk_consistency",  &m_trk_consistency);
    eventTree->Branch("trk_genuine",      &m_trk_genuine);
    eventTree->Branch("trk_unknown",      &m_trk_unknown);
    eventTree->Branch("trk_combinatoric", &m_trk_combinatoric);
    eventTree->Branch("trk_fake", &m_trk_fake);
    eventTree->Branch("trk_matchtp_pdgid", &m_trk_matchtp_pdgid);
    eventTree->Branch("trk_matchtp_pt", &m_trk_matchtp_pt);
    eventTree->Branch("trk_matchtp_eta", &m_trk_matchtp_eta);
    eventTree->Branch("trk_matchtp_phi", &m_trk_matchtp_phi);
    eventTree->Branch("trk_matchtp_z0", &m_trk_matchtp_z0);
    eventTree->Branch("trk_matchtp_dxy", &m_trk_matchtp_dxy);
  }
  if (SaveAllTracks && DoPixelTrack) {
    eventTree->Branch("pixtrk_pt",    &m_pixtrk_pt);
    eventTree->Branch("pixtrk_eta",   &m_pixtrk_eta);
    eventTree->Branch("pixtrk_phi",   &m_pixtrk_phi);
    eventTree->Branch("pixtrk_z0",    &m_pixtrk_z0);
    eventTree->Branch("pixtrk_d0",    &m_pixtrk_d0);
    eventTree->Branch("pixtrk_chi2",  &m_pixtrk_chi2);
    eventTree->Branch("pixtrk_trkchi2",  &m_pixtrk_trkchi2);
    eventTree->Branch("pixtrk_nstub", &m_pixtrk_nstub);
  }

  eventTree->Branch("tp_pt",     &m_tp_pt);
  eventTree->Branch("tp_eta",    &m_tp_eta);
  eventTree->Branch("tp_phi",    &m_tp_phi);
  eventTree->Branch("tp_dxy",    &m_tp_dxy);
  eventTree->Branch("tp_d0",     &m_tp_d0);
  eventTree->Branch("tp_z0",     &m_tp_z0);
  eventTree->Branch("tp_d0_prod",&m_tp_d0_prod);
  eventTree->Branch("tp_z0_prod",&m_tp_z0_prod);
  eventTree->Branch("tp_pdgid",  &m_tp_pdgid);
  eventTree->Branch("tp_nmatch", &m_tp_nmatch);
  eventTree->Branch("tp_npixmatch", &m_tp_npixmatch);
  eventTree->Branch("tp_nstub", &m_tp_nstub);

  eventTree->Branch("matchtrk_pt",      &m_matchtrk_pt);
  eventTree->Branch("matchtrk_eta",     &m_matchtrk_eta);
  eventTree->Branch("matchtrk_phi",     &m_matchtrk_phi);
  eventTree->Branch("matchtrk_z0",      &m_matchtrk_z0);
  eventTree->Branch("matchtrk_chi2",    &m_matchtrk_chi2);
  eventTree->Branch("matchtrk_nstub",   &m_matchtrk_nstub);
  eventTree->Branch("matchtrk_genuine", &m_matchtrk_genuine);
  eventTree->Branch("matchtrk_consistency", &m_matchtrk_consistency);

  eventTree->Branch("matchtrk5p_pt",      &m_matchtrk5p_pt);
  eventTree->Branch("matchtrk5p_eta",     &m_matchtrk5p_eta);
  eventTree->Branch("matchtrk5p_phi",     &m_matchtrk5p_phi);
  eventTree->Branch("matchtrk5p_z0",      &m_matchtrk5p_z0);
  eventTree->Branch("matchtrk5p_d0",      &m_matchtrk5p_d0);
  eventTree->Branch("matchtrk5p_chi2",    &m_matchtrk5p_chi2);
  eventTree->Branch("matchtrk5p_consistency", &m_matchtrk5p_consistency);

  if (DoPixelTrack) {
    eventTree->Branch("matchpixtrk_pt",      &m_matchpixtrk_pt);
    eventTree->Branch("matchpixtrk_eta",     &m_matchpixtrk_eta);
    eventTree->Branch("matchpixtrk_phi",     &m_matchpixtrk_phi);
    eventTree->Branch("matchpixtrk_z0",      &m_matchpixtrk_z0);
    eventTree->Branch("matchpixtrk_d0",      &m_matchpixtrk_d0);
    eventTree->Branch("matchpixtrk_chi2",    &m_matchpixtrk_chi2);
    eventTree->Branch("matchpixtrk_trkchi2", &m_matchpixtrk_trkchi2);
    eventTree->Branch("matchpixtrk_nstub",   &m_matchpixtrk_nstub);
  }

  if (SaveStubs) {
    eventTree->Branch("allstub_x", &m_allstub_x);
    eventTree->Branch("allstub_y", &m_allstub_y);
    eventTree->Branch("allstub_z", &m_allstub_z);
    eventTree->Branch("allstub_pt",    &m_allstub_pt);
    eventTree->Branch("allstub_ptsign",&m_allstub_ptsign);
    eventTree->Branch("allstub_isBarrel", &m_allstub_isBarrel);
    eventTree->Branch("allstub_layer",    &m_allstub_layer);
    eventTree->Branch("allstub_isPS",     &m_allstub_isPS);
  }

}


//////////
// ANALYZE
void L1TrackNtupleMaker::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  if (!(MyProcess==13 || MyProcess==11 || MyProcess==211 || MyProcess==6 || MyProcess==15 || MyProcess==1)) {
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
  m_trk_consistency->clear();
  m_trk_genuine->clear();
  m_trk_unknown->clear();
  m_trk_combinatoric->clear();
  m_trk_fake->clear();
  m_trk_matchtp_pdgid->clear();
  m_trk_matchtp_pt->clear();
  m_trk_matchtp_eta->clear();
  m_trk_matchtp_phi->clear();
  m_trk_matchtp_z0->clear();
  m_trk_matchtp_dxy->clear();

  m_pixtrk_pt->clear();
  m_pixtrk_eta->clear();
  m_pixtrk_phi->clear();
  m_pixtrk_z0->clear();
  m_pixtrk_d0->clear();
  m_pixtrk_chi2->clear();
  m_pixtrk_trkchi2->clear();
  m_pixtrk_nstub->clear();

  m_tp_pt->clear();
  m_tp_eta->clear();
  m_tp_phi->clear();
  m_tp_dxy->clear();
  m_tp_d0->clear();
  m_tp_z0->clear();
  m_tp_d0_prod->clear();
  m_tp_z0_prod->clear();
  m_tp_pdgid->clear();
  m_tp_nmatch->clear();
  m_tp_npixmatch->clear();
  m_tp_nstub->clear();

  m_matchtrk_pt->clear();
  m_matchtrk_eta->clear();
  m_matchtrk_phi->clear();
  m_matchtrk_z0->clear();
  m_matchtrk_chi2->clear();
  m_matchtrk_consistency->clear();
  m_matchtrk_nstub->clear();
  m_matchtrk_genuine->clear();

  m_matchtrk5p_pt->clear();
  m_matchtrk5p_eta->clear();
  m_matchtrk5p_phi->clear();
  m_matchtrk5p_z0->clear();
  m_matchtrk5p_d0->clear();
  m_matchtrk5p_chi2->clear();
  m_matchtrk5p_consistency->clear();

  m_matchpixtrk_pt->clear();
  m_matchpixtrk_eta->clear();
  m_matchpixtrk_phi->clear();
  m_matchpixtrk_z0->clear();
  m_matchpixtrk_d0->clear();
  m_matchpixtrk_chi2->clear();
  m_matchpixtrk_trkchi2->clear();
  m_matchpixtrk_nstub->clear();

  if (SaveStubs) {
    m_allstub_x->clear();
    m_allstub_y->clear();
    m_allstub_z->clear();
    m_allstub_pt->clear();
    m_allstub_ptsign->clear();
    m_allstub_isBarrel->clear();
    m_allstub_layer->clear();
    m_allstub_isPS->clear();
  }


  //-----------------------------------------------------------------------------------------------
  // retrieve various containers
  //-----------------------------------------------------------------------------------------------

  // L1 tracks
  edm::Handle< std::vector< TTTrack< Ref_PixelDigi_ > > > TTTrackHandle;
  //iEvent.getByLabel("TTTracksFromPixelDigis", "Level1TTTracks", TTTrackHandle);
  iEvent.getByLabel(L1TrackInputTag, TTTrackHandle);
  
  // L1 stubs
  edm::Handle< edmNew::DetSetVector< TTStub< Ref_PixelDigi_ > > > TTStubHandle;
  if (SaveStubs) iEvent.getByLabel("TTStubsFromPixelDigis", "StubAccepted", TTStubHandle);

  // L1PixelTracks
  edm::Handle< std::vector< TTPixelTrack > > TTPixelTrackHandle;
  iEvent.getByLabel("L1PixelTrackFit", "Level1PixelTracks", TTPixelTrackHandle);

  // MC truth association maps
  edm::Handle< TTTrackAssociationMap< Ref_PixelDigi_ > > MCTruthTTTrackHandle;
  //iEvent.getByLabel("TTTrackAssociatorFromPixelDigis", "Level1TTTracks", MCTruthTTTrackHandle);
  iEvent.getByLabel(MCTruthTrackInputTag, MCTruthTTTrackHandle);
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
  // If saving stubs, need geometry handles etc.
  // ----------------------------------------------------------------------------------------------

  // geomtry 
  edm::ESHandle< StackedTrackerGeometry >         StackedGeometryHandle;
  const StackedTrackerGeometry*                   theStackedGeometry = 0;
  if (SaveStubs) {  
    iSetup.get< StackedTrackerGeometryRecord >().get(StackedGeometryHandle);
    theStackedGeometry = StackedGeometryHandle.product();
  }

  // magnetic field
  edm::ESHandle< MagneticField > magneticFieldHandle;
  iSetup.get< IdealMagneticFieldRecord >().get(magneticFieldHandle);
  double mMagneticFieldStrength = 0;
  if (SaveStubs) {  
    const MagneticField* theMagneticField = magneticFieldHandle.product();
    mMagneticFieldStrength = theMagneticField->inTesla(GlobalPoint(0,0,0)).z();
  }


  // ----------------------------------------------------------------------------------------------
  // loop over ALL L1 stubs
  // ----------------------------------------------------------------------------------------------

  if (SaveStubs) {
  
    if (DebugMode) cout << endl << "Loop over all stubs... " << endl;

    edmNew::DetSetVector<TTStub<Ref_PixelDigi_> >::const_iterator iterStubDet;
    for ( iterStubDet = TTStubHandle->begin(); iterStubDet != TTStubHandle->end(); ++iterStubDet ) {
      
      edmNew::DetSet<TTStub<Ref_PixelDigi_> >::const_iterator iterTTStub;
      for ( iterTTStub = iterStubDet->begin(); iterTTStub != iterStubDet->end(); ++iterTTStub ) {
	
	const TTStub<Ref_PixelDigi_>* stub=iterTTStub;
	      
	StackedTrackerDetId thisDetId = stub->getDetId();
	
	bool isPS = theStackedGeometry->isPSModule(thisDetId);
	bool isBarrel = thisDetId.isBarrel();

	int layer = -1;
	if (isBarrel) layer = thisDetId.iLayer();
	else layer = thisDetId.iDisk();
	  
	GlobalPoint posStub = theStackedGeometry->findGlobalPosition(stub);

	float tmp_stub_x = posStub.x();
	float tmp_stub_y = posStub.y();
	float tmp_stub_z = posStub.z();
    
	float tmp_stub_pt = theStackedGeometry->findRoughPt(mMagneticFieldStrength,stub);
	float tmp_stub_sign = 1.0;

	float trigBend = stub->getTriggerBend();
	if (trigBend<0 && tmp_stub_z<125.0) 
	  tmp_stub_sign = (-1)*tmp_stub_sign;
	else if (trigBend>0 && tmp_stub_z>125.0)
	  tmp_stub_sign = (-1)*tmp_stub_sign;

	m_allstub_x->push_back(tmp_stub_x);
	m_allstub_y->push_back(tmp_stub_y);
	m_allstub_z->push_back(tmp_stub_z);
	m_allstub_pt->push_back(tmp_stub_pt);
	m_allstub_ptsign->push_back(tmp_stub_sign);
	m_allstub_isPS->push_back(isPS);
	m_allstub_isBarrel->push_back(isBarrel);
	m_allstub_layer->push_back(layer);
	
      }
    }

  }// end if save stubs	


  // ----------------------------------------------------------------------------------------------
  // loop over L1 tracks
  // ----------------------------------------------------------------------------------------------

  if (SaveAllTracks) {
    
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
      float tmp_trk_consistency = iterL1Track->getStubPtConsistency();
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
      m_trk_consistency->push_back(tmp_trk_consistency);
      m_trk_nstub->push_back(tmp_trk_nstub);
      m_trk_genuine->push_back(tmp_trk_genuine);
      m_trk_unknown->push_back(tmp_trk_unknown);
      m_trk_combinatoric->push_back(tmp_trk_combinatoric);
      

      // ----------------------------------------------------------------------------------------------
      // for studying the fake rate
      // ----------------------------------------------------------------------------------------------

      edm::Ptr< TrackingParticle > my_tp = MCTruthTTTrackHandle->findTrackingParticlePtr(l1track_ptr);

      int myFake = 0;

      int myTP_pdgid = -999;
      float myTP_pt = -999;
      float myTP_eta = -999;
      float myTP_phi = -999;
      float myTP_z0 = -999;
      float myTP_dxy = -999;

      if (my_tp.isNull()) myFake = 0;
      else {
	int tmp_eventid = my_tp->eventId().event();

	if (tmp_eventid > 0) myFake = 2;
	else myFake = 1;

	myTP_pdgid = my_tp->pdgId();
	myTP_pt = my_tp->p4().pt();
	myTP_eta = my_tp->p4().eta();
	myTP_phi = my_tp->p4().phi();
	myTP_z0 = my_tp->vertex().z();
	
	float myTP_x0 = my_tp->vertex().x();
	float myTP_y0 = my_tp->vertex().y();
	myTP_dxy = sqrt(myTP_x0*myTP_x0 + myTP_y0*myTP_y0);

	if (DebugMode) {
	  cout << "TP matched to track has pt = " << my_tp->p4().pt() << " eta = " << my_tp->momentum().eta() 
	       << " phi = " << my_tp->momentum().phi() << " z0 = " << my_tp->vertex().z()
	       << " pdgid = " <<  my_tp->pdgId() << " dxy = " << myTP_dxy << endl;
	}
      }

      m_trk_fake->push_back(myFake);

      m_trk_matchtp_pdgid->push_back(myTP_pdgid);
      m_trk_matchtp_pt->push_back(myTP_pt);
      m_trk_matchtp_eta->push_back(myTP_eta);
      m_trk_matchtp_phi->push_back(myTP_phi);
      m_trk_matchtp_z0->push_back(myTP_z0);
      m_trk_matchtp_dxy->push_back(myTP_dxy);


    }//end track loop

  }//end if SaveAllTracks



  // ----------------------------------------------------------------------------------------------
  // loop over L1 pixel tracks
  // ----------------------------------------------------------------------------------------------

  if (SaveAllTracks && DoPixelTrack) {
    
    if (DebugMode) cout << endl << "Loop over L1 pixel tracks!" << endl;
    
    int this_l1pixeltrack = 0;
    std::vector<TTPixelTrack>::const_iterator iterL1PixelTrack;
    for ( iterL1PixelTrack = TTPixelTrackHandle->begin(); iterL1PixelTrack != TTPixelTrackHandle->end(); iterL1PixelTrack++ ) {
      
      edm::Ptr< TTPixelTrack > l1pixeltrack_ptr(TTPixelTrackHandle, this_l1pixeltrack);
      this_l1pixeltrack++;
      
      float tmp_pixtrk_pt   = iterL1PixelTrack->getMomentum().perp();
      float tmp_pixtrk_eta  = iterL1PixelTrack->getMomentum().eta();
      float tmp_pixtrk_phi  = iterL1PixelTrack->getMomentum().phi();
      float tmp_pixtrk_z0   = iterL1PixelTrack->getPOCA().z(); //cm
      float tmp_pixtrk_chi2 = iterL1PixelTrack->getChi2();
      float tmp_pixtrk_x0   = iterL1PixelTrack->getPOCA().x(); //cm
      float tmp_pixtrk_y0   = iterL1PixelTrack->getPOCA().y(); //cm

      float tmp_pixtrk_d0   = -tmp_pixtrk_x0*sin(tmp_pixtrk_phi) + tmp_pixtrk_y0*cos(tmp_pixtrk_phi);

      if (tmp_pixtrk_pt < 2.0) continue;


      // get L1 track corresponding to pixel track for # stubs
      const edm::Ref<std::vector<TTTrack<Ref_PixelDigi_> >, TTTrack<Ref_PixelDigi_> > matched_tttrack = iterL1PixelTrack->getL1Track();

      int tmp_pixtrk_nstub  = (int) matched_tttrack->getStubRefs().size();
      float tmp_pixtrk_trkchi2 = matched_tttrack->getChi2(5);  //pixel tracks are formed from 5-parameter L1 tracks
            
      if (DebugMode) {
	cout << "L1 pixel track, pt: " << tmp_pixtrk_pt << " eta: " << tmp_pixtrk_eta << " phi: " << tmp_pixtrk_phi 
	     << " z0: " << tmp_pixtrk_z0 << " d0: " << tmp_pixtrk_d0 << " chi2: " << tmp_pixtrk_chi2 << " nstub: " << tmp_pixtrk_nstub << endl;
      }
      
      m_pixtrk_pt ->push_back(tmp_pixtrk_pt);
      m_pixtrk_eta->push_back(tmp_pixtrk_eta);
      m_pixtrk_phi->push_back(tmp_pixtrk_phi);
      m_pixtrk_z0 ->push_back(tmp_pixtrk_z0);
      m_pixtrk_d0 ->push_back(tmp_pixtrk_d0);
      m_pixtrk_chi2 ->push_back(tmp_pixtrk_chi2);
      m_pixtrk_trkchi2 ->push_back(tmp_pixtrk_trkchi2);
      m_pixtrk_nstub->push_back(tmp_pixtrk_nstub);
    
    }

  }//end if (SaveAllTracks && DoPixelTrack)



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
    if (MyProcess != 1 && tmp_eventid > 0) continue; //only care about tracking particles from the primary interaction

    float tmp_tp_pt  = iterTP->pt();
    float tmp_tp_eta = iterTP->eta();
    float tmp_tp_phi = iterTP->phi(); 
    float tmp_tp_vz  = iterTP->vz();
    float tmp_tp_vx  = iterTP->vx();
    float tmp_tp_vy  = iterTP->vy();
    int tmp_tp_pdgid = iterTP->pdgId();
    float tmp_tp_z0_prod = tmp_tp_vz;
    float tmp_tp_d0_prod = -tmp_tp_vx*sin(tmp_tp_phi) + tmp_tp_vy*cos(tmp_tp_phi);

    if (MyProcess==13 && abs(tmp_tp_pdgid) != 13) continue;
    if (MyProcess==11 && abs(tmp_tp_pdgid) != 11) continue;
    if ((MyProcess==6 || MyProcess==15 || MyProcess==211) && abs(tmp_tp_pdgid) != 211) continue;


    if (tmp_tp_pt < 1.0) continue;
    if (fabs(tmp_tp_eta) > 2.5) continue;
    if ((MyProcess==6 || MyProcess==15 || MyProcess==1) && tmp_tp_pt < 2.0) continue;


    // ----------------------------------------------------------------------------------------------
    // get d0/z0 propagated back to the IP

    float tmp_tp_t = tan(2.0*atan(1.0)-2.0*atan(exp(-tmp_tp_eta)));

    float delx = -tmp_tp_vx;
    float dely = -tmp_tp_vy;
	
    float A = 0.01*0.5696;
    float Kmagnitude = A / tmp_tp_pt;
    
    float tmp_tp_charge = tp_ptr->charge();
    float K = Kmagnitude * tmp_tp_charge;
    float d = 0;
	
    float tmp_tp_x0p = delx - (d + 1./(2. * K)*sin(tmp_tp_phi));
    float tmp_tp_y0p = dely + (d + 1./(2. * K)*cos(tmp_tp_phi));
    float tmp_tp_rp = sqrt(tmp_tp_x0p*tmp_tp_x0p + tmp_tp_y0p*tmp_tp_y0p);
    float tmp_tp_d0 = tmp_tp_charge*tmp_tp_rp - (1. / (2. * K));
	
    tmp_tp_d0 = tmp_tp_d0*(-1); //fix d0 sign

    static double pi = 4.0*atan(1.0);
    float delphi = tmp_tp_phi-atan2(-K*tmp_tp_x0p,K*tmp_tp_y0p);
    if (delphi<-pi) delphi+=2.0*pi;
    if (delphi>pi) delphi-=2.0*pi;
    float tmp_tp_z0 = tmp_tp_vz+tmp_tp_t*delphi/(2.0*K);
    // ----------------------------------------------------------------------------------------------
    
    if (fabs(tmp_tp_z0) > 30.0) continue;


    // for pions in ttbar, only consider TPs coming from near the IP!
    float dxy = sqrt(tmp_tp_vx*tmp_tp_vx + tmp_tp_vy*tmp_tp_vy);
    float tmp_tp_dxy = dxy;
    if (MyProcess==6 && (dxy > 1.0)) continue;


    if (DebugMode) cout << "Tracking particle, pt: " << tmp_tp_pt << " eta: " << tmp_tp_eta << " phi: " << tmp_tp_phi 
			<< " z0: " << tmp_tp_z0 << " d0: " << tmp_tp_d0 
			<< " z_prod: " << tmp_tp_z0_prod << " d_prod: " << tmp_tp_d0_prod 
			<< " pdgid: " << tmp_tp_pdgid << " eventID: " << iterTP->eventId().event()
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
    int nPixelMatch = 0;
    
    int nStubTP = (int) MCTruthTTStubHandle->findTTStubRefs(tp_ptr).size();

    if (matchedTracks.size() > 0) { 
    
      if (DebugMode && (matchedTracks.size()>1)) cout << "WARNING: TrackingParticle has more than one matched L1 track!" << endl;


      // ----------------------------------------------------------------------------------------------
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
	    cout << "TP matched to track matched to TP ... tp pt = " << my_tp->p4().pt() << " eta = " << my_tp->momentum().eta() 
		 << " phi = " << my_tp->momentum().phi() << " z0 = " << my_tp->vertex().z() << endl;
	  }
	  cout << "   ... matched L1 track has pt = " << matchedTracks.at(it)->getMomentum().perp() 
	       << " eta = " << matchedTracks.at(it)->getMomentum().eta()
	       << " phi = " << matchedTracks.at(it)->getMomentum().phi()
	       << " chi2 = " << matchedTracks.at(it)->getChi2() 
	       << " consistency = " << matchedTracks.at(it)->getStubPtConsistency() 
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
    float tmp_matchtrk_consistency = -999;
    int tmp_matchtrk_nstub  = -999;
    int tmp_matchtrk_genuine = -999;

    float tmp_matchtrk5p_pt   = -999;
    float tmp_matchtrk5p_eta  = -999;
    float tmp_matchtrk5p_phi  = -999;
    float tmp_matchtrk5p_x0   = -999;
    float tmp_matchtrk5p_y0   = -999;
    float tmp_matchtrk5p_z0   = -999;
    float tmp_matchtrk5p_d0   = -999;
    float tmp_matchtrk5p_chi2 = -999;
    float tmp_matchtrk5p_consistency = -999;

    float tmp_matchpixtrk_pt   = -999;
    float tmp_matchpixtrk_eta  = -999;
    float tmp_matchpixtrk_phi  = -999;
    float tmp_matchpixtrk_x0   = -999;
    float tmp_matchpixtrk_y0   = -999;
    float tmp_matchpixtrk_z0   = -999;
    float tmp_matchpixtrk_d0   = -999;
    int tmp_matchpixtrk_nstub  = -999;
    float tmp_matchpixtrk_chi2 = -999;
    float tmp_matchpixtrk_trkchi2 = -999;
    

    if (nMatch > 1) cout << "WARNING *** 2 or more matches to genuine L1 tracks ***" << endl;

    if (nMatch > 0) {
      tmp_matchtrk_pt   = matchedTracks.at(i_track)->getMomentum().perp();
      tmp_matchtrk_eta  = matchedTracks.at(i_track)->getMomentum().eta();
      tmp_matchtrk_phi  = matchedTracks.at(i_track)->getMomentum().phi();
      tmp_matchtrk_z0   = matchedTracks.at(i_track)->getPOCA().z();
      tmp_matchtrk_chi2 = matchedTracks.at(i_track)->getChi2();
      tmp_matchtrk_consistency = matchedTracks.at(i_track)->getStubPtConsistency();
      tmp_matchtrk_nstub  = (int) matchedTracks.at(i_track)->getStubRefs().size();
      tmp_matchtrk_genuine = 1;

      tmp_matchtrk5p_pt   = matchedTracks.at(i_track)->getMomentum(5).perp();
      tmp_matchtrk5p_eta  = matchedTracks.at(i_track)->getMomentum(5).eta();
      tmp_matchtrk5p_phi  = matchedTracks.at(i_track)->getMomentum(5).phi();
      tmp_matchtrk5p_z0   = matchedTracks.at(i_track)->getPOCA(5).z();
      tmp_matchtrk5p_x0   = matchedTracks.at(i_track)->getPOCA(5).x();
      tmp_matchtrk5p_y0   = matchedTracks.at(i_track)->getPOCA(5).y();

      tmp_matchtrk5p_d0   = -tmp_matchtrk5p_x0*sin(tmp_matchtrk5p_phi) + tmp_matchtrk5p_y0*cos(tmp_matchtrk5p_phi);

      tmp_matchtrk5p_chi2 = matchedTracks.at(i_track)->getChi2(5);
      tmp_matchtrk5p_consistency = matchedTracks.at(i_track)->getStubPtConsistency(5);


      if (DoPixelTrack) {
	
	bool pixelmatch = false;	
	
	// ----------------------------------------------------------------------------------------------
	// CHECK FOR MATCHING PIXEL TRACK 
	
	int this_l1pixeltrack = 0;
	std::vector<TTPixelTrack>::const_iterator iterL1PixelTrack;
	for ( iterL1PixelTrack = TTPixelTrackHandle->begin(); iterL1PixelTrack != TTPixelTrackHandle->end(); iterL1PixelTrack++ ) {
	  
	  edm::Ptr< TTPixelTrack > l1pixeltrack_ptr(TTPixelTrackHandle, this_l1pixeltrack);
	  this_l1pixeltrack++;
	  
	  // get L1 track corresponding to pixel track
	  const edm::Ref<std::vector<TTTrack<Ref_PixelDigi_> >, TTTrack<Ref_PixelDigi_> > matched_tttrack = iterL1PixelTrack->getL1Track();

	  if (!pixelmatch && matchedTracks.at(i_track)->isTheSameAs(*matched_tttrack) == true) {
	    
	    pixelmatch = true;
	    nPixelMatch++;

	    tmp_matchpixtrk_pt  = iterL1PixelTrack->getMomentum().perp();
	    tmp_matchpixtrk_eta = iterL1PixelTrack->getMomentum().eta();
	    tmp_matchpixtrk_phi = iterL1PixelTrack->getMomentum().phi();
	    tmp_matchpixtrk_x0  = iterL1PixelTrack->getPOCA().x();
	    tmp_matchpixtrk_y0  = iterL1PixelTrack->getPOCA().y();
	    tmp_matchpixtrk_z0  = iterL1PixelTrack->getPOCA().z();
	    tmp_matchpixtrk_chi2 = iterL1PixelTrack->getChi2();
	    
	    tmp_matchpixtrk_d0   = -tmp_matchpixtrk_x0*sin(tmp_matchpixtrk_phi) + tmp_matchpixtrk_y0*cos(tmp_matchpixtrk_phi);

	    tmp_matchpixtrk_nstub = (int) matched_tttrack->getStubRefs().size();
	    tmp_matchpixtrk_trkchi2 = matched_tttrack->getChi2(5);

	  }
	}//end loop match pixel track
      }//end if DoPixelTrack

    }//end (nMatch > 0)

    if (DebugMode) cout << "Matching 5p track, d0 = " << tmp_matchtrk5p_d0 << ", matching pixel track, d0 = " << tmp_matchpixtrk_d0 << endl << endl;

    m_tp_pt->push_back(tmp_tp_pt);
    m_tp_eta->push_back(tmp_tp_eta);
    m_tp_phi->push_back(tmp_tp_phi);
    m_tp_dxy->push_back(tmp_tp_dxy);
    m_tp_z0->push_back(tmp_tp_z0);
    m_tp_d0->push_back(tmp_tp_d0);
    m_tp_z0_prod->push_back(tmp_tp_z0_prod);
    m_tp_d0_prod->push_back(tmp_tp_d0_prod);
    m_tp_pdgid->push_back(tmp_tp_pdgid);
    m_tp_nmatch->push_back(nMatch);
    m_tp_npixmatch->push_back(nPixelMatch);
    m_tp_nstub->push_back(nStubTP);

    m_matchtrk_pt ->push_back(tmp_matchtrk_pt);
    m_matchtrk_eta->push_back(tmp_matchtrk_eta);
    m_matchtrk_phi->push_back(tmp_matchtrk_phi);
    m_matchtrk_z0 ->push_back(tmp_matchtrk_z0);
    m_matchtrk_chi2 ->push_back(tmp_matchtrk_chi2);
    m_matchtrk_consistency->push_back(tmp_matchtrk_consistency);
    m_matchtrk_nstub->push_back(tmp_matchtrk_nstub);
    m_matchtrk_genuine->push_back(tmp_matchtrk_genuine);

    m_matchtrk5p_pt ->push_back(tmp_matchtrk5p_pt);
    m_matchtrk5p_eta->push_back(tmp_matchtrk5p_eta);
    m_matchtrk5p_phi->push_back(tmp_matchtrk5p_phi);
    m_matchtrk5p_z0 ->push_back(tmp_matchtrk5p_z0);
    m_matchtrk5p_d0 ->push_back(tmp_matchtrk5p_d0);
    m_matchtrk5p_chi2->push_back(tmp_matchtrk5p_chi2);
    m_matchtrk5p_consistency->push_back(tmp_matchtrk5p_consistency);


    if (DoPixelTrack) {
      m_matchpixtrk_pt ->push_back(tmp_matchpixtrk_pt);
      m_matchpixtrk_eta->push_back(tmp_matchpixtrk_eta);
      m_matchpixtrk_phi->push_back(tmp_matchpixtrk_phi);
      m_matchpixtrk_z0 ->push_back(tmp_matchpixtrk_z0);
      m_matchpixtrk_d0 ->push_back(tmp_matchpixtrk_d0);
      m_matchpixtrk_chi2 ->push_back(tmp_matchpixtrk_chi2);
      m_matchpixtrk_nstub->push_back(tmp_matchpixtrk_nstub);
      m_matchpixtrk_trkchi2 ->push_back(tmp_matchpixtrk_trkchi2);
    }

    
  } //end loop tracking particles
  

  eventTree->Fill();


} // end of analyze()


///////////////////////////
// DEFINE THIS AS A PLUG-IN
DEFINE_FWK_MODULE(L1TrackNtupleMaker);
