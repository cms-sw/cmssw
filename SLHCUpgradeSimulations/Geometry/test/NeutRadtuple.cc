#include "SLHCUpgradeSimulations/Geometry/test/NeutRadtuple.h"

#include "SimDataFormats/Track/interface/SimTrack.h"

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FastSimulation/Event/interface/FSimEvent.h"
#include "FastSimulation/Event/interface/FBaseSimEvent.h"
#include "FastSimulation/Event/interface/FSimTrack.h"
#include "FastSimulation/Event/interface/FSimTrackContainer.h"
#include "FastSimulation/Particle/interface/RawParticle.h"

#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"
#include "SimTracker/TrackAssociation/interface/TrackAssociatorByChi2.h"
#include "SimTracker/TrackAssociation/interface/TrackAssociatorByHits.h"
#include "SimTracker/Records/interface/TrackAssociatorRecord.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"

// For ROOT                                                                                                            
#include <TROOT.h>
#include <TTree.h>
#include <TFile.h>
#include <TF1.h>
#include <TH2F.h>
#include <TH1F.h>

using namespace std;
using namespace edm;
using namespace reco;

NeutRadtuple::NeutRadtuple(edm::ParameterSet const& conf) :
  conf_(conf),
  tfile_(0),
  tptree_(0)
{
}

NeutRadtuple::~NeutRadtuple() { }

void NeutRadtuple::endJob()
{
  std::cout << " NeutRadtuple::endJob" << std::endl;
  tfile_->Write();
  tfile_->Close();
}

void NeutRadtuple::beginJob(const edm::EventSetup& es)
{
  std::cout << " NeutRadtuple::beginJob" << std::endl;
  std::string outputFile = conf_.getParameter<std::string>("OutputFile");

  tfile_ = new TFile ( outputFile.c_str() , "RECREATE" );
  tptree_ = new TTree("Radtuple","Neutrino Radiation Length analyzer ntuple");

  int bufsize = 64000;

  //Common Branch                                                                                                             
  tptree_->Branch("evt", &evt_,"run/I:evtnum:numfs", bufsize);
  tptree_->Branch("trk", &trk_,"pdgid/I:nlyrs:theta/F:phi:eta:zee:mom:eng", bufsize);
  tptree_->Branch("lyr", &lyr_,"laynm/I:radln/F:layRpos:layZpos",bufsize);
  init();
}

void NeutRadtuple::analyze(const edm::Event& event, const edm::EventSetup& es)
{

  edm::Handle<edm::FSimTrackContainer > myFSimTracks;
  event.getByType(myFSimTracks);

  edm::Handle<edm::SimVertexContainer > mySimVertices;
  event.getByType(mySimVertices);

  const int numfs = myFSimTracks->size();

  for(size_t i = 0; i < myFSimTracks->size(); ++ i) {
    const FSimTrack & fp = (*myFSimTracks)[i];
    const int mynlyrs = fp.nLayers();
    SimTrack sp = fp.simTrack();
    int mypdgid = sp.type();
    math::XYZTLorentzVectorD spmom = sp.momentum();
    float mytheta = spmom.theta();
    float myphi = spmom.phi();
    float myeta = spmom.eta();
    float mymom = spmom.P();
    float myeng = spmom.energy();
    int ivert = sp.vertIndex();
    float myzee = -9999999.0;
    if (ivert == 0) {
      myzee = (*mySimVertices)[ivert].position().z();
    }
    for(int j = 0; j < mynlyrs; ++j) {
      int   mylaynm = fp.layerNum(j);
      float myradln = fp.layerRadL(j);
      float myRlayer = fp.layerRpos(j);
      float myZlayer = fp.layerZpos(j);

      //      std::cout << "Track " << i << ":  layer, PID, R, Z, phi, eta, zee, radl = ";
      //std::cout << mylaynm << ", " ;
      //std::cout << mypdgid << ", " ;
      //std::cout << myRlayer << ", " ;
      //std::cout << myZlayer << ", " ;
      //std::cout << myphi << ", " ;
      //std::cout << myeta << ", ";
      //std::cout << myzee << ", ";
      //std::cout << myradln  ;
      //std::cout << std::endl;
  
      fillEvt(numfs, event);
      fillTrk(mypdgid,mynlyrs,mytheta,myphi,myeta,myzee,mymom,myeng);
      fillLyr(mylaynm,myradln,myRlayer,myZlayer);
      tptree_->Fill();
      init();
    }
  }
}

void NeutRadtuple::fillEvt(const int numfs, const edm::Event& E)
{
  evt_.run = E.id().run();
  evt_.evtnum = E.id().event();
  evt_.numfs = numfs;
}

void NeutRadtuple::fillTrk(const int pdgid, const int nlyrs, const float theta, 
                           const float phi, const float eta, const float zee, 
                           const float mom, const float eng)
{
  trk_.pdgid = pdgid;
  trk_.nlyrs = nlyrs;
  trk_.theta = theta;
  trk_.phi   = phi;
  trk_.eta   = eta;
  trk_.zee   = zee;
  trk_.mom   = mom;
  trk_.eng   = eng;
}

void NeutRadtuple::fillLyr(const int laynm, const float radln, const float layRpos, const float layZpos)
{
  lyr_.laynm = laynm;
  lyr_.radln = radln;
  lyr_.layRpos = layRpos;
  lyr_.layZpos = layZpos;
}

void NeutRadtuple::init()
{
  evt_.init();
  trk_.init();
  lyr_.init();
}

void NeutRadtuple::evt::init()
{
  int dummy_int = 9999;
  run = dummy_int;
  evtnum = dummy_int;
  numfs = dummy_int;
}

void NeutRadtuple::trk::init()
{
  int dummy_int = 9999;
  float dummy_float = 9999.0;
  pdgid = dummy_int;
  nlyrs = dummy_int;
  theta = dummy_float;
  phi = dummy_float;
  eta = dummy_float;
  zee = dummy_float;
  mom = dummy_float;
  eng = dummy_float;
}

void NeutRadtuple::lyr::init()
{
  int dummy_int = 9999;
  float dummy_float = 9999.0;
  laynm = dummy_int;
  radln = dummy_float;
  layRpos = dummy_float;
  layZpos = dummy_float;
}

//define this as a plug-in                                                                                                    
DEFINE_FWK_MODULE(NeutRadtuple);
