#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"

//#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"

#include "RecoTracker/PixelVertexFinding/interface/PVPositionBuilder.h"
#include "RecoTracker/PixelVertexFinding/interface/PVClusterComparer.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexFitter.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include <iostream>
#include <vector>
#include <cmath>
#include "TTree.h"
#include "TFile.h"
#include "TDirectory.h"

using namespace std;

class PixelVertexTest : public edm::EDAnalyzer {
public:
  explicit PixelVertexTest(const edm::ParameterSet& conf);
  ~PixelVertexTest();
  virtual void beginJob();
  virtual void analyze(const edm::Event& ev, const edm::EventSetup& es);
  virtual void endJob();
private:
  edm::ParameterSet conf_; 
  // How noisy should I be
  int verbose_;
  // Tree of simple vars for testing resolution eff etc
  TTree *t_;
  TFile *f_;
  int ntrk_;
  static const int maxtrk_=1000;
  double pt_[maxtrk_];
  double z0_[maxtrk_];
  double errz0_[maxtrk_];
  //  double tanl_[maxtrk_];
  double theta_[maxtrk_];
  int nvtx_;
  static const int maxvtx_=15;
  double vz_[maxvtx_];
  double vzwt_[maxvtx_];
  double errvz_[maxvtx_];
  double errvzwt_[maxvtx_];
  int nvtx2_;
  double vz2_[maxvtx_];
  double trk2avg_[maxvtx_];
  double errvz2_[maxvtx_];
  int ntrk2_[maxvtx_];
  double sumpt2_[maxvtx_];
  double simx_;
  double simy_;
  double simz_;
  double vxkal_[maxvtx_];
  double vykal_[maxvtx_];
  double vzkal_[maxvtx_];
};

PixelVertexTest::PixelVertexTest(const edm::ParameterSet& conf)
  : conf_(conf),t_(0),f_(0)
{
  edm::LogInfo("PixelVertexTest")<<" CTOR";
}

PixelVertexTest::~PixelVertexTest()
{
  edm::LogInfo("PixelVertexTest")<<" DTOR";
  delete f_;
  //  delete t_;
}

void PixelVertexTest::beginJob() {
  // How noisy?
  verbose_ = conf_.getUntrackedParameter<unsigned int>("Verbosity",0);

  // Make my little tree
  std::string file = conf_.getUntrackedParameter<std::string>("OutputTree","mytree.root");
  const char* cwd= gDirectory->GetPath();
  f_ = new TFile(file.c_str(),"RECREATE");
  t_ = new TTree("t","Pixel Vertex Testing");
  t_->Branch("nvtx",&nvtx_,"nvtx/I");
  t_->Branch("vz",vz_,"vz[nvtx]/D");
  t_->Branch("errvz",errvz_,"errvz[nvtx]/D");
  t_->Branch("vzwt",vzwt_,"vzwt[nvtx]/D");
  t_->Branch("errvzwt",errvzwt_,"errvzwt[nvtx]/D");
  t_->Branch("nvtx2",&nvtx2_,"nvtx2/I");
  t_->Branch("vz2",vz2_,"vz2[nvtx2]/D");
  t_->Branch("trk2avg",trk2avg_,"trk2avg[nvtx2]/D");
  t_->Branch("errvz2",errvz2_,"errvz2[nvtx2]/D");
  t_->Branch("ntrk2",ntrk2_,"ntrk2[nvtx2]/I");
  t_->Branch("sumpt2",sumpt2_,"sumpt2[nvtx2]/D");
  t_->Branch("ntrk",&ntrk_,"ntrk/I");
  t_->Branch("pt",pt_,"pt[ntrk]/D");
  t_->Branch("z0",z0_,"z0[ntrk]/D");
  t_->Branch("errz0",errz0_,"errz0[ntrk]/D");
  //  t_->Branch("tanl",tanl_,"tanl[ntrk]/D");
  t_->Branch("theta",theta_,"theta[ntrk]/D");
  t_->Branch("simx",&simx_,"simx/D");
  t_->Branch("simy",&simy_,"simy/D");
  t_->Branch("simz",&simz_,"simz/D");
  t_->Branch("vxkal",vxkal_,"vxkal[nvtx2]/D");
  t_->Branch("vykal",vykal_,"vykal[nvtx2]/D");
  t_->Branch("vzkal",vzkal_,"vzkal[nvtx2]/D");
  gDirectory->cd(cwd);
}

void PixelVertexTest::analyze(
    const edm::Event& ev, const edm::EventSetup& es)
{
  cout <<"*** PixelVertexTest, analyze event: " << ev.id() << endl;

  edm::ESHandle<MagneticField> field;
  es.get<IdealMagneticFieldRecord>().get(field);

  edm::InputTag simG4 = conf_.getParameter<edm::InputTag>( "simG4" );
  edm::Handle<edm::SimVertexContainer> simVtcs;
  ev.getByLabel( simG4, simVtcs);
  if (verbose_ > 0) {
    cout << "simulated vertices: "<< simVtcs->size() << std::endl;
  }
//  simx_ = (simVtcs->size() > 0) ? (*simVtcs)[0].position().x()/10 : -9999.0;
//  simy_ = (simVtcs->size() > 0) ? (*simVtcs)[0].position().y()/10 : -9999.0;
//  simz_ = (simVtcs->size() > 0) ? (*simVtcs)[0].position().z()/10 : -9999.0;
// No longer need to convert from mm as of version 1_2_0
  simx_ = (simVtcs->size() > 0) ? (*simVtcs)[0].position().x() : -9999.0;
  simy_ = (simVtcs->size() > 0) ? (*simVtcs)[0].position().y() : -9999.0;
  simz_ = (simVtcs->size() > 0) ? (*simVtcs)[0].position().z() : -9999.0;
  if (verbose_ > 1) {
    for (int i=0; i<simVtcs->size(); i++) {
      std::cout << (*simVtcs)[i].parentIndex() << ": " << (*simVtcs)[i].position().x() << ", " << (*simVtcs)[i].position().y() << ", " << (*simVtcs)[i].position().z() << ";  ";
    }
    std::cout << "\n" << std::endl;
  }


  edm::Handle<reco::TrackCollection> trackCollection;
  std::string trackCollName = conf_.getParameter<std::string>("TrackCollection");
  ev.getByLabel(trackCollName,trackCollection);
  const reco::TrackCollection tracks = *(trackCollection.product());

  reco::TrackRefVector trks;

  if (verbose_ > 0) {
    std::cout << *(trackCollection.provenance()) << std::endl;
    cout << "Reconstructed "<< tracks.size() << " tracks" << std::endl;
  }
  ntrk_=0;
  for (unsigned int i=0; i<tracks.size(); i++) {
    if (verbose_ > 0) {
      cout << "\tmomentum: " << tracks[i].momentum()
	   << "\tPT: " << tracks[i].pt()<< endl;
      cout << "\tvertex: " << tracks[i].vertex()
	   << "\tZ0: " << tracks[i].dz() << " +- " << tracks[i].dzError() << endl;
      cout << "\tcharge: " << tracks[i].charge()<< endl;
    }
    trks.push_back( reco::TrackRef(trackCollection, i) );
    // Fill ntuple vars
    if (ntrk_ < maxtrk_) {
      pt_[ntrk_] = tracks[i].pt();
      z0_[ntrk_] = tracks[i].dz();
      //      errz0_[ntrk_] = std::sqrt( tracks[i].covariance(3,3) );
      errz0_[ntrk_] = tracks[i].dzError();
      //      tanl_[ntrk_] = tracks[i].tanDip();
      theta_[ntrk_] = tracks[i].theta();
      ntrk_++;
    }
    if (verbose_ > 0) cout <<"------------------------------------------------"<<endl;
  }
  PVPositionBuilder pos;  
  nvtx_ = 0;
  vz_[nvtx_] = pos.average(trks).value();
  errvz_[nvtx_] = pos.average(trks).error();
  vzwt_[nvtx_] = pos.wtAverage(trks).value();
  errvzwt_[nvtx_] = pos.wtAverage(trks).error();
  nvtx_++;
  if (verbose_ > 0) {
    std::cout << "The average z-position of these tracks is " << vz_[0] << " +- " << errvz_[0] << std::endl;
    std::cout << "The weighted average z-position of these tracks is " << vzwt_[0] << " +- " << errvzwt_[0] << std::endl;
  }

  // NOW let's see if my vertex producer did a darn thing...
  edm::Handle<reco::VertexCollection> vertexCollection;
  ev.getByLabel("pixelVertices",vertexCollection);
  const reco::VertexCollection vertexes = *(vertexCollection.product());
  if (verbose_ > 0) {
    std::cout << *(vertexCollection.provenance()) << std::endl;
    cout << "Reconstructed "<< vertexes.size() << " vertexes" << std::endl;
  }
  nvtx2_ = vertexes.size();
  PVClusterComparer vcompare;
  for (int i=0; i<nvtx2_ && i<maxvtx_; i++) {
    vz2_[i] = vertexes[i].z();
    errvz2_[i] = std::sqrt(vertexes[i].error(2,2));
    ntrk2_[i] = vertexes[i].tracksSize();
    sumpt2_[i] = vcompare.pTSquaredSum(vertexes[i]);
    // Now calculate my own average position by hand to cross check conversion process
    //    trks.clear(); // not yet implemented
    while (! trks.empty()) trks.erase( trks.begin() );
    for (reco::track_iterator j=vertexes[i].tracks_begin(); j!=vertexes[i].tracks_end(); ++j) 
      trks.push_back( *j );
    trk2avg_[i] = pos.wtAverage(trks).value();
  }

  // Now let's send off our tracks to the Kalman fitter to see what great things happen...
  if (nvtx2_ > 0) {
    vector<reco::TransientTrack> t_tks;
    for (int i=0; i<nvtx2_ && i<maxvtx_; i++) {
      t_tks.clear();
      for (reco::track_iterator j=vertexes[i].tracks_begin(); j!=vertexes[i].tracks_end(); ++j) {
	t_tks.push_back( reco::TransientTrack(**j,field.product()) );
      }
      KalmanVertexFitter kvf;
      //      TransientVertex tv = kvf.vertex(t_tks,GlobalPoint(vertexes[i].x(),vertexes[i].y(),vertexes[i].z()));
      TransientVertex tv = kvf.vertex(t_tks);
      if (verbose_>0) std::cout << "Kalman Position: " << reco::Vertex::Point(tv.position()) << std::endl;
      vxkal_[i] = tv.position().x();
      vykal_[i] = tv.position().y();
      vzkal_[i] = tv.position().z();
    }
  }
    
  

  // Finally, fill the tree with the above values
  t_->Fill();
}

void PixelVertexTest::endJob() {
  if (t_) t_->Print();
  if (f_) {
    f_->Print();
    f_->Write();
  }
}

DEFINE_FWK_MODULE(PixelVertexTest);
