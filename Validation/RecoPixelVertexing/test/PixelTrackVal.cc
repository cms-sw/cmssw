#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"


#include <iostream>
#include <vector>
#include <cmath>
#include <TH1.h>
#include "TFile.h"

using namespace std;
template <class T> T sqr( T t) {return t*t;}


class PixelTrackVal : public edm::EDAnalyzer {
public:
  explicit PixelTrackVal(const edm::ParameterSet& conf);
  ~PixelTrackVal();
  virtual void beginJob(const edm::EventSetup& es);
  virtual void analyze(const edm::Event& ev, const edm::EventSetup& es);
  virtual void endJob();
private:
  float deltaR(const math::XYZVector & mom1, const math::XYZVector & mom2) const;

  edm::ParameterSet conf_; 
  // How noisy should I be
  int verbose_;
  TFile *f_;
  TH1 *h_Pt, *h_dR, *h_VtxZ, *h_TIP, *h_VtxZ_Pull, *h_Nan;
};

PixelTrackVal::PixelTrackVal(const edm::ParameterSet& conf)
  : conf_(conf),f_(0)
{
  edm::LogInfo("PixelTrackVal")<<" CTOR";
}

PixelTrackVal::~PixelTrackVal()
{
  edm::LogInfo("PixelTrackVal")<<" DTOR";
  delete f_;
}

void PixelTrackVal::beginJob(const edm::EventSetup& es) {
  // How noisy?
  verbose_ = conf_.getUntrackedParameter<unsigned int>("Verbosity",0);

  // Make my little tree
  std::string file = conf_.getUntrackedParameter<std::string>("HistoFile","pixelTrackHistos.root");
//  const char* cwd= gDirectory->GetPath();
  f_ = new TFile(file.c_str(),"RECREATE");

  h_Pt        = new TH1F("h_Pt","h_Pt",31, -2., 1.2);
  h_dR        = new TH1F("h_dR","h_dR",30,0.,0.06);
  h_TIP       = new TH1F("h_TIP","h_TIP",100,-0.1,0.1);
  h_VtxZ      = new TH1F("h_VtxZ","h_VtxZ",100,-0.1,0.1);
  h_VtxZ_Pull = new TH1F("h_VtxZ_Pull","h_VtxZ_Pull",80,0.,8);
  h_Nan       = new TH1F("h_Nan","Illegal values for x,y,z,xx,xy,xz,yy,yz,zz",9,0.5,9.5);
}

void PixelTrackVal::analyze(
    const edm::Event& ev, const edm::EventSetup& es)
{

  using namespace edm;
  using namespace std;
  using namespace reco;

  cout <<"*** PixelTrackVal, analyze event: " << ev.id() << endl;



//------------------------ simulated tracks
  Handle<reco::TrackCollection> trackCollection;
  std::string trackCollName = conf_.getParameter<std::string>("TrackCollection");
  ev.getByLabel(trackCollName,trackCollection);
  const reco::TrackCollection tracks = *(trackCollection.product());

  typedef reco::TrackCollection::const_iterator IT;

  if (verbose_ > 0) {
    std::cout << *(trackCollection.provenance()) << std::endl;
    cout << "Reconstructed "<< tracks.size() << " tracks" << std::endl;
  }

  for (unsigned int idx=0; idx<tracks.size(); idx++) {

    const reco::Track * it= &tracks[idx];
    h_Nan->Fill(1.,isnan(it->momentum().x())*1.);
    h_Nan->Fill(2.,isnan(it->momentum().y())*1.);
    h_Nan->Fill(3.,isnan(it->momentum().z())*1.);
    
    bool problem = false;
    int index = 3;
    for (int i = 0; i != 3; i++) {
      for (int j = i; j != 3; j++) {
	  index++;
	  h_Nan->Fill(index*1., isnan(it->covariance(i, j))*1.);
	  if (isnan(it->covariance(i, j))) problem = true;
	  // in addition, diagonal element must be positive
	  if (j == i && it->covariance(i, j) < 0) {
	    h_Nan->Fill(index*1., 1.);
	    problem = true;
	  }
      }
    }
    if (problem) std::cout <<" *** PROBLEM **" << std::endl;

    if (verbose_ > 0) {
      cout << "\tmomentum: " << tracks[idx].momentum()
	   << "\tPT: " << tracks[idx].pt()<< endl;
      cout << "\tvertex: " << tracks[idx].vertex()
         << "\tTIP: "<< tracks[idx].d0() << " +- " << tracks[idx].d0Error()
	   << "\tZ0: " << tracks[idx].dz() << " +- " << tracks[idx].dzError() << endl;
      cout << "\tcharge: " << tracks[idx].charge()<< endl;
    }
  }

//------------------------ simulated tracks
   
   InputTag simG4 = conf_.getParameter<edm::InputTag>( "simG4" );
   Handle<SimVertexContainer> simVtcs;
   ev.getByLabel( simG4, simVtcs);
   std::cout << "SimVertex " << simVtcs->size() << std::endl;

     for(edm::SimVertexContainer::const_iterator v=simVtcs->begin();
       v!=simVtcs->end(); ++v){
       std::cout << "simvtx "
             << std::setw(10) << std::setprecision(3)
             << v->position().x() << " "
             << v->position().y() << " "
             << v->position().z() << " "
             << v->parentIndex() << " "
             << v->noParent() << " "
             << std::endl;
     }

   Handle<SimTrackContainer> simTrks;
   ev.getByLabel( simG4, simTrks);
   std::cout << "simtrks " << simTrks->size() << std::endl;

//-------------- association
  // matching cuts from Marcin
  float detaMax=0.012;
  float dRMax=0.025;
  typedef SimTrackContainer::const_iterator IP;
  for (IP p=simTrks->begin(); p != simTrks->end(); p++) {
    if ( (*p).noVertex() ) continue;
    if ( (*p).type() == -99) continue;
    if ( (*p).vertIndex() != 0) continue;

    math::XYZVector mom_gen( (*p).momentum().x(), (*p).momentum().y(), (*p).momentum().z());
    float phi_gen = (*p).momentum().phi();
    float pt_gen = (*p).momentum().perp();
    float eta_gen = (*p).momentum().eta();
    HepLorentzVector vtx =(*simVtcs)[p->vertIndex()].position();
    float z_gen  = vtx.z()/10.;

//     cout << "\tmomentum: " <<  (*p).momentum()
//          <<" vtx: "<<(*p).vertIndex()<<" type: "<<(*p).type()
//          << endl;

    typedef reco::TrackCollection::const_iterator IT;
    for (IT it=tracks.begin(); it!=tracks.end(); it++) {
      math::XYZVector mom_rec = (*it).momentum();
      float phi_rec = (*it).momentum().phi();
      float pt_rec = (*it).pt();
      float z_rec  = (*it).vertex().z();
      float eta_rec = (*it).momentum().eta();
//    float chi2   = (*it).chi2();
      float dphi = phi_gen - phi_rec;
      while (dphi > M_PI) dphi -=2*M_PI;
      while (dphi < -M_PI) dphi +=2*M_PI;
      float deta = eta_gen-eta_rec;
      float dz = z_gen-z_rec;
      float dR = deltaR( mom_gen, mom_rec);
      //
      // matched track
      //
      if (fabs(deta) < 0.3 && fabs(dphi) < 0.3) h_dR->Fill( dR);
      if (fabs(deta) < detaMax && dR < dRMax) {
        h_Pt->Fill( (pt_gen - pt_rec)/pt_gen);
        h_TIP->Fill( it->d0() );
        h_VtxZ->Fill( dz );
        h_VtxZ_Pull->Fill( fabs( dz/it->dzError()) );
      }
    }
  } 
}

void PixelTrackVal::endJob() 
{
  if (f_) f_->Write();
}

float PixelTrackVal::deltaR(const  math::XYZVector & m1, const  math::XYZVector & m2) const
{
  float dphi = m1.phi()-m2.phi();
  while (dphi > 2*M_PI) dphi-=2*M_PI;
  while (dphi < -2*M_PI) dphi+=2*M_PI;
  float deta = m1.eta() - m2.eta();
  float dr = sqrt( sqr(dphi) + sqr(deta));
  return dr;

}


DEFINE_FWK_MODULE(PixelTrackVal);
