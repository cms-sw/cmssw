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


#include <iostream>
#include <vector>
#include <cmath>
#include <TH1.h>
#include "TFile.h"

using namespace std;

class PixelTrackVal : public edm::EDAnalyzer {
public:
  explicit PixelTrackVal(const edm::ParameterSet& conf);
  ~PixelTrackVal();
  virtual void beginJob(const edm::EventSetup& es);
  virtual void analyze(const edm::Event& ev, const edm::EventSetup& es);
  virtual void endJob();
private:
  edm::ParameterSet conf_; 
  // How noisy should I be
  int verbose_;
  TFile *f_;
  TH1 *h_Pt, *h_TIP, *h_ErrTIP, *h_VtxZ, *h_VtxZ_Pull, *h_Nan;
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

  h_Pt        = new TH1F("h_Pt","h_Pt",20, 0., 20.);
  h_TIP       = new TH1F("h_TIP","h_TIP",100,-0.1,0.1);
  h_ErrTIP    = new TH1F("h_ErrTIP","h_ErrTIP",50,0.,0.017);
  h_VtxZ      = new TH1F("h_VtxZ","h_VtxZ",100,-0.1,0.1);
  h_VtxZ_Pull = new TH1F("h_VtxZ_Pull","h_VtxZ_Pull",100,0.,10);
  h_Nan       = new TH1F("h_Nan","Illegal values for x,y,z,xx,xy,xz,yy,yz,zz",9,0.5,9.5);
}

void PixelTrackVal::analyze(
    const edm::Event& ev, const edm::EventSetup& es)
{
  cout <<"*** PixelTrackVal, analyze event: " << ev.id() << endl;
  edm::Handle<reco::TrackCollection> trackCollection;
  std::string trackCollName = conf_.getParameter<std::string>("TrackCollection");
  ev.getByLabel(trackCollName,trackCollection);
  const reco::TrackCollection tracks = *(trackCollection.product());

 typedef reco::TrackCollection::const_iterator IT;

  if (verbose_ > 0) {
    std::cout << *(trackCollection.provenance()) << std::endl;
    cout << "Reconstructed "<< tracks.size() << " tracks" << std::endl;
  }
  for (unsigned int i=0; i<tracks.size(); i++) {
    if (verbose_ > 0) {
      cout << "\tmomentum: " << tracks[i].momentum()
	   << "\tPT: " << tracks[i].pt()<< endl;
      cout << "\tvertex: " << tracks[i].vertex()
         << "\tTIP: "<< tracks[i].d0() << " +- " << tracks[i].d0Error()
	   << "\tZ0: " << tracks[i].dz() << " +- " << tracks[i].dzError() << endl;
      cout << "\tcharge: " << tracks[i].charge()<< endl;
    }
  }

  typedef reco::TrackCollection::const_iterator IT;
  for (IT it=tracks.begin(); it!=tracks.end(); it++) {
    h_Pt->Fill( it->pt());
    h_TIP->Fill( it->d0() );
    h_ErrTIP->Fill(it->d0Error());
    h_VtxZ->Fill( it->dz() );
    h_VtxZ_Pull->Fill( it->dz()/it->dzError() );

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
  } 
}

void PixelTrackVal::endJob() 
{
  if (f_) f_->Write();
}

DEFINE_FWK_MODULE(PixelTrackVal)
