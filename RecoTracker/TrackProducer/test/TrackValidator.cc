#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

#include <iostream>
#include <string>

#include <TH1.h>
#include <TH2.h>
#include <TROOT.h>
#include <TFile.h>
#include <TCanvas.h>

using namespace edm;
using namespace std;

class TrackValidator : public edm::EDAnalyzer {
 public:
  TrackValidator(const edm::ParameterSet& pset)
    : sim(pset.getParameter<string>("sim")),
      label(pset.getParameter<string>("label")),
      out(pset.getParameter<string>("out")),
      min(pset.getParameter<double>("min")),
      max(pset.getParameter<double>("max")),
      nint(pset.getParameter<int>("nint"))
  {}

  ~TrackValidator(){}

  void beginJob( const EventSetup & ) {
    double step=(max-min)/nint;
    ostringstream title,name;
    etaintervals.push_back(0);
    for (double d=min;d<max;d=d+step) {
      etaintervals.push_back(d+step);
      totSIM.push_back(0);
      totREC.push_back(0);
      name.str("");
      title.str("");
      name <<"pt["<<d<<","<<d+step<<"]";
      title <<"p_{t} residue "<< d << "<#eta<"<<d+step;
      ptdistrib.push_back(new TH1F(name.str().c_str(),title.str().c_str(), 200, -2, 2 ));
      name.str("");
      title.str("");
      name <<"eta["<<d<<","<<d+step<<"]";
      title <<"eta residue "<< d << "<#eta<"<<d+step;
      etadistrib.push_back(new TH1F(name.str().c_str(),title.str().c_str(), 200, -0.2, 0.2 ));
    }

    h_ptSIM     = new TH1F("ptSIM", "generated p_{t}", 5500, 0, 110 );
    h_etaSIM    = new TH1F("etaSIM", "generated pseudorapidity", 500, 0, 5 );
    h_tracksSIM = new TH1F("tracksSIM","number of simluated tracks",100,-0.5,99.5);
    h_vertposSIM= new TH1F("vertposSIM","Transverse position of sim vertices",1000,-0.5,10000.5);

//     h_pt     = new TH1F("pt", "p_{t} residue", 2000, -500, 500 );
    h_pt     = new TH1F("pt", "p_{t} residue", 200, -2, 2 );
    h_pt2    = new TH1F("pt2", "p_{t} residue (#tracks>1)", 300, -15, 15 );
    h_eta    = new TH1F("eta", "pseudorapidity residue", 1000, -0.1, 0.1 );
    h_tracks = new TH1F("tracks","number of reconstructed tracks",10,-0.5,9.5);
    h_nchi2  = new TH1F("nchi2", "normalized chi2", 200, 0, 20 );
    h_hits   = new TH1F("hits", "number of hits per track", 30, -0.5, 29.5 );
    h_effic  = new TH1F("effic","efficiency vs #eta",nint,&etaintervals[0]);
    h_ptrmsh = new TH1F("PtRMS","PtRMS vs #eta",nint,&etaintervals[0]);
    h_deltaeta= new TH1F("etaRMS","etaRMS vs #eta",nint,&etaintervals[0]);
    h_charge  = new TH1F("charge","charge",3,-1.5,1.5);

    h_pullTheta = new TH1F("pullTheta","pull of theta parameter",100,-10,10);
    h_pullPhi0  = new TH1F("pullPhi0","pull of phi0 parameter",1000,-10,10);
//     h_pullD0    = new TH1F("pullD0","pull of d0 parameter",100,-10,10);
//     h_pullDz    = new TH1F("pullDz","pull of dz parameter",100,-10,10);

    chi2_vs_nhits= new TH2F("chi2_vs_nhits","chi2 vs nhits",25,0,25,100,0,10);
    chi2_vs_eta  = new TH2F("chi2_vs_eta","chi2 vs eta",nint,min,max,100,0,10);
    nhits_vs_eta = new TH2F("nhits_vs_eta","nhits vs eta",nint,min,max,25,0,25);
    ptres_vs_eta = new TH2F("ptres_vs_eta","ptresidue vs eta",nint,min,max,200,-2,2);
    etares_vs_eta = new TH2F("etares_vs_eta","etaresidue vs eta",nint,min,max,200,-0.1,0.1);
  }

  virtual void analyze(const edm::Event& event, const edm::EventSetup& setup){

    //
    //get collections from the event
    //
    edm::Handle<SimTrackContainer> simTrackCollection;
    event.getByLabel(sim, simTrackCollection);
    const SimTrackContainer simTC = *(simTrackCollection.product());

    edm::Handle<SimVertexContainer> simVertexCollection;
    event.getByLabel(sim, simVertexCollection);
    const SimVertexContainer simVC = *(simVertexCollection.product());

    edm::Handle<reco::TrackCollection> trackCollection;
    event.getByLabel(label, trackCollection);
    const reco::TrackCollection tC = *(trackCollection.product());

    //
    //fill simulation histograms
    //
    h_tracksSIM->Fill(simTC.size());
    for (SimTrackContainer::const_iterator simTrack=simTC.begin(); simTrack!=simTC.end(); simTrack++){
      h_ptSIM->Fill(simTrack->momentum().perp());
      h_etaSIM->Fill(simTrack->momentum().pseudoRapidity());

      if (simTrack->type()!=13) continue;
      //compute number of tracks per eta interval
      int i=0;
      for (vector<double>::iterator h=etaintervals.begin(); h!=etaintervals.end()-1; h++){
	if (abs(simTrack->momentum().pseudoRapidity())>etaintervals[i]&&
	    abs(simTrack->momentum().pseudoRapidity())<etaintervals[i+1]) {
	  totSIM[i]++;
	  bool doit=false;
	  for (reco::TrackCollection::const_iterator track=tC.begin(); track!=tC.end(); track++){
	    if (abs(track->pt()-simTrack->momentum().perp())<(simTrack->momentum().perp()*0.1)) doit=true; 
	  }
	  if (doit) totREC[i]++;
	}
	i++;
      }
    }

    for (SimVertexContainer::const_iterator simVertex=simVC.begin(); simVertex!=simVC.end(); simVertex++){
      h_vertposSIM->Fill(simVertex->position().perp());
      if (0.5 < simVertex->position().perp() && simVertex->position().perp() < 1000) {
	cout << "" << endl;
	cout << "simVertex->position().perp(): " << simVertex->position().perp() << endl;
	cout << "simVertex->position().z()   : " << simVertex->position().z() << endl;
      }
    }

    //
    //fill reconstructed track histograms
    //
    h_tracks->Fill(tC.size());
    for (reco::TrackCollection::const_iterator track=tC.begin(); track!=tC.end(); track++){
      
      //nchi2 and hits global distributions
      h_nchi2->Fill(track->normalizedChi2());
      //      h_hitsCTF->Fill(track->numberOfValidHits());//found());
      h_hits->Fill(track->found());
      //       chi2_vs_nhits->Fill(track->numberOfValidHits(),track->normalizedChi2());
      chi2_vs_nhits->Fill(track->found(),track->normalizedChi2());
      chi2_vs_eta->Fill(track->eta(),track->normalizedChi2());
      //       nhits_vs_eta->Fill(track->eta(),track->numberOfValidHits());
      nhits_vs_eta->Fill(track->eta(),track->found());
      h_charge->Fill( track->charge() );

      //pt, eta residue, theta, phi0, d0, dz pull
      double ptres =1000;
      double etares=1000;
      double thetares=1000;
      double phi0res=1000;
//       double d0res=1000;
//       double dzres=1000;
      for (SimTrackContainer::const_iterator simTrack=simTC.begin(); simTrack!=simTC.end(); simTrack++){
	if (simTrack->type()!=13) continue;
	double tmp=track->pt()-simTrack->momentum().perp();
	if (tC.size()>1) h_pt2->Fill(tmp);
	if (abs(tmp)<abs(ptres)) {
	  ptres=tmp; 
	  etares=track->eta()-simTrack->momentum().pseudoRapidity();
	  thetares=(track->theta()-simTrack->momentum().theta())/track->thetaError();
 	  phi0res=(track->phi0()-simTrack->momentum().phi())/track->phi0Error();
	}
      }
      h_pt->Fill(ptres);
      h_eta->Fill(etares);
      ptres_vs_eta->Fill(track->eta(),ptres);
      etares_vs_eta->Fill(track->eta(),etares);
      h_pullTheta->Fill(thetares);
      h_pullPhi0->Fill(phi0res);


      //pt residue distribution per eta interval
      int i=0;
      for (vector<TH1F*>::iterator h=ptdistrib.begin(); h!=ptdistrib.end(); h++){
	for (SimTrackContainer::const_iterator simTrack=simTC.begin(); simTrack!=simTC.end(); simTrack++){
	  if (simTrack->type()!=13) continue;
	  ptres=1000;
	  if (abs(simTrack->momentum().pseudoRapidity())>etaintervals[i]&&
	      abs(simTrack->momentum().pseudoRapidity())<etaintervals[i+1]) {
	    double tmp=track->pt()-simTrack->momentum().perp();
	    if (abs(tmp)<abs(ptres)) ptres=tmp;
	  }
	}
	(*h)->Fill(ptres);
	i++;
      }
      //eta residue distribution per eta interval
      i=0;
      for (vector<TH1F*>::iterator h=etadistrib.begin(); h!=etadistrib.end(); h++){
	for (SimTrackContainer::const_iterator simTrack=simTC.begin(); simTrack!=simTC.end(); simTrack++){
	  if (simTrack->type()!=13) continue;
	  etares=1000; 
	  ptres =1000;
	  if (abs(simTrack->momentum().pseudoRapidity())>etaintervals[i]&&
	      abs(simTrack->momentum().pseudoRapidity())<etaintervals[i+1]) {
	    double tmp=track->pt()-simTrack->momentum().perp();
// 	    double tmp2=track->eta()-simTrack->momentum().pseudoRapidity();
	    if (abs(tmp)<abs(ptres)) etares=track->eta()-simTrack->momentum().pseudoRapidity();
	  }
	}
	(*h)->Fill(etares);
	i++;
      }
    }


  }

  void endJob() {
    TFile hFile( out.c_str(), "UPDATE" );

//     if ( (TDirectory *) hFile.Get(label.c_str())!=0 ) hFile.SetOption("RECREATE");

    TDirectory * p = hFile.mkdir(label.c_str());

    //write simulation histos
    TDirectory * simD = p->mkdir("simulation");
    simD->cd();
    h_ptSIM->Write();
    h_etaSIM->Write();
    h_tracksSIM->Write();
    h_vertposSIM->Write();

    //fill pt rms plot versus eta and write pt residue distribution per eta interval histo
    TDirectory * ptD = p->mkdir("ptdistribution");
    ptD->cd();
    int i=0;
    for (vector<TH1F*>::iterator h=ptdistrib.begin(); h!=ptdistrib.end(); h++){
      (*h)->Write();
      h_ptrmsh->Fill(etaintervals[i+1]-0.00001 ,(*h)->GetRMS());
      i++;
    }

    //fill eta rms plot versus eta and write eta residue distribution per eta interval histo
    TDirectory * etaD = p->mkdir("etadistribution");
    etaD->cd();
    i=0;
    for (vector<TH1F*>::iterator h=etadistrib.begin(); h!=etadistrib.end(); h++){
      (*h)->Write();
      h_deltaeta->Fill(etaintervals[i+1]-0.00001 ,(*h)->GetRMS());
      i++;
    }

    //write the other histos
    p->cd();
    int j=0;
    for (vector<int>::iterator h=totSIM.begin(); h!=totSIM.end(); h++){
//       cout << "etaintervals[j+1]: " << etaintervals[j+1] << endl;
//       cout << "((double) totREC[j])/((double) totSIM[j]): " << ((double) totREC[j])/((double) totSIM[j]) << endl;
      h_effic->Fill(etaintervals[j+1]-0.00001, ((double) totREC[j])/((double) totSIM[j]));
      j++;
    }

    h_pt->Write();
    h_pt2->Write();
    h_eta->Write();
    h_tracks->Write();
    h_nchi2->Write();
    h_hits->Write();
    h_effic->Write();
    h_ptrmsh->Write();
    h_deltaeta->Write();
    chi2_vs_nhits->Write();
    chi2_vs_eta->Write();
    nhits_vs_eta->Write();
    ptres_vs_eta->Write();
    etares_vs_eta->Write();
    h_charge->Write();

    h_pullTheta->Write();
    h_pullPhi0->Write();

    hFile.Close();
  }

private:
  string sim,label,out;
  double  min,max;
  int nint;
  TH1F *h_ptSIM, *h_etaSIM, *h_tracksSIM, *h_vertposSIM;
  TH1F *h_tracks, *h_nchi2, *h_hits, *h_effic, *h_ptrmsh, *h_deltaeta, *h_charge;
  TH1F *h_pt, *h_eta, *h_pullTheta,*h_pullPhi0,*h_pullD0,*h_pullDz, *h_pt2;
  TH2F *chi2_vs_nhits, *chi2_vs_eta, *nhits_vs_eta, *ptres_vs_eta, *etares_vs_eta;
  vector<double> etaintervals;
  vector<int> totSIM,totREC;
  vector<TH1F*> ptdistrib;
  vector<TH1F*> etadistrib;
 

};

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(TrackValidator);

