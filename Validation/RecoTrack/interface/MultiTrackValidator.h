#ifndef MultiTrackValidator_h
#define MultiTrackValidator_h

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "MagneticField/Engine/interface/MagneticField.h" 
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h" 

#include "SimTracker/TrackAssociation/interface/TrackAssociatorByChi2.h"

#include <iostream>
#include <string>

#include <TH1.h>
#include <TH2.h>
#include <TROOT.h>
#include <TFile.h>
#include <TCanvas.h>
#include <TGraphErrors.h>

using namespace edm;
using namespace std;

class MultiTrackValidator : public edm::EDAnalyzer {
 public:

  MultiTrackValidator(const edm::ParameterSet& pset):
    sim(pset.getParameter<string>("sim")),
    label(pset.getParameter< vector<string> >("label")),
    out(pset.getParameter<string>("out")),
    open(pset.getParameter<string>("open")),
    min(pset.getParameter<double>("min")),
    max(pset.getParameter<double>("max")),
    nint(pset.getParameter<int>("nint"))
    {
      hFile = new TFile( out.c_str(), open.c_str() );
    }
  
  ~MultiTrackValidator(){
    if (hFile!=0) {
      hFile->Close();
      delete hFile;
    }
  }

  void beginJob( const EventSetup &);
  virtual void analyze(const edm::Event&, const edm::EventSetup& );
  void endJob();

 private:

  string sim;
  vector<string> label;
  string out, open;
  double  min, max;
  int nint;
  
  vector<TH1F*> h_ptSIM, h_etaSIM, h_tracksSIM, h_vertposSIM;
  vector<TH1F*> h_tracks, h_nchi2, h_nchi2_prob, h_hits, h_effic, h_ptrmsh, h_deltaeta, h_charge;
  vector<TH1F*> h_pt, h_eta, h_pullTheta,h_pullPhi0,h_pullD0,h_pullDz,h_pullK, h_pt2;
  vector<TH2F*> chi2_vs_nhits, chi2_vs_eta, nhits_vs_eta, ptres_vs_eta, etares_vs_eta;
  vector<TH1F*> h_assochi2, h_assochi2_prob, h_hits_eta;
  
  vector< vector<double> > etaintervals;
  vector< vector<double> > hitseta;
  vector< vector<int> > totSIM,totREC;
  
  vector< vector<TH1F*> > ptdistrib;
  vector< vector<TH1F*> > etadistrib;
  TFile *  hFile;  

  edm::ESHandle<MagneticField> theMF;

  TrackAssociatorBase * associator;
  TrackAssociatorByChi2 * associatorForParamAtPca;
  
};


#endif
