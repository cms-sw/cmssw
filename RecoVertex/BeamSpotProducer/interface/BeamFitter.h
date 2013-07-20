#ifndef BeamFitter_H
#define BeamFitter_H

/**_________________________________________________________________
   class:   BeamFitter.h
   package: RecoVertex/BeamSpotProducer



 author: Francisco Yumiceva, Fermilab (yumiceva@fnal.gov)
         Geng-Yuan Jeng, UC Riverside (Geng-Yuan.Jeng@cern.ch)

 version $Id: BeamFitter.h,v 1.50 2013/04/11 23:08:42 wmtan Exp $

 ________________________________________________________________**/

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"
#include "RecoVertex/BeamSpotProducer/interface/BSTrkParameters.h"
#include "RecoVertex/BeamSpotProducer/interface/BSFitter.h"
#include "RecoVertex/BeamSpotProducer/interface/PVFitter.h"

// ROOT
#include "TFile.h"
#include "TTree.h"
#include "TH1F.h"

#include <fstream>

class BeamFitter {
 public:
  BeamFitter() {}
  BeamFitter(const edm::ParameterSet& iConfig);
  virtual ~BeamFitter();

  void readEvent(const edm::Event& iEvent);

  bool runFitter();
  bool runBeamWidthFitter();
  bool runPVandTrkFitter();
  bool runFitterNoTxt();

  reco::BeamSpot getBeamWidth() { return fbeamWidthFit; }
  void runAllFitter();
  void resetTrkVector() { fBSvector.clear(); }
  void resetTotTrk() { ftotal_tracks=0; }
  void resetLSRange() { fbeginLumiOfFit=fendLumiOfFit=-1; }
  void resetRefTime() { freftime[0] = freftime[1] = 0; }
  void setRefTime(time_t t0, time_t t1) {
    freftime[0] = t0;
    freftime[1] = t1;
    // Make sure the string representation of the time
    // is up-to-date
    updateBTime();
  }

  std::pair<time_t,time_t> getRefTime(){
    return std::make_pair(freftime[0], freftime[1]);
  }

  void resetPVFitter() { MyPVFitter->resetAll(); }

  //---these are added to fasciliate BeamMonitor stuff for DIP
  std::size_t  getPVvectorSize() {return (MyPVFitter->getpvStore()).size(); }
  //sc
  void resizeBSvector(unsigned int nsize){
    fBSvector.erase(fBSvector.begin(),fBSvector.begin()+nsize);
   }

  //ssc
  void resizePVvector(unsigned int npvsize){
       MyPVFitter->resizepvStore(npvsize);
   }

 //ssc
  void SetPVInfo(const std::vector<float> &v1_){
     ForDIPPV_.clear();
     ForDIPPV_.assign( v1_.begin(), v1_.end() );
    }

//----------------

  void dumpTxtFile(std::string &,bool);
  void dumpBWTxtFile(std::string &);
  void write2DB();
  reco::BeamSpot getBeamSpot() { return fbeamspot; }
  std::map<int, reco::BeamSpot> getBeamSpotMap() { return fbspotPVMap; }
  std::vector<BSTrkParameters> getBSvector() { return fBSvector; }
  TH1F * getCutFlow() { return h1cutFlow; }
  void subtractFromCutFlow(const TH1F* toSubtract) {
    h1cutFlow->Add(toSubtract, -1.0);
    for (unsigned int i=0; i<sizeof(countPass)/sizeof(countPass[0]); i++){
      countPass[i] = h1cutFlow->GetBinContent(i+1);
    }
  }

  void resetCutFlow() {
    h1cutFlow->Reset();
    ftotal_tracks = 0;
    for (unsigned int i=0; i<sizeof(countPass)/sizeof(countPass[0]); i++)
      countPass[i]=0;
  }

  //ssc
  int getRunNumber() {
    return frun;
  }

  std::pair<int,int> getFitLSRange() {
    return std::make_pair(fbeginLumiOfFit, fendLumiOfFit);
  }
  void setFitLSRange(int ls0,int ls1) {
    fbeginLumiOfFit = ls0;
    fendLumiOfFit = ls1;
  }
  void setRun( int run) { frun = run; }

  int getNTracks() {
    return fBSvector.size();
  }
  int getNPVs() {
    return MyPVFitter->getNPVs();
  }
  const std::map<int, int> &getNPVsperBX() {
    return MyPVFitter->getNPVsperBX();
  }
 private:

  const char * formatBTime( const std::time_t &);
  // Update the fbeginTimeOfFit etc from the refTime
  void updateBTime();
  std::vector<BSTrkParameters> fBSvector;
  reco::BeamSpot fbeamspot;
  reco::BeamSpot fbeamWidthFit;
  std::map< int, reco::BeamSpot> fbspotPVMap;
  BSFitter *fmyalgo;
  std::ofstream fasciiFile;
  std::ofstream fasciiDIP;

  bool debug_;
  bool appendRunTxt_;
  edm::InputTag tracksLabel_;
  edm::InputTag vertexLabel_;
  bool writeTxt_;
  bool writeDIPTxt_;
  bool writeDIPBadFit_;
  std::string outputTxt_;
  std::string outputDIPTxt_;
  double trk_MinpT_;
  double trk_MaxZ_;
  double trk_MaxEta_;
  double trk_MaxIP_;
  int trk_MinNTotLayers_;
  int trk_MinNPixLayers_;
  double trk_MaxNormChi2_;
  std::vector<std::string> trk_Algorithm_;
  std::vector<std::string> trk_Quality_;
  std::vector<reco::TrackBase::TrackQuality> quality_;
  std::vector<reco::TrackBase::TrackAlgorithm> algorithm_;
  double inputBeamWidth_;
  double convergence_;
  int ftotal_tracks;
  int min_Ntrks_;
  bool isMuon_;
  bool fitted_;
  bool ffilename_changed;
   
  //ssc
  std::vector<float> ForDIPPV_; 
  

  // ntuple
  TH1F* h1z;
  bool saveNtuple_;
  bool saveBeamFit_;
  bool savePVVertices_;
  std::string outputfilename_;
  TFile* file_;
  TTree* ftree_;
  double ftheta;
  double fpt;
  double feta;
  int    fcharge;
  double fnormchi2;
  double fphi0;
  double fd0;
  double fd0bs;
  double fsigmad0;
  double fz0;
  double fsigmaz0;
  int fnTotLayerMeas;
  int fnPixelLayerMeas;
  int fnStripLayerMeas;
  int fnTIBLayerMeas;
  int fnTIDLayerMeas;
  int fnTOBLayerMeas;
  int fnTECLayerMeas;
  int fnPXBLayerMeas;
  int fnPXFLayerMeas;
  double fd0phi_chi2;
  double fd0phi_d0;
  double fcov[7][7];
  double fvx;
  double fvy;
  int frun;
  int flumi;
  bool fquality;
  bool falgo;
  bool fpvValid;
  double fpvx, fpvy, fpvz;
  std::time_t freftime[2];

  //beam fit results
  TTree* ftreeFit_;
  int frunFit;
  int fbeginLumiOfFit;
  int fendLumiOfFit;
  char fbeginTimeOfFit[32];
  char fendTimeOfFit[32];
  double fx;
  double fy;
  double fz;
  double fsigmaZ;
  double fdxdz;
  double fdydz;
  double fxErr;
  double fyErr;
  double fzErr;
  double fsigmaZErr;
  double fdxdzErr;
  double fdydzErr;
  double fwidthX;
  double fwidthY;
  double fwidthXErr;
  double fwidthYErr;

  TH1F *h1ntrks;
  TH1F *h1vz_event;
  TH1F *h1cutFlow;
  int countPass[9];

  PVFitter *MyPVFitter;
  TTree* fPVTree_;

};

#endif
