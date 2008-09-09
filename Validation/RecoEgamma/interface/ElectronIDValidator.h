#ifndef ElectronIDValidator_h
#define ElectronIDValidator_h

//----------------------------------------------------------

#include <memory>

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/EDProduct.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/ValueMap.h"  

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

#include "TString.h"
#include <vector>


class TFile;
class TTree;
class TH1F;
class TH2F;
class TH1I;
class TProfile;
class TString;

class ElectronIDValidator : public edm::EDAnalyzer
{
 public:
  
  explicit ElectronIDValidator(const edm::ParameterSet& conf);
  
  virtual ~ElectronIDValidator();
  
  virtual void beginJob(edm::EventSetup const& iSetup);
  virtual void endJob();
  virtual void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup); 

 private:

  TFile *histfile_;
  TTree *tree_;

  //MC truth
  
  TH1F *h_mcNum;
  TH1F *h_eleNum;
  TH1F *h_gamNum;
  
  TH1F *h_simEta;
  TH1F *h_simPt;
  TH1F *h_simPhi;
  TH1F *h_simAbsEta;

  TH1F *hB_simEta;
  TH1F *hB_simPt;
  TH1F *hB_simPhi;
  TH1F *hB_simAbsEta;

  TH1F *hEC_simEta;
  TH1F *hEC_simPt;
  TH1F *hEC_simPhi;
  TH1F *hEC_simAbsEta;
  
  //=========================== Ele Classes ================================ 
  
  int n_class;
    
  std::vector<TString> v_class;
 
  //=====================Electron Physical Variables========================

  int n_eleVar;
  int n_eleMatch;
      
  std::vector<TString> v_eleVar;
 
  //======================Supercluster Physical Variables====================
  
  int n_sclVar;
  
  std::vector<TString> v_sclVar;
 
  //======================Initialize Histos Pointers=========================
  
  
  TH1F *h_ele[23][6];     //Electron Histos: Contains All RecoEle, Matched Reco, and RecoEle that have passed the MC matching AND the eleID  
  TH1F *h_scl[6][6];      //Electron's SC Histos: Contains All RecoEle's SC, Matched Reco'SC, and RecoEle'SC that have passed the MC matching AND the eleID
  TH1F *h_IDele[23][6];   //Electron Histos: Contains RecoEle that have passed the eleID
  TH1F *h_IDscl[6][6];    //Electron's SC Histos: Contains RecoEle's SC that have passed the eleID
  TH1F *h_ele_eff[4][6];  //Electron Efficiencies Histos: THe Efficiency is calculated with the formula #(MatchedEle && IDele)/#MCele
 
  //===Barrel===   
  TH1F *hB_ele[23][6];
  TH1F *hB_scl[6][6];
  TH1F *hB_IDele[23][6];  
  TH1F *hB_IDscl[6][6];
  TH1F *hB_ele_eff[4][6];
    
  //===EndCaps===  
  TH1F *hEC_ele[23][6];
  TH1F *hEC_scl[6][6];
  TH1F *hEC_IDele[23][6];
  TH1F *hEC_IDscl[6][6];
  TH1F *hEC_ele_eff[4][6];
  
  std::string outputFile_; 
  std::string electronCollection_;
  
  //For Ele ID Value maps Reading 
  std::string electronLabelRobustL_;
  std::string electronLabelRobustT_;
  std::string electronLabelLoose_;
  std::string electronLabelTight_;
  
  //For the Cluster Shape reading
  edm::InputTag reducedBarrelRecHitCollection_;
  edm::InputTag reducedEndcapRecHitCollection_;
  
  //MC truth
  edm::InputTag mcTruthCollection_;
 
  double maxPt_;
  double maxAbsEta_;
  double deltaR_; 

  //Ele counters
  int n_ele_mc;
  
  int n_ele[6];

  // histos limits and binning
  double etamin;
  double etamax;
  double phimin;
  double phimax;
  double ptmax;
  double pmax;
  double eopmax;
  double eopmaxsht;
  double detamin;
  double detamax;
  double dphimin;
  double dphimax;
  double detamatchmin;
  double detamatchmax;
  double dphimatchmin;
  double dphimatchmax;
  double fhitsmax;
  double lhitsmax;
  double hoemin;
  double hoemax;
  double popmin;
  double popmax;
  double zmassmin;
  double zmassmax;
  double hitsmax;
  double sigmaeemax;
  double sigmappmax;
  double esopoutmax;
  double vertexzmin;
  double vertexzmax;
     
  int nbineta;
  int nbinp;
  int nbinpt;
  int nbinpteff;
  int nbinphi;
  int nbinp2D;
  int nbinpt2D;
  int nbineta2D;
  int nbinphi2D;
  int nbineop;
  int nbineop2D;
  int nbinfhits;
  int nbinlhits;
  int nbinxyz;
  int nbindeta;
  int nbindphi;
  int nbindetamatch;
  int nbindphimatch;
  int nbindetamatch2D;
  int nbindphimatch2D;
  int nbinhoe;
  int nbinpop;
  int nbinzmass;
  int nbinhits;
  int nbinsigmaee;
  int nbinsigmapp;
  int nbinesopout;
};

#endif
