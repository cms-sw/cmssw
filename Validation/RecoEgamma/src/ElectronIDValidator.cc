// -*- C++ -*-
//
// Package:    Validation/RecoEgamma
// Class:      ElectronIDValidator
// 
/**\class ElectronIDValidator Validation/RecoEgamma/src/ElectronIDValidator.cc

 Description: GsfElectrons IDanalyzer using MC truth

 Implementation:
     <Notes on implementation>
 */
//
// Original Author:  Leonardo Di Matteo
//         Created:  Tue Sep 9 17:13:06 CEST 2008
//
//

//user include files
#include "Validation/RecoEgamma/interface/ElectronIDValidator.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"

#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "TLorentzVector.h"

#include <iostream>
#include "TMath.h"
#include "TFile.h"
#include "TH1F.h"
#include "TH1I.h"
#include "TH2F.h"
#include "TProfile.h"
#include "TTree.h"

using namespace reco;

ElectronIDValidator::ElectronIDValidator(const edm::ParameterSet& conf)
{
 
 //Set Output files
  
 outputFile_ = conf.getParameter<std::string>("outputFile");
 histfile_   = new TFile(outputFile_.c_str(),"RECREATE");
 
 //Get Input Objects
 
 electronCollection_   = conf.getParameter<std::string>("electronCollection");
 
 electronLabelRobustL_ = conf.getParameter<std::string>("electronLabelRobustL");
 electronLabelRobustT_ = conf.getParameter<std::string>("electronLabelRobustT");
 electronLabelLoose_   = conf.getParameter<std::string>("electronLabelLoose");
 electronLabelTight_   = conf.getParameter<std::string>("electronLabelTight");
 
 reducedBarrelRecHitCollection_ = conf.getParameter<edm::InputTag>("reducedBarrelRecHitCollection");
 reducedEndcapRecHitCollection_ = conf.getParameter<edm::InputTag>("reducedEndcapRecHitCollection");

 mcTruthCollection_    = conf.getParameter<edm::InputTag>("mcTruthCollection");
 
 //Get Histos Parameters
 
 maxPt_  =  conf.getParameter<double>("MaxPt");
 maxAbsEta_  =  conf.getParameter<double>("MaxAbsEta");
 deltaR_  =  conf.getParameter<double>("DeltaR");
 etamin = conf.getParameter<double>("Etamin");
 etamax = conf.getParameter<double>("Etamax");
 phimin = conf.getParameter<double>("Phimin");
 phimax = conf.getParameter<double>("Phimax");
 ptmax = conf.getParameter<double>("Ptmax");
 pmax = conf.getParameter<double>("Pmax");
 eopmax = conf.getParameter<double>("Eopmax");
 eopmaxsht = conf.getParameter<double>("Eopmaxsht");
 detamin = conf.getParameter<double>("Detamin");
 detamax = conf.getParameter<double>("Detamax");
 dphimin = conf.getParameter<double>("Dphimin");
 dphimax = conf.getParameter<double>("Dphimax");
 detamatchmin = conf.getParameter<double>("Detamatchmin");
 detamatchmax = conf.getParameter<double>("Detamatchmax");
 dphimatchmin = conf.getParameter<double>("Dphimatchmin");
 dphimatchmax = conf.getParameter<double>("Dphimatchmax");
 fhitsmax = conf.getParameter<double>("Fhitsmax");
 lhitsmax = conf.getParameter<double>("Lhitsmax");
 hoemin  = conf.getParameter<double>("Dhoemin");
 hoemax  = conf.getParameter<double>("Dhoemax");
 popmin  = conf.getParameter<double>("Dpopmin");
 popmax  = conf.getParameter<double>("Dpopmax");
 zmassmin  = conf.getParameter<double>("Dzmassmin");
 zmassmax  = conf.getParameter<double>("Dzmassmax");
 hitsmax = conf.getParameter<double>("Dhitsmax");
 sigmaeemax = conf.getParameter<double>("Dsigmaeemax");
 sigmappmax = conf.getParameter<double>("Dsigmappmax");
 esopoutmax = conf.getParameter<double>("Desopoutmax");
 vertexzmin = conf.getParameter<double>("Dvertexzmin");
 vertexzmax = conf.getParameter<double>("Dvertexzmax");
 
 nbineta = conf.getParameter<int>("Nbineta");
 nbineta2D = conf.getParameter<int>("Nbineta2D");
 nbinp = conf.getParameter<int>("Nbinp");
 nbinpt = conf.getParameter<int>("Nbinpt");
 nbinp2D = conf.getParameter<int>("Nbinp2D");
 nbinpt2D = conf.getParameter<int>("Nbinpt2D");
 nbinpteff = conf.getParameter<int>("Nbinpteff");
 nbinphi = conf.getParameter<int>("Nbinphi");
 nbinphi2D = conf.getParameter<int>("Nbinphi2D");
 nbineop = conf.getParameter<int>("Nbineop");
 nbineop2D = conf.getParameter<int>("Nbineop2D");
 nbinfhits = conf.getParameter<int>("Nbinfhits");
 nbinlhits = conf.getParameter<int>("Nbinlhits");
 nbinxyz = conf.getParameter<int>("Nbinxyz");
 nbindeta = conf.getParameter<int>("Nbindeta");
 nbindphi = conf.getParameter<int>("Nbindphi");
 nbindetamatch = conf.getParameter<int>("Nbindetamatch");
 nbindphimatch = conf.getParameter<int>("Nbindphimatch");
 nbindetamatch2D = conf.getParameter<int>("Nbindetamatch2D");
 nbindphimatch2D = conf.getParameter<int>("Nbindphimatch2D");
 nbinhoe  =  conf.getParameter<int>("Nbinhoe");
 nbinpop = conf.getParameter<int>("Nbinpop");
 nbinzmass = conf.getParameter<int>("Nbinzmass");
 nbinhits = conf.getParameter<int>("Nbinhits");
 nbinsigmaee = conf.getParameter<int>("Nbinsigmaee");
 nbinsigmapp = conf.getParameter<int>("Nbinsigmapp");
 nbinesopout = conf.getParameter<int>("Nbinesopout");
 //=========================== Ele Classes ==================================  

 // ==== 0 -> Robust-Loose || 1 -> Robust-Tight || 2 -> Loose || 3 -> Tight || 4 -> Matched || 5 -> All ====

 n_class = 6;
  
 v_class.push_back("robustL");
 v_class.push_back("robustT");
 v_class.push_back("loose");
 v_class.push_back("tight");
 v_class.push_back("matched");
 v_class.push_back("all");
      
  //=====================Electron Physical Variables========================
 
 n_eleVar = 23;
 n_eleMatch = 8;
 
  //Only matched ele ==> Plots for MatchedEle, MatchedEle&&EleID
 v_eleVar.push_back("simEta");
 v_eleVar.push_back("simPt");
 v_eleVar.push_back("simPhi");
 v_eleVar.push_back("simAbsEta");
 v_eleVar.push_back("PoPtrue");
 v_eleVar.push_back("PhiMnPhiTrue");
 v_eleVar.push_back("EtaMnEtaTrue");
 v_eleVar.push_back("ZinvMass");
  
  //All reco ==> Plots for MatchedEle, MathcedEle&&EleID, EleID, All RecoEle
 v_eleVar.push_back("charge");
 v_eleVar.push_back("vertexPt");
 v_eleVar.push_back("vertexEta");
 v_eleVar.push_back("vertexPhi");
 v_eleVar.push_back("HoE");
 v_eleVar.push_back("outerPt");
 v_eleVar.push_back("dEtaSc_propVtx");
 v_eleVar.push_back("dPhiSc_propVtx");
 v_eleVar.push_back("dEtaCl_propOut");
 v_eleVar.push_back("dPhiCl_propOut");
 v_eleVar.push_back("fbremMean");
 v_eleVar.push_back("eSeedOverPout");
 v_eleVar.push_back("EoP");
 v_eleVar.push_back("foundHits");
 v_eleVar.push_back("vertexZ");

  //======================Supercluster Physical Variables====================
  
 n_sclVar = 6;
  
 //Ele's SC ==> Plots for MatchedEle, MathcedEle&&EleID, EleID, All RecoEle
 v_sclVar.push_back("sigmaEta");
 v_sclVar.push_back("sigmaPhi");
 v_sclVar.push_back("En");
 v_sclVar.push_back("Eta");
 v_sclVar.push_back("Phi");
 v_sclVar.push_back("eSeedOverPin");

 //==================Ele counters===========================================
 
 n_ele_mc  = 0 ;
 
 for ( int i=0; i < n_class; i++ ) n_ele[i] = 0 ;
  
}

ElectronIDValidator::~ElectronIDValidator(){
  // do anything here that needs to be done at desctruction time => (e.g. close files, deallocate resources etc.)
 histfile_->Write();
 histfile_->Close();
}

void ElectronIDValidator::beginJob(edm::EventSetup const&iSetup){
 
  //==================Histos limits, binning and allocation==================================
 
 for( int i = 0 ; i < n_class; i++ ) {
  
  //Definitions of all Histos limits and binning => For the order see the TString Vector v_eleVar
  
  int hbin_ele[23] = { nbineta, nbinpteff, nbinphi, nbineta/2, nbinpop, nbindphi, nbindeta, nbinzmass, 5, nbinpt, nbineta, nbinphi, nbinhoe, nbinpt, nbindetamatch, nbindphimatch, nbindetamatch, nbindphimatch, 50, nbinesopout, nbineop, nbinhits, nbinxyz } ;
  
  double hmin_ele[23] = { etamin, 5., phimin, 0., popmin, dphimin, detamin, zmassmin, -2., 0., etamin, phimin, hoemin, 0., detamatchmin, dphimatchmin, detamatchmin, dphimatchmin, 0., 0., 0., 0., vertexzmin} ;
  
  double hmax_ele[23] = { etamax, ptmax, phimax, etamax, popmax, dphimax, detamax, zmassmax, 2., ptmax, etamax, phimax, hoemax, ptmax, detamatchmax, dphimatchmax, detamatchmax, dphimatchmax, 1., esopoutmax, eopmax, hitsmax, vertexzmax } ;
  
  int hbin_scl[6]  = { nbinsigmaee, nbinsigmapp, nbinp, nbineta, nbinphi, nbineop } ;
  
  double hmin_scl[6]  = { 0., 0., 0.,  etamin, phimin, 0. } ;
  
  double hmax_scl[6]  = { sigmaeemax, sigmappmax, pmax, etamax, phimax, eopmax } ;

  for ( int j = 0; j < n_eleVar ; j++ ) {
   
   if ( i == n_class && j < n_eleMatch ) continue ;
   
   // ==== Histos of MatchedEle, Matched&&EleID, All Reco. j is the index on the Variables, i the one on the eleClasses  
   h_ele[j][i] = new TH1F ( "h_ele_"+ v_eleVar[j] + "_" + v_class[i], v_eleVar[j] + " " + v_class[i], hbin_ele[j], hmin_ele[j], hmax_ele[j] );
   
   //Barrel
   hB_ele[j][i]  = (TH1F*) h_ele[j][i] -> Clone ( "hB_ele_" + v_eleVar[j] + "_" + v_class[i] + "_barrel" );
   //EndCaps
   hEC_ele[j][i] = (TH1F*) h_ele[j][i] -> Clone ( "hEC_ele_" + v_eleVar[j] + "_" + v_class[i] + "_endcaps" );

   // ==== Histos of RecoEleID, All Reco. j is the index on the Variables, i the one on the eleClasses  
   h_IDele[j][i]   = (TH1F*) h_ele[j][i] -> Clone ( "h_IDele_" + v_eleVar[j] + "_" + v_class[i] );
   hB_IDele[j][i]  = (TH1F*) h_ele[j][i] -> Clone ( "hB_IDele_" + v_eleVar[j] + "_" + v_class[i] + "_barrel" );
   hEC_IDele[j][i] = (TH1F*) h_ele[j][i] -> Clone ( "hEC_IDele_" + v_eleVar[j] + "_" + v_class[i] + "_endcaps" );
   
  }

  
  for ( int j = 0; j < n_sclVar ; j++ ) { 
  
   h_scl[j][i] = new TH1F ( "h_scl_"+ v_sclVar[j] + "_" + v_class[i], v_sclVar[j] + " " + v_class[i], hbin_scl[j], hmin_scl[j], hmax_scl[j] );
  
   hB_scl[j][i]  = (TH1F*) h_scl[j][i]   -> Clone ( "hB_scl_" + v_sclVar[j] + "_" + v_class[i] + "_barrel" );
   hEC_scl[j][i] = (TH1F*) h_scl[j][i]   -> Clone ( "hEC_scl_" + v_sclVar[j] + "_" + v_class[i] + "_endcaps" );
   
   h_IDscl[j][i] = (TH1F*) h_scl[j][i]   -> Clone ( "h_IDscl_" + v_sclVar[j] + "_" + v_class[i] );
   hB_IDscl[j][i]  = (TH1F*) h_scl[j][i] -> Clone ( "hB_IDscl_" + v_sclVar[j] + "_" + v_class[i] + "_barrel" );
   hEC_IDscl[j][i] = (TH1F*) h_scl[j][i] -> Clone ( "hEC_IDscl_" + v_sclVar[j] + "_" + v_class[i] + "_endcaps" );
  
  }
   
 }
 
 histfile_->cd();
 
 // mc truth 
 h_mcNum              = new TH1F( "h_mcNum",              "# mc particles",    nbinfhits,0.,fhitsmax );
 h_eleNum             = new TH1F( "h_mcNum_ele",          "# mc electrons",             nbinfhits,0.,fhitsmax);
 h_gamNum             = new TH1F( "h_mcNum_gam",          "# mc gammas",             nbinfhits,0.,fhitsmax);
 
 h_simEta             = new TH1F( "h_mc_eta",             "mc #eta",           nbineta,etamin,etamax); 
 h_simPt              = new TH1F( "h_mc_Pt",              "mc pt",            nbinpteff,5.,ptmax); 
 h_simPhi             = new TH1F( "h_mc_Phi",             "mc phi",            nbinphi,phimin,phimax); 
 h_simAbsEta          = new TH1F( "h_mc_AbsEta",          "mc absEta",         nbineta/2, 0., etamax );

 hB_simEta             = new TH1F( "hB_mc_eta",             "mc #eta barrel",           nbineta,etamin,etamax); 
 hB_simPt              = new TH1F( "hB_mc_Pt",              "mc pt barrel",            nbinpteff,5.,ptmax); 
 hB_simPhi             = new TH1F( "hB_mc_Phi",             "mc phi barrel",            nbinphi,phimin,phimax); 
 hB_simAbsEta          = new TH1F( "hB_mc_AbsEta",          "mc absEta barrel",         nbineta/2, 0., etamax );

 hEC_simEta             = new TH1F( "hEC_mc_eta",             "mc #eta endcaps",           nbineta,etamin,etamax); 
 hEC_simPt              = new TH1F( "hEC_mc_Pt",              "mc pt endcaps",            nbinpteff,5.,ptmax); 
 hEC_simPhi             = new TH1F( "hEC_mc_Phi",             "mc phi endcaps",            nbinphi,phimin,phimax); 
 hEC_simAbsEta          = new TH1F( "hEC_mc_AbsEta",          "mc absEta endcaps",         nbineta/2, 0., etamax );
    
 //mc histos titles
 h_mcNum              -> GetXaxis()-> SetTitle("# MC particles");
 h_eleNum             -> GetXaxis()-> SetTitle("# MC ele");
 h_gamNum             -> GetXaxis()-> SetTitle("# MC gammas");
 h_simEta             -> GetXaxis()-> SetTitle("true #eta");
 h_simPt              -> GetXaxis()-> SetTitle("true #Pt");
 h_simPhi             -> GetXaxis()-> SetTitle("true #Phi");
 h_simAbsEta          -> GetXaxis()-> SetTitle("true #AbsEta");
 
}

void ElectronIDValidator::endJob(){

 histfile_->cd();
 
 //std::cout << "efficiency calculation " << std::endl; 
  
   
 for ( int i = 0; i < n_class-1; i++ ) {
  
  //Calculate the efficiencies in eta (j=0), pt (j=1), phi(j=2) and |eta|(j=3)
  for ( int j = 0; j< 4; j++ ) {

   h_ele_eff[j][i] =  (TH1F*) h_ele[j][i] -> Clone ( "h_ele_" + v_eleVar[j] + "Eff_" + v_class[i] );
   h_ele_eff[j][i] -> Reset();
   if ( j==0 )    h_ele_eff[j][i] -> Divide( h_ele[j][i] , h_simEta,1,1);
   if ( j==1 )    h_ele_eff[j][i] -> Divide( h_ele[j][i] , h_simPt,1,1);
   if ( j==2 )    h_ele_eff[j][i] -> Divide( h_ele[j][i] , h_simPhi,1,1);
   if ( j==3 )    h_ele_eff[j][i] -> Divide( h_ele[j][i] , h_simAbsEta,1,1);
   h_ele_eff[j][i] -> Print();
   h_ele_eff[j][i] -> GetXaxis() ->SetTitle("#" + v_eleVar[j]);
   h_ele_eff[j][i] -> GetYaxis() -> SetTitle("eff");
      
   h_ele_eff[j][i] -> Write();

   hB_ele_eff[j][i] =  (TH1F*) hB_ele[j][i] -> Clone ( "hB_ele_" + v_eleVar[j] + "Eff_" + v_class[i] );
   hB_ele_eff[j][i] -> Reset();
   if ( j==0 )    hB_ele_eff[j][i] -> Divide( hB_ele[j][i] , hB_simEta,1,1);
   if ( j==1 )    hB_ele_eff[j][i] -> Divide( hB_ele[j][i] , hB_simPt,1,1);
   if ( j==2 )    hB_ele_eff[j][i] -> Divide( hB_ele[j][i] , hB_simPhi,1,1);
   if ( j==3 )    hB_ele_eff[j][i] -> Divide( hB_ele[j][i] , hB_simAbsEta,1,1);
   hB_ele_eff[j][i] -> Print();
   hB_ele_eff[j][i] -> GetXaxis() ->SetTitle("#" + v_eleVar[j] + "barrel" );
   hB_ele_eff[j][i] -> GetYaxis() -> SetTitle("eff");
      
   hB_ele_eff[j][i] -> Write();

   hEC_ele_eff[j][i] =  (TH1F*) hEC_ele[j][i] -> Clone ( "hEC_ele_" + v_eleVar[j] + "Eff_" + v_class[i] );
   hEC_ele_eff[j][i] -> Reset();
   if ( j==0 )    hEC_ele_eff[j][i] -> Divide( hEC_ele[j][i] , hEC_simEta,1,1);
   if ( j==1 )    hEC_ele_eff[j][i] -> Divide( hEC_ele[j][i] , hEC_simPt,1,1);
   if ( j==2 )    hEC_ele_eff[j][i] -> Divide( hEC_ele[j][i] , hEC_simPhi,1,1);
   if ( j==3 )    hEC_ele_eff[j][i] -> Divide( hEC_ele[j][i] , hEC_simAbsEta,1,1);
   hEC_ele_eff[j][i] -> Print();
   hEC_ele_eff[j][i] -> GetXaxis() ->SetTitle("#" + v_eleVar[j] + "endcaps" );
   hEC_ele_eff[j][i] -> GetYaxis() -> SetTitle("eff");
      
   hEC_ele_eff[j][i] -> Write();
      
  }
  
 }
  
 //mc Truth 
 h_mcNum    ->Write();
 h_eleNum   ->Write();
 h_gamNum   ->Write();
 h_simEta   ->Write();         
 h_simPt    ->Write();         
 h_simPhi   ->Write();
 h_simAbsEta->Write();
 
 hB_simEta   ->Write();         
 hB_simPt    ->Write();         
 hB_simPhi   ->Write();
 hB_simAbsEta->Write();
          
 hEC_simEta   ->Write();         
 hEC_simPt    ->Write();         
 hEC_simPhi   ->Write();
 hEC_simAbsEta->Write();

 //Write all the histos
 for ( int i = 0; i < n_class; i++ ) {
  
  for ( int j = 0; j < n_eleVar; j++ ) {

   if ( i == n_class-1 && j < n_eleMatch ) continue ; //For AllReco(i==n_class-1) there are no Histos with MC variables (j<n_eleMatch)
   
   h_ele[j][i] -> Write();
   hB_ele[j][i] -> Write();
   hEC_ele[j][i] -> Write();

   if (  j < n_eleMatch || i > n_class - 3 ) continue ; //For AllReco and MatchedEle (i>n_class-3) don't write histos (already done before). For IDele don't write histos with MC variables (j<n_eleMatch)
   
   h_IDele[j][i] -> Write();
   hB_IDele[j][i] -> Write();
   hEC_IDele[j][i] -> Write();
   
  }
  
  for ( int j = 0; j < n_sclVar; j++ ) {
   
   h_scl[j][i] -> Write();
   hB_scl[j][i] -> Write();
   hEC_scl[j][i] -> Write();

   if ( i == n_class -1 ) continue ; //For AllReco(i==n_class-1) there are no Histos with MC variables (j<n_eleMatch)
   
   h_IDscl[j][i] -> Write();
   hB_IDscl[j][i] -> Write();
   hEC_IDscl[j][i] -> Write();

  }
  
   //Recostruction Efficiency values : Error calculated using binomial statistic
   //  std::cout << "\n\n" + v_class[i] + " reco Electron efficiency recostruction WRT MCele value is --> " << (double)n_ele[i]/(double)n_ele_mc * 100. << "% <-- " << " Error: " << sqrt( (double)n_ele[i]/(double)n_ele_mc * (1 - (double)n_ele[i]/(double)n_ele_mc) / n_ele_mc ) * 100. << std::endl ;
  
  //ElctronID efficiencies (With respect to Matched Ele)
  //  if ( i < 4 ) std::cout << "\n\n" + v_class[i] + " reco Electron efficiency recostruction WRT MatchedEle value is --> " << (double)n_ele[i]/(double)n_ele[4] * 100. << "% <-- " << " Error: " << sqrt( (double)n_ele[i]/(double)n_ele[4] * (1 - (double)n_ele[i]/(double)n_ele[4]) / n_ele[4] ) * 100. << std::endl ;
    
 }

}

void ElectronIDValidator::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
 
 // std::cout << "analyzing new event " << std::endl;

 // get reco electrons  
 edm::Handle<reco::GsfElectronCollection> gsfElectrons;
 iEvent.getByLabel(electronCollection_,gsfElectrons); 
 edm::LogInfo("")<<"\n\n =================> Treating event "<<iEvent.id()<<" Number of electrons "<<gsfElectrons.product()->size();
 
 // get MC truth
 edm::Handle<edm::HepMCProduct> hepMC;
 iEvent.getByLabel(mcTruthCollection_,hepMC);
 
 
  //===============Read the value maps===================
 
 //Robust-Loose
 edm::Handle<edm::ValueMap<float> > RL_eIDValueMap;
 iEvent.getByLabel( electronLabelRobustL_ , RL_eIDValueMap );
 const edm::ValueMap<float> & RobustL_Ele = * RL_eIDValueMap ;

 //Robust-Tight
 edm::Handle<edm::ValueMap<float> > RT_eIDValueMap;
 iEvent.getByLabel( electronLabelRobustT_ , RT_eIDValueMap );
 const edm::ValueMap<float> & RobustT_Ele = * RT_eIDValueMap ;
 
 //Loose
 edm::Handle<edm::ValueMap<float> > L_eIDValueMap;
 iEvent.getByLabel( electronLabelLoose_ , L_eIDValueMap );
 const edm::ValueMap<float> & Loose_Ele = * L_eIDValueMap ;
 
  //Tight
 edm::Handle<edm::ValueMap<float> > T_eIDValueMap;
 iEvent.getByLabel( electronLabelTight_ , T_eIDValueMap );
 const edm::ValueMap<float> & Tight_Ele = * T_eIDValueMap ; 
 
 //=========================================================
 
 // Initialize MC particles
 HepMC::GenParticle* genPc=0;
 const HepMC::GenEvent *myGenEvent = hepMC->GetEvent();
 int mcNum=0, gamNum=0, eleNum=0, eleMatch=0;
 HepMC::FourVector pAssSim;
 
 // Initalize reco 4-P vectors
 TLorentzVector ele4momentum,tot4momentum ;
 tot4momentum.SetPtEtaPhiE( 0, 0, 0, 0 );
 
 //Initialize SC shape tool
 EcalClusterLazyTools lazyTools ( iEvent , iSetup , reducedBarrelRecHitCollection_ , reducedEndcapRecHitCollection_ );

   //Loop on all reco electrons
 for ( reco::GsfElectronCollection::const_iterator gsfIter=gsfElectrons->begin(); gsfIter!=gsfElectrons->end(); gsfIter++) {
  
  n_ele[5]++ ;

   //Find SC shape
  std::vector<float> vCov = lazyTools.covariances(*(gsfIter->superCluster()->seed())) ;
  double sigma_ee = sqrt( vCov[0] );
  double sigma_phiphi = sqrt( vCov[2] );
 
  //Find GSF's SC
  reco::SuperClusterRef sclRef = gsfIter->superCluster();
  
  double eSeed = gsfIter ->superCluster()->seed()->energy();
  double pin  = gsfIter  ->trackMomentumAtVtx().R();   
  double eSeedOverPin = eSeed/pin; 
                
  h_ele[n_eleMatch][5]    -> Fill( gsfIter->charge() );
  h_ele[n_eleMatch+1][5]  -> Fill( gsfIter->pt() );
  h_ele[n_eleMatch+2][5]  -> Fill( gsfIter->eta() );
  h_ele[n_eleMatch+3][5]  -> Fill( gsfIter->phi() );
  h_ele[n_eleMatch+4][5]  -> Fill( gsfIter->hadronicOverEm());
  h_ele[n_eleMatch+5][5]  -> Fill( gsfIter->gsfTrack()->outerMomentum().Rho() );
  h_ele[n_eleMatch+6][5]  -> Fill( gsfIter->deltaEtaSuperClusterTrackAtVtx());
  h_ele[n_eleMatch+7][5]  -> Fill( gsfIter->deltaPhiSuperClusterTrackAtVtx()); 
  h_ele[n_eleMatch+8][5]  -> Fill( gsfIter->deltaEtaSeedClusterTrackAtCalo()); 
  h_ele[n_eleMatch+9][5]  -> Fill( gsfIter->deltaPhiSeedClusterTrackAtCalo()); 
  h_ele[n_eleMatch+10][5] -> Fill( 1 - gsfIter->gsfTrack()->outerMomentum().R()/gsfIter->gsfTrack()->innerMomentum().R() );
  h_ele[n_eleMatch+11][5] -> Fill( gsfIter->eSeedClusterOverPout() );
  h_ele[n_eleMatch+12][5] -> Fill( gsfIter->eSuperClusterOverP() );
  h_ele[n_eleMatch+13][5] -> Fill( gsfIter->gsfTrack()->numberOfValidHits());
  h_ele[n_eleMatch+14][5] -> Fill( gsfIter->vertex().z());
    
  h_scl[0][5]             -> Fill( sigma_ee );
  h_scl[1][5]             -> Fill( sigma_phiphi );
  h_scl[2][5]             -> Fill( sclRef->energy() );
  h_scl[3][5]             -> Fill( sclRef->eta());
  h_scl[4][5]             -> Fill( sclRef->phi());
  h_scl[5][5]             -> Fill( eSeedOverPin );

  //Fill Barrel histos
  if ( fabs(gsfIter->eta()) < 1.5 ) {

   hB_ele[n_eleMatch][5]    -> Fill( gsfIter->charge() );
   hB_ele[n_eleMatch+1][5]  -> Fill( gsfIter->pt() );
   hB_ele[n_eleMatch+2][5]  -> Fill( gsfIter->eta() );
   hB_ele[n_eleMatch+3][5]  -> Fill( gsfIter->phi() );
   hB_ele[n_eleMatch+4][5]  -> Fill( gsfIter->hadronicOverEm());
   hB_ele[n_eleMatch+5][5]  -> Fill( gsfIter->gsfTrack()->outerMomentum().Rho() );
   hB_ele[n_eleMatch+6][5]  -> Fill( gsfIter->deltaEtaSuperClusterTrackAtVtx());
   hB_ele[n_eleMatch+7][5]  -> Fill( gsfIter->deltaPhiSuperClusterTrackAtVtx()); 
   hB_ele[n_eleMatch+8][5]  -> Fill( gsfIter->deltaEtaSeedClusterTrackAtCalo()); 
   hB_ele[n_eleMatch+9][5]  -> Fill( gsfIter->deltaPhiSeedClusterTrackAtCalo()); 
   hB_ele[n_eleMatch+10][5] -> Fill( 1 - gsfIter->gsfTrack()->outerMomentum().R()/gsfIter->gsfTrack()->innerMomentum().R() );
   hB_ele[n_eleMatch+11][5] -> Fill( gsfIter->eSeedClusterOverPout() );
   hB_ele[n_eleMatch+12][5] -> Fill( gsfIter->eSuperClusterOverP() );
   hB_ele[n_eleMatch+13][5] -> Fill( gsfIter->gsfTrack()->numberOfValidHits());
   hB_ele[n_eleMatch+14][5] -> Fill( gsfIter->vertex().z());
    
   hB_scl[0][5]             -> Fill( sigma_ee );
   hB_scl[1][5]             -> Fill( sigma_phiphi );
   hB_scl[2][5]             -> Fill( sclRef->energy() );
   hB_scl[3][5]             -> Fill( sclRef->eta());
   hB_scl[4][5]             -> Fill( sclRef->phi());
   hB_scl[5][5]             -> Fill( eSeedOverPin );

  }

  else {
  
   //Fill Endcaps Histos
   hEC_ele[n_eleMatch][5]    -> Fill( gsfIter->charge() );
   hEC_ele[n_eleMatch+1][5]  -> Fill( gsfIter->pt() );
   hEC_ele[n_eleMatch+2][5]  -> Fill( gsfIter->eta() );
   hEC_ele[n_eleMatch+3][5]  -> Fill( gsfIter->phi() );
   hEC_ele[n_eleMatch+4][5]  -> Fill( gsfIter->hadronicOverEm());
   hEC_ele[n_eleMatch+5][5]  -> Fill( gsfIter->gsfTrack()->outerMomentum().Rho() );
   hEC_ele[n_eleMatch+6][5]  -> Fill( gsfIter->deltaEtaSuperClusterTrackAtVtx());
   hEC_ele[n_eleMatch+7][5]  -> Fill( gsfIter->deltaPhiSuperClusterTrackAtVtx()); 
   hEC_ele[n_eleMatch+8][5]  -> Fill( gsfIter->deltaEtaSeedClusterTrackAtCalo()); 
   hEC_ele[n_eleMatch+9][5]  -> Fill( gsfIter->deltaPhiSeedClusterTrackAtCalo()); 
   hEC_ele[n_eleMatch+10][5] -> Fill( 1 - gsfIter->gsfTrack()->outerMomentum().R()/gsfIter->gsfTrack()->innerMomentum().R() );
   hEC_ele[n_eleMatch+11][5] -> Fill( gsfIter->eSeedClusterOverPout() );
   hEC_ele[n_eleMatch+12][5] -> Fill( gsfIter->eSuperClusterOverP() );
   hEC_ele[n_eleMatch+13][5] -> Fill( gsfIter->gsfTrack()->numberOfValidHits());
   hEC_ele[n_eleMatch+14][5] -> Fill( gsfIter->vertex().z());
 
   hEC_scl[0][5]             -> Fill( sigma_ee - 0.02*(fabs(gsfIter->eta()) - 2.3) );//correct sigmaetaeta dependence on eta in endcap
   hEC_scl[1][5]             -> Fill( sigma_phiphi );
   hEC_scl[2][5]             -> Fill( sclRef->energy() );
   hEC_scl[3][5]             -> Fill( sclRef->eta());
   hEC_scl[4][5]             -> Fill( sclRef->phi());
   hEC_scl[5][5]             -> Fill( eSeedOverPin );
  
  }

            //==============================Electron Classes=========================
     
  int i = 0;
          
  //Find electron position in the reco collection
  i = (int)( gsfIter - gsfElectrons->begin() );
  edm::Ref<reco::GsfElectronCollection> electronRef(gsfElectrons,i);
     
  //Find electron ID
  float eID[4] = { RobustL_Ele[electronRef], RobustT_Ele[electronRef], Loose_Ele[electronRef], Tight_Ele[electronRef] } ;
  
  //Print RecoEle ID                 
  //std::cout << " Reco Electron number " << i <<"  RL " << eID[0] << "  RT " << eID[1] << "  L " << eID[2] << "  T " << eID[3] << std::endl;

  for ( int k = 0; k < n_class-2; k++ ) {
      
   if ( eID[k] > 0.5 ) {
    
    h_IDele[n_eleMatch][k]    -> Fill( gsfIter->charge() );
    h_IDele[n_eleMatch+1][k]  -> Fill( gsfIter->pt() );
    h_IDele[n_eleMatch+2][k]  -> Fill( gsfIter->eta() );
    h_IDele[n_eleMatch+3][k]  -> Fill( gsfIter->phi() );
    h_IDele[n_eleMatch+4][k]  -> Fill( gsfIter->hadronicOverEm());
    h_IDele[n_eleMatch+5][k]  -> Fill( gsfIter->gsfTrack()->outerMomentum().Rho() );
    h_IDele[n_eleMatch+6][k]  -> Fill( gsfIter->deltaEtaSuperClusterTrackAtVtx());
    h_IDele[n_eleMatch+7][k]  -> Fill( gsfIter->deltaPhiSuperClusterTrackAtVtx()); 
    h_IDele[n_eleMatch+8][k]  -> Fill( gsfIter->deltaEtaSeedClusterTrackAtCalo()); 
    h_IDele[n_eleMatch+9][k]  -> Fill( gsfIter->deltaPhiSeedClusterTrackAtCalo()); 
    h_IDele[n_eleMatch+10][k] -> Fill( 1 - gsfIter->gsfTrack()->outerMomentum().R()/gsfIter->gsfTrack()->innerMomentum().R() );
    h_IDele[n_eleMatch+11][k] -> Fill( gsfIter->eSeedClusterOverPout() );
    h_IDele[n_eleMatch+12][k] -> Fill( gsfIter->eSuperClusterOverP() );
    h_IDele[n_eleMatch+13][k] -> Fill( gsfIter->gsfTrack()->numberOfValidHits());
    h_IDele[n_eleMatch+14][k] -> Fill( gsfIter->vertex().z());
       
    h_IDscl[0][k]             -> Fill( sigma_ee );
    h_IDscl[1][k]             -> Fill( sigma_phiphi );
    h_IDscl[2][k]             -> Fill( sclRef->energy() );
    h_IDscl[3][k]             -> Fill( sclRef->eta());
    h_IDscl[4][k]             -> Fill( sclRef->phi());
    h_IDscl[5][k]             -> Fill( eSeedOverPin );

    //Fill Barrel histos
    if ( fabs(gsfIter->eta()) < 1.5 ) {

     hB_IDele[n_eleMatch][k]    -> Fill( gsfIter->charge() );
     hB_IDele[n_eleMatch+1][k]  -> Fill( gsfIter->pt() );
     hB_IDele[n_eleMatch+2][k]  -> Fill( gsfIter->eta() );
     hB_IDele[n_eleMatch+3][k]  -> Fill( gsfIter->phi() );
     hB_IDele[n_eleMatch+4][k]  -> Fill( gsfIter->hadronicOverEm());
     hB_IDele[n_eleMatch+5][k]  -> Fill( gsfIter->gsfTrack()->outerMomentum().Rho() );
     hB_IDele[n_eleMatch+6][k]  -> Fill( gsfIter->deltaEtaSuperClusterTrackAtVtx());
     hB_IDele[n_eleMatch+7][k]  -> Fill( gsfIter->deltaPhiSuperClusterTrackAtVtx()); 
     hB_IDele[n_eleMatch+8][k]  -> Fill( gsfIter->deltaEtaSeedClusterTrackAtCalo()); 
     hB_IDele[n_eleMatch+9][k]  -> Fill( gsfIter->deltaPhiSeedClusterTrackAtCalo()); 
     hB_IDele[n_eleMatch+10][k] -> Fill( 1 - gsfIter->gsfTrack()->outerMomentum().R()/gsfIter->gsfTrack()->innerMomentum().R() );
     hB_IDele[n_eleMatch+11][k] -> Fill( gsfIter->eSeedClusterOverPout() );
     hB_IDele[n_eleMatch+12][k] -> Fill( gsfIter->eSuperClusterOverP() );
     hB_IDele[n_eleMatch+13][k] -> Fill( gsfIter->gsfTrack()->numberOfValidHits());
     hB_IDele[n_eleMatch+14][k] -> Fill( gsfIter->vertex().z());
    
     hB_IDscl[0][k]             -> Fill( sigma_ee );
     hB_IDscl[1][k]             -> Fill( sigma_phiphi );
     hB_IDscl[2][k]             -> Fill( sclRef->energy() );
     hB_IDscl[3][k]             -> Fill( sclRef->eta());
     hB_IDscl[4][k]             -> Fill( sclRef->phi());
     hB_IDscl[5][k]             -> Fill( eSeedOverPin );

    }

    else {
  
     //Fill Endcaps Histos
     hEC_IDele[n_eleMatch][k]    -> Fill( gsfIter->charge() );
     hEC_IDele[n_eleMatch+1][k]  -> Fill( gsfIter->pt() );
     hEC_IDele[n_eleMatch+2][k]  -> Fill( gsfIter->eta() );
     hEC_IDele[n_eleMatch+3][k]  -> Fill( gsfIter->phi() );
     hEC_IDele[n_eleMatch+4][k]  -> Fill( gsfIter->hadronicOverEm());
     hEC_IDele[n_eleMatch+5][k]  -> Fill( gsfIter->gsfTrack()->outerMomentum().Rho() );
     hEC_IDele[n_eleMatch+6][k]  -> Fill( gsfIter->deltaEtaSuperClusterTrackAtVtx());
     hEC_IDele[n_eleMatch+7][k]  -> Fill( gsfIter->deltaPhiSuperClusterTrackAtVtx()); 
     hEC_IDele[n_eleMatch+8][k]  -> Fill( gsfIter->deltaEtaSeedClusterTrackAtCalo()); 
     hEC_IDele[n_eleMatch+9][k]  -> Fill( gsfIter->deltaPhiSeedClusterTrackAtCalo()); 
     hEC_IDele[n_eleMatch+10][k] -> Fill( 1 - gsfIter->gsfTrack()->outerMomentum().R()/gsfIter->gsfTrack()->innerMomentum().R() );
     hEC_IDele[n_eleMatch+11][k] -> Fill( gsfIter->eSeedClusterOverPout() );
     hEC_IDele[n_eleMatch+12][k] -> Fill( gsfIter->eSuperClusterOverP() );
     hEC_IDele[n_eleMatch+13][k] -> Fill( gsfIter->gsfTrack()->numberOfValidHits());
     hEC_IDele[n_eleMatch+14][k] -> Fill( gsfIter->vertex().z());

     hEC_IDscl[0][k]             -> Fill( sigma_ee - 0.02*(fabs(gsfIter->eta()) - 2.3) );//correct sigmaetaeta dependence on eta in endcap
     hEC_IDscl[1][k]             -> Fill( sigma_phiphi );
     hEC_IDscl[2][k]             -> Fill( sclRef->energy() );
     hEC_IDscl[3][k]             -> Fill( sclRef->eta());
     hEC_IDscl[4][k]             -> Fill( sclRef->phi());
     hEC_IDscl[5][k]             -> Fill( eSeedOverPin );
          
    }   
   
   }
  
  }
  
 }
      

 //Loop on MC particles 
 for ( HepMC::GenEvent::particle_const_iterator mcIter=myGenEvent->particles_begin(); mcIter != myGenEvent->particles_end(); mcIter++ ) {
    
  // number of mc particles
  mcNum++;
  
  // counts photons
  if ((*mcIter)->pdg_id() == 22 ){ gamNum++; }       

  // select electrons
  if ( (*mcIter)->pdg_id() == 11 || (*mcIter)->pdg_id() == -11 ){       

   // single primary electrons or electrons from Zs or Ws
   HepMC::GenParticle* mother = 0;
   if ( (*mcIter)->production_vertex() )  {
    if ( (*mcIter)->production_vertex()->particles_begin(HepMC::parents) != 
           (*mcIter)->production_vertex()->particles_end(HepMC::parents))  
     mother = *((*mcIter)->production_vertex()->particles_begin(HepMC::parents));
   } 
   if ( ((mother == 0) || ((mother != 0) && (mother->pdg_id() == 23))
          || ((mother != 0) && (mother->pdg_id() == 32))
          || ((mother != 0) && (fabs(mother->pdg_id()) == 24)))) {       
   
    genPc=(*mcIter);
    pAssSim = genPc->momentum();

    if (pAssSim.perp() < 5. || fabs(pAssSim.eta())> maxAbsEta_) continue;
       
    eleNum++;
    n_ele_mc++;
 
    //Fill MC Histos
    h_simEta   -> Fill( pAssSim.eta() );
    h_simPt    -> Fill( pAssSim.perp() );          
    h_simPhi   -> Fill( pAssSim.phi() );          
    h_simAbsEta-> Fill( fabs(pAssSim.eta()) );
    
    //Barrel
    if ( fabs(pAssSim.eta()) < 1.5 ) {
    
     hB_simEta   -> Fill( pAssSim.eta() );
     hB_simPt    -> Fill( pAssSim.perp() );          
     hB_simPhi   -> Fill( pAssSim.phi() );          
     hB_simAbsEta-> Fill( fabs(pAssSim.eta()) );
      
    }
    
    else {

     //EndCaps
     hEC_simEta   -> Fill( pAssSim.eta() );
     hEC_simPt    -> Fill( pAssSim.perp() );          
     hEC_simPhi   -> Fill( pAssSim.phi() );          
     hEC_simAbsEta-> Fill( fabs(pAssSim.eta()) );
               
    }
      
    // looking for the best matching gsf electron
    bool okGsfFound = false;
    double gsfOkRatio = 999999.;

      // find best matched electron
    reco::GsfElectron bestGsfElectron;
    reco::GsfElectronCollection::const_iterator bestGsfIter;
    for (reco::GsfElectronCollection::const_iterator gsfIter=gsfElectrons->begin();
         gsfIter!=gsfElectrons->end(); gsfIter++){
	
          float deltaphi=fabs( gsfIter->phi()-pAssSim.phi() );
          if(deltaphi>6.283185308) deltaphi -= 6.283185308;
          if(deltaphi>3.141592654) deltaphi = 6.283185308-deltaphi;

          double deltaR = sqrt(pow((gsfIter->eta()-pAssSim.eta()),2)) + deltaphi*deltaphi ;
          if ( deltaR < deltaR_ ){
           if ( (genPc->pdg_id() == 11) && (gsfIter->charge() < 0.) || (genPc->pdg_id() == -11) &&
                 (gsfIter->charge() > 0.) ){
            double tmpGsfRatio = gsfIter->p()/pAssSim.t();
            if ( fabs(tmpGsfRatio-1) < fabs(gsfOkRatio-1) ) {
             gsfOkRatio = tmpGsfRatio;
             bestGsfElectron = *gsfIter;
             bestGsfIter = gsfIter;
             okGsfFound = true;
            } 
                 } 
          } 
         } // loop over rec ele to look for the best one	

         if (okGsfFound){
                    
          //Invariant mass calculation 
          eleMatch++;
          ele4momentum.SetPtEtaPhiE( bestGsfElectron.pt(), bestGsfElectron.eta(), bestGsfElectron.phi(), bestGsfElectron.energy()); 
          tot4momentum = tot4momentum + ele4momentum ;
          double ZinvMass = tot4momentum.M() ;
          
          //Find SC shape
          std::vector<float> vCov = lazyTools.covariances(*(bestGsfElectron.superCluster()->seed())) ;
          double sigma_ee = sqrt( vCov[0] );
          double sigma_phiphi = sqrt( vCov[2] );
          
          //Find GSF's SC
          reco::SuperClusterRef sclRef = bestGsfElectron.superCluster();
          
          double eSeed = bestGsfElectron.superCluster()->seed()->energy();
          double pin  = bestGsfElectron.trackMomentumAtVtx().R();   
          double eSeedOverPin = eSeed/pin; 

          //==============================Electron Classes=========================
     
          int i = 0;
          //Find electron position in the reco collection
          i = (int)( bestGsfIter - gsfElectrons->begin() );
          edm::Ref<reco::GsfElectronCollection> electronRef(gsfElectrons,i);
     
          //Find electron ID
          float eID[5] = { RobustL_Ele[electronRef], RobustT_Ele[electronRef], Loose_Ele[electronRef], Tight_Ele[electronRef], 1. } ;
                   
          //Print MatchedEle ID
          //std::cout << "Matched Electron number " << i <<"  RL " << eID[0] << "  RT " << eID[1] << "  L " << eID[2] << "  T " << eID[3] << " M " << eID[4] << std::endl;
          
          for ( int k = 0; k < n_class-1; k++ ) {
      
           if ( eID[k] > 0.5 ) {
                          
            n_ele[k]++ ;
                                                
            h_ele[0][k]     -> Fill( pAssSim.eta() );
            h_ele[1][k]     -> Fill( pAssSim.perp() );
            h_ele[2][k]     -> Fill( pAssSim.phi() );
            h_ele[3][k]     -> Fill( fabs(pAssSim.eta()) );
            h_ele[4][k]     -> Fill( bestGsfElectron.p()/pAssSim.t());
            h_ele[5][k]     -> Fill( bestGsfElectron.phi()-pAssSim.phi());
            h_ele[6][k]     -> Fill( bestGsfElectron.eta()-pAssSim.eta());
            if ( eleMatch ==2 ) h_ele[7][k] -> Fill( ZinvMass );   
            
            h_ele[n_eleMatch][k]    -> Fill( bestGsfElectron.charge() );
            h_ele[n_eleMatch+1][k]  -> Fill( bestGsfElectron.pt() );
            h_ele[n_eleMatch+2][k]  -> Fill( bestGsfElectron.eta() );
            h_ele[n_eleMatch+3][k]  -> Fill( bestGsfElectron.phi() );
            h_ele[n_eleMatch+4][k]  -> Fill( bestGsfElectron.hadronicOverEm());
            h_ele[n_eleMatch+5][k]  -> Fill( bestGsfElectron.gsfTrack()->outerMomentum().Rho() );
            h_ele[n_eleMatch+6][k]  -> Fill( bestGsfElectron.deltaEtaSuperClusterTrackAtVtx());
            h_ele[n_eleMatch+7][k]  -> Fill( bestGsfElectron.deltaPhiSuperClusterTrackAtVtx()); 
            h_ele[n_eleMatch+8][k]  -> Fill( bestGsfElectron.deltaEtaSeedClusterTrackAtCalo()); 
            h_ele[n_eleMatch+9][k]  -> Fill( bestGsfElectron.deltaPhiSeedClusterTrackAtCalo()); 
            h_ele[n_eleMatch+10][k] -> Fill( 1 - bestGsfElectron.gsfTrack()->outerMomentum().R()/bestGsfElectron.gsfTrack()->innerMomentum().R() );
            h_ele[n_eleMatch+11][k] -> Fill( bestGsfElectron.eSeedClusterOverPout() );
            h_ele[n_eleMatch+12][k] -> Fill( bestGsfElectron.eSuperClusterOverP() );
            h_ele[n_eleMatch+13][k] -> Fill( bestGsfElectron.gsfTrack()->numberOfValidHits());
            h_ele[n_eleMatch+14][k] -> Fill( bestGsfElectron.vertex().z() );

            h_scl[0][k]     -> Fill( sigma_ee );
            h_scl[1][k]     -> Fill( sigma_phiphi );
            h_scl[2][k]     -> Fill( sclRef->energy() );
            h_scl[3][k]     -> Fill( sclRef->eta());
            h_scl[4][k]     -> Fill( sclRef->phi());
            h_scl[5][k]     -> Fill( eSeedOverPin );

           
           //Fill Barrel Histos
            if ( fabs(bestGsfElectron.eta()) < 1.5 ) {
             
             hB_ele[0][k]     -> Fill( pAssSim.eta() );
             hB_ele[1][k]     -> Fill( pAssSim.perp() );
             hB_ele[2][k]     -> Fill( pAssSim.phi() );
             hB_ele[3][k]     -> Fill( fabs(pAssSim.eta()) );
             hB_ele[4][k]     -> Fill( bestGsfElectron.p()/pAssSim.t());
             hB_ele[5][k]     -> Fill( bestGsfElectron.phi()-pAssSim.phi());
             hB_ele[6][k]     -> Fill( bestGsfElectron.eta()-pAssSim.eta());
             if ( eleMatch ==2 ) hB_ele[7][k] -> Fill( ZinvMass );   
            
             hB_ele[n_eleMatch][k]    -> Fill( bestGsfElectron.charge() );
             hB_ele[n_eleMatch+1][k]  -> Fill( bestGsfElectron.pt() );
             hB_ele[n_eleMatch+2][k]  -> Fill( bestGsfElectron.eta() );
             hB_ele[n_eleMatch+3][k]  -> Fill( bestGsfElectron.phi() );
             hB_ele[n_eleMatch+4][k]  -> Fill( bestGsfElectron.hadronicOverEm());
             hB_ele[n_eleMatch+5][k]  -> Fill( bestGsfElectron.gsfTrack()->outerMomentum().Rho() );
             hB_ele[n_eleMatch+6][k]  -> Fill( bestGsfElectron.deltaEtaSuperClusterTrackAtVtx());
             hB_ele[n_eleMatch+7][k]  -> Fill( bestGsfElectron.deltaPhiSuperClusterTrackAtVtx()); 
             hB_ele[n_eleMatch+8][k]  -> Fill( bestGsfElectron.deltaEtaSeedClusterTrackAtCalo()); 
             hB_ele[n_eleMatch+9][k]  -> Fill( bestGsfElectron.deltaPhiSeedClusterTrackAtCalo()); 
             hB_ele[n_eleMatch+10][k] -> Fill( 1 - bestGsfElectron.gsfTrack()->outerMomentum().R()/bestGsfElectron.gsfTrack()->innerMomentum().R() );
             hB_ele[n_eleMatch+11][k] -> Fill( bestGsfElectron.eSeedClusterOverPout() );
             hB_ele[n_eleMatch+12][k] -> Fill( bestGsfElectron.eSuperClusterOverP() );
             hB_ele[n_eleMatch+13][k] -> Fill( bestGsfElectron.gsfTrack()->numberOfValidHits());
             hB_ele[n_eleMatch+14][k] -> Fill( bestGsfElectron.vertex().z() );
             
             hB_scl[0][k]     -> Fill( sigma_ee );
             hB_scl[1][k]     -> Fill( sigma_phiphi );
             hB_scl[2][k]     -> Fill( sclRef->energy() );
             hB_scl[3][k]     -> Fill( sclRef->eta());
             hB_scl[4][k]     -> Fill( sclRef->phi());
             hB_scl[5][k]     -> Fill( eSeedOverPin );
            
            }

            else {
             
             hEC_ele[0][k]     -> Fill( pAssSim.eta() );
             hEC_ele[1][k]     -> Fill( pAssSim.perp() );
             hEC_ele[2][k]     -> Fill( pAssSim.phi() );
             hEC_ele[3][k]     -> Fill( fabs(pAssSim.eta()) );
             hEC_ele[4][k]     -> Fill( bestGsfElectron.p()/pAssSim.t());
             hEC_ele[5][k]     -> Fill( bestGsfElectron.phi()-pAssSim.phi());
             hEC_ele[6][k]     -> Fill( bestGsfElectron.eta()-pAssSim.eta());
             if ( eleMatch ==2 ) hEC_ele[7][k] -> Fill( ZinvMass );   
            
             hEC_ele[n_eleMatch][k]    -> Fill( bestGsfElectron.charge() );
             hEC_ele[n_eleMatch+1][k]  -> Fill( bestGsfElectron.pt() );
             hEC_ele[n_eleMatch+2][k]  -> Fill( bestGsfElectron.eta() );
             hEC_ele[n_eleMatch+3][k]  -> Fill( bestGsfElectron.phi() );
             hEC_ele[n_eleMatch+4][k]  -> Fill( bestGsfElectron.hadronicOverEm());
             hEC_ele[n_eleMatch+5][k]  -> Fill( bestGsfElectron.gsfTrack()->outerMomentum().Rho() );
             hEC_ele[n_eleMatch+6][k]  -> Fill( bestGsfElectron.deltaEtaSuperClusterTrackAtVtx());
             hEC_ele[n_eleMatch+7][k]  -> Fill( bestGsfElectron.deltaPhiSuperClusterTrackAtVtx()); 
             hEC_ele[n_eleMatch+8][k]  -> Fill( bestGsfElectron.deltaEtaSeedClusterTrackAtCalo()); 
             hEC_ele[n_eleMatch+9][k]  -> Fill( bestGsfElectron.deltaPhiSeedClusterTrackAtCalo()); 
             hEC_ele[n_eleMatch+10][k] -> Fill( 1 - bestGsfElectron.gsfTrack()->outerMomentum().R()/bestGsfElectron.gsfTrack()->innerMomentum().R() );
             hEC_ele[n_eleMatch+11][k] -> Fill( bestGsfElectron.eSeedClusterOverPout() );
             hEC_ele[n_eleMatch+12][k] -> Fill( bestGsfElectron.eSuperClusterOverP() );
             hEC_ele[n_eleMatch+13][k] -> Fill( bestGsfElectron.gsfTrack()->numberOfValidHits());
             hEC_ele[n_eleMatch+14][k] -> Fill( bestGsfElectron.vertex().z() );
             
             hEC_scl[0][k]     -> Fill( sigma_ee - 0.02*(fabs(bestGsfElectron.eta()) - 2.3) );//correct sigmaetaeta dependence on eta in endcap
             hEC_scl[1][k]     -> Fill( sigma_phiphi );
             hEC_scl[2][k]     -> Fill( sclRef->energy() );
             hEC_scl[3][k]     -> Fill( sclRef->eta());
             hEC_scl[4][k]     -> Fill( sclRef->phi());
             hEC_scl[5][k]     -> Fill( eSeedOverPin );
           
            }
           
           }
                      
          }
 
         }

 
         h_mcNum  -> Fill( mcNum );
         h_eleNum -> Fill( eleNum );
         h_gamNum -> Fill( gamNum );

          }
  }

 }

}



