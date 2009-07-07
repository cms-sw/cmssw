#include "Validation/RecoMET/interface/METTester.h"
// author: Mike Schmitt, University of Florida
// first version 8/24/2006
// modification: Bobby Scurlock
// date:  03.11.2006
// note:  added RMS(METx) vs SumET capability
// modification: Rick Cavanaugh
// date:  05.11.2006
// note:  cleaned up constructor and beginJob, removed int conv. warning
//        added configuration params
// modification: Mike Schmitt
// date:  02.28.2007
// note:  code rewrite. Now uses STL map for monitoring element container. 
// modification: Bobby Scurlock
// date:  04.03.2007
// note:  Eliminated automated resolution fitting. This is now done in a ROOT script.

// date:  02.04.2009 
// note:  Added option to use fine binning or course binning for histos

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
//#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/METReco/interface/GenMET.h"
#include "DataFormats/METReco/interface/GenMETCollection.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/METCollection.h"
#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/METReco/interface/PFMETCollection.h"
#include "DataFormats/MuonReco/interface/MuonMETCorrectionData.h"
#include "DataFormats/Common/interface/ValueMap.h"  
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/MuonMETCorrectionData.h"

#include <vector>
#include <utility>
#include <ostream>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <memory>
#include "DQMServices/Core/interface/DQMStore.h"

METTester::METTester(const edm::ParameterSet& iConfig)
{
  inputMETLabel_           = iConfig.getParameter<edm::InputTag>("InputMETLabel");
  METType_                 = iConfig.getUntrackedParameter<std::string>("METType");
  finebinning_             = iConfig.getUntrackedParameter<bool>("FineBinning");
}

//void METTester::beginJob(const edm::EventSetup& iSetup)
//void METTester::beginJob()
void METTester::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup)
{
  // get ahold of back-end interface
  dbe_ = edm::Service<DQMStore>().operator->();
  
  if (dbe_) {
    //    TString dirName = "RecoMETV/METTask/MET/";
    //TString dirName = "JetMET/EventInfo/CertificationSummary/MET_Global/";
    TString dirName = "RecoMETV/MET_Global/";
    TString label(inputMETLabel_.label());
    dirName += label;
    dbe_->setCurrentFolder((string)dirName);
    
    if (METType_ == "CaloMET")
      { 
	// CaloMET Histograms
	if(!finebinning_)
	  {
	    me["hNevents"]                = dbe_->book1D("METTask_Nevents","METTask_Nevents",1,-0.5,1); 
	    me["hCaloMEx"]                = dbe_->book1D("METTask_CaloMEx","METTask_CaloMEx",500,-999.5,499.5); 
	    me["hCaloMEy"]                = dbe_->book1D("METTask_CaloMEy","METTask_CaloMEy",500,-999.5,499.5);
	    //        me["hCaloEz"]                 = dbe_->book1D("METTask_CaloEz","METTask_CaloEz",2001,-500,501);
	    me["hCaloMETSig"]             = dbe_->book1D("METTask_CaloMETSig","METTask_CaloMETSig",25,-0.5,24.5);
	    me["hCaloMET"]                = dbe_->book1D("METTask_CaloMET","METTask_CaloMET",1200,-0.5,1199.5);
	    me["hCaloMETPhi"]             = dbe_->book1D("METTask_CaloMETPhi","METTask_CaloMETPhi",80,-4,4);
	    me["hCaloSumET"]              = dbe_->book1D("METTask_CaloSumET","METTask_CaloSumET",800,-0.5,7999.5);   //10GeV
	    me["hCaloMaxEtInEmTowers"]    = dbe_->book1D("METTask_CaloMaxEtInEmTowers","METTask_CaloMaxEtInEmTowers",600,-0.5,2999.5);   //5GeV
	    me["hCaloMaxEtInHadTowers"]   = dbe_->book1D("METTask_CaloMaxEtInHadTowers","METTask_CaloMaxEtInHadTowers",600,-.05,2999.5);  //5GeV
	    me["hCaloEtFractionHadronic"] = dbe_->book1D("METTask_CaloEtFractionHadronic","METTask_CaloEtFractionHadronic",100,0,1);
	    me["hCaloEmEtFraction"]       = dbe_->book1D("METTask_CaloEmEtFraction","METTask_CaloEmEtFraction",100,0,1);
	    me["hCaloHadEtInHB"]          = dbe_->book1D("METTask_CaloHadEtInHB","METTask_CaloHadEtInHB",1000, -0.5, 4999.5);  //5GeV  
	    me["hCaloHadEtInHO"]          = dbe_->book1D("METTask_CaloHadEtInHO","METTask_CaloHadEtInHO",250, -0.5, 499.5);  //5GeV
	    me["hCaloHadEtInHE"]          = dbe_->book1D("METTask_CaloHadEtInHE","METTask_CaloHadEtInHE",200, -0.5, 399.5);  //5GeV
	    me["hCaloHadEtInHF"]          = dbe_->book1D("METTask_CaloHadEtInHF","METTask_CaloHadEtInHF",100, -0.5, 199.5);  //5GeV
	    me["hCaloEmEtInHF"]           = dbe_->book1D("METTask_CaloEmEtInHF","METTask_CaloEmEtInHF",100, -0.5, 99.5);   //5GeV
	    me["hCaloEmEtInEE"]           = dbe_->book1D("METTask_CaloEmEtInEE","METTask_CaloEmEtInEE",100,0,199.5);    //5GeV
	    me["hCaloEmEtInEB"]           = dbe_->book1D("METTask_CaloEmEtInEB","METTask_CaloEmEtInEB",1200,0, 5999.5);   //5GeV
	  }
	else
	  {
	    //FineBinnning
	    me["hNevents"]                = dbe_->book1D("METTask_Nevents","METTask_Nevents",1,-0.5,1);
	    me["hCaloMEx"]                = dbe_->book1D("METTask_CaloMEx","METTask_CaloMEx",4001,-1000,1001);
	    me["hCaloMEy"]                = dbe_->book1D("METTask_CaloMEy","METTask_CaloMEy",4001,-1000,1001);
	    //me["hCaloEz"]                 = dbe_->book1D("METTask_CaloEz","METTask_CaloEz",2001,-500,501);
	    me["hCaloMETSig"]             = dbe_->book1D("METTask_CaloMETSig","METTask_CaloMETSig",51,0,51);
	    me["hCaloMET"]                = dbe_->book1D("METTask_CaloMET","METTask_CaloMET",2001,0,2001);
	    me["hCaloMETPhi"]             = dbe_->book1D("METTask_CaloMETPhi","METTask_CaloMETPhi",80,-4,4);
	    me["hCaloSumET"]              = dbe_->book1D("METTask_CaloSumET","METTask_CaloSumET",10001,0,10001);
	    me["hCaloMaxEtInEmTowers"]    = dbe_->book1D("METTask_CaloMaxEtInEmTowers","METTask_CaloMaxEtInEmTowers",4001,0,4001);
 	    me["hCaloMaxEtInHadTowers"]   = dbe_->book1D("METTask_CaloMaxEtInHadTowers","METTask_CaloMaxEtInHadTowers",4001,0,4001);
	    me["hCaloEtFractionHadronic"] = dbe_->book1D("METTask_CaloEtFractionHadronic","METTask_CaloEtFractionHadronic",100,0,1);
	    me["hCaloEmEtFraction"]       = dbe_->book1D("METTask_CaloEmEtFraction","METTask_CaloEmEtFraction",100,0,1);
	    me["hCaloHadEtInHB"]          = dbe_->book1D("METTask_CaloHadEtInHB","METTask_CaloHadEtInHB",8001,0,8001);
	    me["hCaloHadEtInHO"]          = dbe_->book1D("METTask_CaloHadEtInHO","METTask_CaloHadEtInHO",4001,0,4001);
	    me["hCaloHadEtInHE"]          = dbe_->book1D("METTask_CaloHadEtInHE","METTask_CaloHadEtInHE",4001,0,4001);
	    me["hCaloHadEtInHF"]          = dbe_->book1D("METTask_CaloHadEtInHF","METTask_CaloHadEtInHF",4001,0,4001);
	    me["hCaloHadEtInEB"]          = dbe_->book1D("METTask_CaloHadEtInEB","METTask_CaloHadEtInEB",8001,0,8001);
	    me["hCaloHadEtInEE"]          = dbe_->book1D("METTask_CaloHadEtInEE","METTask_CaloHadEtInEE",4001,0,4001);
	    me["hCaloEmEtInHF"]           = dbe_->book1D("METTask_CaloEmEtInHF","METTask_CaloEmEtInHF",4001,0,4001);
	    me["hCaloEmEtInEE"]           = dbe_->book1D("METTask_CaloEmEtInEE","METTask_CaloEmEtInEE",4001,0,4001);
	    me["hCaloEmEtInEB"]           = dbe_->book1D("METTask_CaloEmEtInEB","METTask_CaloEmEtInEB",8001,0,8001);
	  }
      }
    
    else if (METType_ == "GenMET")
      {
	// GenMET Histograms
	
	if(!finebinning_)
	  {
	    me["hNevents"]                = dbe_->book1D("METTask_Nevents","METTask_Nevents",1,0,1); 
	    me["hGenMEx"]                 = dbe_->book1D("METTask_GenMEx","METTask_GenMEx",1000,-999.5,999.5);
	    me["hGenMEy"]                 = dbe_->book1D("METTask_GenMEy","METTask_GenMEy",1000,-999.5,999.5);
	    //        me["hGenEz"]                  = dbe_->book1D("METTask_GenEz","METTask_GenEz",2001,-500,501);
	    me["hGenMETSig"]              = dbe_->book1D("METTask_GenMETSig","METTask_GenMETSig",51,0,51);
	    me["hGenMET"]                 = dbe_->book1D("METTask_GenMET","METTask_GenMET", 2000,-0.5,1999.5);
	    me["hGenMETPhi"]              = dbe_->book1D("METTask_GenMETPhi","METTask_GenMETPhi",80,-4,4);
	    me["hGenSumET"]               = dbe_->book1D("METTask_GenSumET","METTask_GenSumET",1000,-0.5,9999.5);
	    me["hGenEmEnergy"]            = dbe_->book1D("METTask_GenEmEnergy","METTask_GenEmEnergy",800,-0.5,3999.5);
	    me["hGenHadEnergy"]           = dbe_->book1D("METTask_GenHadEnergy","METTask_GenHadEnergy",800,-0.5,3999.5);
	    me["hGenInvisibleEnergy"]     = dbe_->book1D("METTask_GenInvisibleEnergy","METTask_GenInvisibleEnergy",800,-0.5,3999.5);
	    me["hGenAuxiliaryEnergy"]     = dbe_->book1D("METTask_GenAuxiliaryEnergy","METTask_GenAuxiliaryEnergy",800,-0.5,3999.5);    
	    
	  }
	else
	  {
	       me["hNevents"]                = dbe_->book1D("METTask_Nevents","METTask_Nevents",1,0,1);
	       me["hGenMEx"]                 = dbe_->book1D("METTask_GenMEx","METTask_GenMEx",4001,-1000,1001);
	       me["hGenMEy"]                 = dbe_->book1D("METTask_GenMEy","METTask_GenMEy",4001,-1000,1001);
	       //me["hGenEz"]                  = dbe_->book1D("METTask_GenEz","METTask_GenEz",2001,-500,501);
	       me["hGenMETSig"]              = dbe_->book1D("METTask_GenMETSig","METTask_GenMETSig",51,0,51);
	       me["hGenMET"]                 = dbe_->book1D("METTask_GenMET","METTask_GenMET",2001,0,2001);
	       me["hGenMETPhi"]              = dbe_->book1D("METTask_GenMETPhi","METTask_GenMETPhi",80,-4,4);
	       me["hGenSumET"]               = dbe_->book1D("METTask_GenSumET","METTask_GenSumET",10001,0,10001);
	       me["hGenEmEnergy"]            = dbe_->book1D("METTask_GenEmEnergy","METTask_GenEmEnergy",4001,0,4001);
	       me["hGenHadEnergy"]           = dbe_->book1D("METTask_GenHadEnergy","METTask_GenHadEnergy",4001,0,4001);
	       me["hGenInvisibleEnergy"]     = dbe_->book1D("METTask_GenInvisibleEnergy","METTask_GenInvisibleEnergy",4001,0,4001);
	       me["hGenAuxiliaryEnergy"]     = dbe_->book1D("METTask_GenAuxiliaryEnergy","METTask_GenAuxiliaryEnergy",4001,0,4001);
	  }
      }
    else if (METType_ == "MET")
      {
	// MET Histograms
	if(!finebinning_)
	  {
	    me["hNevents"]                = dbe_->book1D("METTask_Nevents","METTask_Nevents",1,0,1); 
	    me["hMEx"]                = dbe_->book1D("METTask_MEx","METTask_MEx",1000,-999.5,999.5);
	    me["hMEy"]                = dbe_->book1D("METTask_MEy","METTask_MEy",1000,-999.5,999.5);
	    //me["hEz"]                 = dbe_->book1D("METTask_Ez","METTask_Ez",1000,-999.5,999.5);
	    me["hMETSig"]             = dbe_->book1D("METTask_METSig","METTask_METSig",50,-0.5,49.5);
	    me["hMET"]                = dbe_->book1D("METTask_MET","METTask_MET",2000,-0.5,1999.5);
	    me["hMETPhi"]             = dbe_->book1D("METTask_METPhi","METTask_METPhi",80,-4,4);
	    me["hSumET"]              = dbe_->book1D("METTask_SumET","METTask_SumET",1000,0,9999.5);   
	    
	  }
	else
	  {
	    me["hNevents"]                = dbe_->book1D("METTask_Nevents","METTask_Nevents",1,0,1);
	    me["hMEx"]                = dbe_->book1D("METTask_MEx","METTask_MEx",2001,-500,501);
	    me["hMEy"]                = dbe_->book1D("METTask_MEy","METTask_MEy",2001,-500,501);
	    //me["hEz"]                 = dbe_->book1D("METTask_Ez","METTask_Ez",2001,-500,501);
	    me["hMETSig"]             = dbe_->book1D("METTask_METSig","METTask_METSig",51,0,51);
	    me["hMET"]                = dbe_->book1D("METTask_MET","METTask_MET",2001,0,2001);
	    me["hMETPhi"]             = dbe_->book1D("METTask_METPhi","METTask_METPhi",80,-4,4);
	    me["hSumET"]              = dbe_->book1D("METTask_SumET","METTask_SumET",4001,0,4001);

	  }
      }
    else if (METType_ == "PFMET")
      {
	// PFMET Histograms                                                                                                                  
	if(!finebinning_)
	  {
	    me["hNevents"]                = dbe_->book1D("METTask_Nevents","METTask_Nevents",1,0,1);   
	    me["hMEx"]                = dbe_->book1D("METTask_MEx","METTask_MEx",1000,-999.5,999.5);
	    me["hMEy"]                = dbe_->book1D("METTask_MEy","METTask_MEy",1000,-999.5,999.5);
	    //        me["hEz"]                 = dbe_->book1D("METTask_Ez","METTask_Ez",2001,-500,501);
	    me["hMETSig"]             = dbe_->book1D("METTask_METSig","METTask_METSig",51,0,51);
	    me["hMET"]                = dbe_->book1D("METTask_MET","METTask_MET",2000,-0.5,1999.5);
	    me["hMETPhi"]             = dbe_->book1D("METTask_METPhi","METTask_METPhi",80,-4,4);
	    me["hSumET"]              = dbe_->book1D("METTask_SumET","METTask_SumET",1000,-0.50,9999.5);     
	    
	   }
	else
	  {
	    //FineBin
	    me["hNevents"]                = dbe_->book1D("METTask_Nevents","METTask_Nevents",1,0,1);
	    me["hMEx"]                = dbe_->book1D("METTask_MEx","METTask_MEx",2001,-500,501);
	    me["hMEy"]                = dbe_->book1D("METTask_MEy","METTask_MEy",2001,-500,501);
	    //me["hEz"]                 = dbe_->book1D("METTask_Ez","METTask_Ez",2001,-500,501);
	    me["hMETSig"]             = dbe_->book1D("METTask_METSig","METTask_METSig",51,0,51);
	    me["hMET"]                = dbe_->book1D("METTask_MET","METTask_MET",2001,0,2001);
	    me["hMETPhi"]             = dbe_->book1D("METTask_METPhi","METTask_METPhi",80,-4,4);
	    me["hSumET"]              = dbe_->book1D("METTask_SumET","METTask_SumET",4001,0,4001);
	    
	  }
      }
    else if (METType_ == "TCMET" || inputMETLabel_.label() == "corMetGlobalMuons")
      {
	//TCMET or MuonCorrectedCaloMET Histograms                                                                                                                  
	if(!finebinning_)
	  {
	    me["hNevents"]                = dbe_->book1D("METTask_Nevents","METTask_Nevents",1,0,1);   
	    me["hMEx"]                = dbe_->book1D("METTask_MEx","METTask_MEx",1000,-999.5,999.5);
	    me["hMEy"]                = dbe_->book1D("METTask_MEy","METTask_MEy",1000,-999.5,999.5);
	    me["hMETSig"]             = dbe_->book1D("METTask_METSig","METTask_METSig",51,0,51);
	    me["hMET"]                = dbe_->book1D("METTask_MET","METTask_MET",2000,-0.5,1999.5);
	    me["hMETPhi"]             = dbe_->book1D("METTask_METPhi","METTask_METPhi",80,-4,4);
	    me["hSumET"]              = dbe_->book1D("METTask_SumET","METTask_SumET",1000,-0.50,9999.5);     
	   
	    me["hMExCorrection"]       = dbe_->book1D("METTask_MExCorrection","METTask_MExCorrection", 1000, -500.0,500.0);
	    me["hMEyCorrection"]       = dbe_->book1D("METTask_MEyCorrection","METTask_MEyCorrection", 1000, -500.0,500.0);
	    me["hMuonCorrectionFlag"]      = dbe_->book1D("METTask_CorrectionFlag", "METTask_CorrectionFlag", 5, -0.5, 4.5);
	   }
	else
	  {
	    //FineBin
	    me["hNevents"]            = dbe_->book1D("METTask_Nevents","METTask_Nevents",1,0,1);
	    me["hMEx"]                = dbe_->book1D("METTask_MEx","METTask_MEx",2001,-500,501);
	    me["hMEy"]                = dbe_->book1D("METTask_MEy","METTask_MEy",2001,-500,501);
	    me["hMETSig"]             = dbe_->book1D("METTask_METSig","METTask_METSig",51,0,51);
	    me["hMET"]                = dbe_->book1D("METTask_MET","METTask_MET",2001,0,2001);
	    me["hMETPhi"]             = dbe_->book1D("METTask_METPhi","METTask_METPhi",80,-4,4);
	    me["hSumET"]              = dbe_->book1D("METTask_SumET","METTask_SumET",4001,0,4001);
	    me["hMExCorrection"]      = dbe_->book1D("METTask_MExCorrection","METTask_MExCorrection", 2000, -500.0,500.0);
	    me["hMEyCorrection"]      = dbe_->book1D("METTask_MEyCorrection","METTask_MEyCorrection", 2000, -500.0,500.0);
	    me["hMuonCorrectionFlag"]     = dbe_->book1D("METTask_CorrectionFlag", "METTask_CorrectionFlag", 5, -0.5, 4.5);	  	   
	  }
	
	if(METType_ == "TCMET")
	  {
	    me["hMuonCorrectionFlag"]->setBinLabel(1,"Not Corrected");
	    me["hMuonCorrectionFlag"]->setBinLabel(2,"Global Fit");
	    me["hMuonCorrectionFlag"]->setBinLabel(3,"Tracker Fit");
	    me["hMuonCorrectionFlag"]->setBinLabel(4,"STA Fit");
	    me["hMuonCorrectionFlag"]->setBinLabel(5,"Treated as Pion");
	  }
	else if( inputMETLabel_.label() == "corMetGlobalMuons") 
	  {
	    me["hMuonCorrectionFlag"]->setBinLabel(1,"Not Corrected");
	    me["hMuonCorrectionFlag"]->setBinLabel(2,"Global Fit");
	    me["hMuonCorrectionFlag"]->setBinLabel(3,"Tracker Fit");
	    me["hMuonCorrectionFlag"]->setBinLabel(4,"STA Fit");
	  }
	
      }
    else
      {
	edm::LogInfo("OutputInfo") << " METType not correctly specified!'";// << outputFile_.c_str();
      }
  }

}

void METTester::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  if (METType_ == "CaloMET")
    { 
      const CaloMET *calomet;
      // Get CaloMET
      edm::Handle<CaloMETCollection> calo;
      iEvent.getByLabel(inputMETLabel_, calo);
      if (!calo.isValid()) {
	edm::LogInfo("OutputInfo") << " failed to retrieve data required by MET Task";
	edm::LogInfo("OutputInfo") << " MET Task cannot continue...!";
	return;
      } else {
	const CaloMETCollection *calometcol = calo.product();
	calomet = &(calometcol->front());
      }
      // ==========================================================
      // Reconstructed MET Information
      double caloSumET = calomet->sumEt();
      double caloMETSig = calomet->mEtSig();
      //      double caloEz = calomet->e_longitudinal();
      double caloMET = calomet->pt();
      double caloMEx = calomet->px();
      double caloMEy = calomet->py();
      double caloMETPhi = calomet->phi();
      double caloMaxEtInEMTowers = calomet->maxEtInEmTowers();
      double caloMaxEtInHadTowers = calomet->maxEtInHadTowers();
      double caloEtFractionHadronic = calomet->etFractionHadronic();
      double caloEmEtFraction = calomet->emEtFraction();
      double caloHadEtInHB = calomet->hadEtInHB();
      double caloHadEtInHO = calomet->hadEtInHO();
      double caloHadEtInHE = calomet->hadEtInHE();
      double caloHadEtInHF = calomet->hadEtInHF();
      double caloEmEtInEB = calomet->emEtInEB();
      double caloEmEtInEE = calomet->emEtInEE();
      double caloEmEtInHF = calomet->emEtInHF();

      edm::LogInfo("OutputInfo") << caloMET << " " << caloSumET << endl;
      me["hNevents"]->Fill(0.5);
      me["hCaloMEx"]->Fill(caloMEx);
      me["hCaloMEy"]->Fill(caloMEy);
      me["hCaloMET"]->Fill(caloMET);
      me["hCaloMETPhi"]->Fill(caloMETPhi);
      me["hCaloSumET"]->Fill(caloSumET);
      me["hCaloMETSig"]->Fill(caloMETSig);
      //      me["hCaloEz"]->Fill(caloEz);
      me["hCaloMaxEtInEmTowers"]->Fill(caloMaxEtInEMTowers);
      me["hCaloMaxEtInHadTowers"]->Fill(caloMaxEtInHadTowers);
      me["hCaloEtFractionHadronic"]->Fill(caloEtFractionHadronic);
      me["hCaloEmEtFraction"]->Fill(caloEmEtFraction);
      me["hCaloHadEtInHB"]->Fill(caloHadEtInHB);
      me["hCaloHadEtInHO"]->Fill(caloHadEtInHO);
      me["hCaloHadEtInHE"]->Fill(caloHadEtInHE);
      me["hCaloHadEtInHF"]->Fill(caloHadEtInHF);
      me["hCaloEmEtInEB"]->Fill(caloEmEtInEB);
      me["hCaloEmEtInEE"]->Fill(caloEmEtInEE);
      me["hCaloEmEtInHF"]->Fill(caloEmEtInHF);
    }
  
  else if (METType_ == "GenMET")
    {
      const GenMET *genmet;
      // Get Generated MET
      edm::Handle<GenMETCollection> gen;
      iEvent.getByLabel(inputMETLabel_, gen);
      if (!gen.isValid()) {
	edm::LogInfo("OutputInfo") << " failed to retrieve data required by MET Task";
	edm::LogInfo("OutputInfo") << " MET Task cannot continue...!";
	return;
      } else {
	const GenMETCollection *genmetcol = gen.product();
	genmet = &(genmetcol->front());
      }    
      
      // ==========================================================
      // Genenerated MET Information  
      double genSumET = genmet->sumEt();
      double genMET = genmet->pt();
      double genMEx = genmet->px();
      double genMEy = genmet->py();
      double genMETPhi = genmet->phi();
      double genMETSig = genmet->mEtSig();
      double genEmEnergy = genmet->emEnergy();
      double genHadEnergy = genmet->hadEnergy();
      double genInvisibleEnergy= genmet->invisibleEnergy();
      double genAuxiliaryEnergy= genmet->auxiliaryEnergy();
      
      me["hNevents"]->Fill(0);
      me["hGenMEx"]->Fill(genMEx);
      me["hGenMEy"]->Fill(genMEy);
      me["hGenMET"]->Fill(genMET);
      me["hGenMETPhi"]->Fill(genMETPhi);
      me["hGenSumET"]->Fill(genSumET);
      me["hGenMETSig"]->Fill(genMETSig);
      //me["hGenEz"]->Fill(genEz);
      me["hGenEmEnergy"]->Fill(genEmEnergy);
      me["hGenHadEnergy"]->Fill(genHadEnergy);
      me["hGenInvisibleEnergy"]->Fill(genInvisibleEnergy);
      me["hGenAuxiliaryEnergy"]->Fill(genAuxiliaryEnergy);
      me["hNevents"]->Fill(0.5);
    }
  else if( METType_ == "PFMET")
    {
      const PFMET *pfmet;
      edm::Handle<PFMETCollection> hpfmetcol;
      iEvent.getByLabel(inputMETLabel_,hpfmetcol);
      if(!hpfmetcol.isValid()){
	edm::LogInfo("OutputInfo") << "falied to retrieve data require by MET Task";
	edm::LogInfo("OutputInfo") << "MET Taks cannot continue...!";
	return;
      }
      else
	{
	  const PFMETCollection *pfmetcol = hpfmetcol.product();
	  pfmet = &(pfmetcol->front());
	}
      // Reconstructed MET Information                                                                                                     
      double SumET = pfmet->sumEt();
      double MET = pfmet->pt();
      double MEx = pfmet->px();
      double MEy = pfmet->py();
      double METPhi = pfmet->phi();
      double METSig = pfmet->mEtSig();
      me["hMEx"]->Fill(MEx);
      me["hMEy"]->Fill(MEy);
      me["hMET"]->Fill(MET);
      me["hMETPhi"]->Fill(METPhi);
      me["hSumET"]->Fill(SumET);
      me["hMETSig"]->Fill(METSig);
      me["hNevents"]->Fill(0.5);
    }
  else if (METType_ == "MET")
    {
      const MET *met;
      // Get Generated MET
      edm::Handle<METCollection> hmetcol;
      iEvent.getByLabel(inputMETLabel_, hmetcol);
      if (!hmetcol.isValid()) {
	edm::LogInfo("OutputInfo") << " failed to retrieve data required by MET Task";
	edm::LogInfo("OutputInfo") << " MET Task cannot continue...!";
	return;
      } else {
	const METCollection *metcol = hmetcol.product();
	met = &(metcol->front());
      }   

      // Reconstructed MET Information
      double SumET = met->sumEt();
      double MET = met->pt();
      double MEx = met->px();
      double MEy = met->py();
      double METPhi = met->phi();
      double METSig = met->mEtSig();

      me["hMEx"]->Fill(MEx);
      me["hMEy"]->Fill(MEy);
      me["hMET"]->Fill(MET);
      me["hMETPhi"]->Fill(METPhi);
      me["hSumET"]->Fill(SumET);
      me["hMETSig"]->Fill(METSig);
      me["hNevents"]->Fill(0.5);

    }
  else if( METType_ == "TCMET" )
    {
      const MET *tcMet;
      edm::Handle<METCollection> htcMetcol;
      iEvent.getByLabel(inputMETLabel_,htcMetcol);
      if(!htcMetcol.isValid()){
	edm::LogInfo("OutputInfo") << "falied to retrieve data require by MET Task";
	edm::LogInfo("OutputInfo") << "MET Taks cannot continue...!";
	return;
      }
      else
	{
	  const METCollection *tcMetcol = htcMetcol.product();
	  tcMet = &(tcMetcol->front());
	}

      // Reconstructed TCMET Information                                                                                                     
      double SumET = tcMet->sumEt();
      double MET = tcMet->pt();
      double MEx = tcMet->px();
      double MEy = tcMet->py();
      double METPhi = tcMet->phi();
      double METSig = tcMet->mEtSig();
      me["hMEx"]->Fill(MEx);
      me["hMEy"]->Fill(MEy);
      me["hMET"]->Fill(MET);
      me["hMETPhi"]->Fill(METPhi);
      me["hSumET"]->Fill(SumET);
      me["hMETSig"]->Fill(METSig);
      me["hNevents"]->Fill(0.5);

      edm::Handle< edm::ValueMap<reco::MuonMETCorrectionData> > tcMet_ValueMap_Handle;
      iEvent.getByLabel("muonTCMETValueMapProducer" , "muCorrData", tcMet_ValueMap_Handle);
      
      edm::Handle< reco::MuonCollection > muon_Handle;
      iEvent.getByLabel("muons", muon_Handle);

      const unsigned int nMuons = muon_Handle->size();      
      for( unsigned int mus = 0; mus < nMuons; mus++ ) 
	{
	  reco::MuonRef muref( muon_Handle, mus);
	  reco::MuonMETCorrectionData muCorrData = (*tcMet_ValueMap_Handle)[muref];
	  
	  me["hMExCorrection"] -> Fill(muCorrData.corrY());
	  me["hMEyCorrection"] -> Fill(muCorrData.corrX());
	  me["hMuonCorrectionFlag"]-> Fill(muCorrData.type());
	}

    }
  else if( inputMETLabel_.label() == "corMetGlobalMuons" )
    {
      const CaloMET *corMetGlobalMuons;
      edm::Handle<CaloMETCollection> hcorMetGlobalMuonscol;
      iEvent.getByLabel(inputMETLabel_, hcorMetGlobalMuonscol );
      if(! hcorMetGlobalMuonscol.isValid()){
	edm::LogInfo("OutputInfo") << "falied to retrieve data require by MET Task";
	edm::LogInfo("OutputInfo") << "MET Taks cannot continue...!";
	return;
      }
      else
	{	 
	  const CaloMETCollection *corMetGlobalMuonscol = hcorMetGlobalMuonscol.product();
	  corMetGlobalMuons = &(corMetGlobalMuonscol->front());
	}

      // Reconstructed TCMET Information                                                                                                     
      double SumET = corMetGlobalMuons->sumEt();
      double MET = corMetGlobalMuons->pt();
      double MEx = corMetGlobalMuons->px();
      double MEy = corMetGlobalMuons->py();
      double METPhi = corMetGlobalMuons->phi();
      double METSig = corMetGlobalMuons->mEtSig();
      me["hMEx"]->Fill(MEx);
      me["hMEy"]->Fill(MEy);
      me["hMET"]->Fill(MET);
      me["hMETPhi"]->Fill(METPhi);
      me["hSumET"]->Fill(SumET);
      me["hMETSig"]->Fill(METSig);
      me["hNevents"]->Fill(0.5);

      edm::Handle< edm::ValueMap<reco::MuonMETCorrectionData> > corMetGlobalMuons_ValueMap_Handle;
      iEvent.getByLabel("muonMETValueMapProducer" , "muCorrData", corMetGlobalMuons_ValueMap_Handle);
      
      edm::Handle< reco::MuonCollection > muon_Handle;
      iEvent.getByLabel("muons", muon_Handle);

      const unsigned int nMuons = muon_Handle->size();      
      for( unsigned int mus = 0; mus < nMuons; mus++ ) 
	{
	  reco::MuonRef muref( muon_Handle, mus);
	  reco::MuonMETCorrectionData muCorrData = (*corMetGlobalMuons_ValueMap_Handle)[muref];
	  
	  me["hMExCorrection"] -> Fill(muCorrData.corrY());
	  me["hMEyCorrection"] -> Fill(muCorrData.corrX());
	  me["hMuonCorrectionFlag"]-> Fill(muCorrData.type());
	}

    }

}

void METTester::endJob() 
{
 
}
