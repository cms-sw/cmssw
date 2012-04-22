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
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/METReco/interface/GenMET.h"
#include "DataFormats/METReco/interface/GenMETCollection.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/METFwd.h"
#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/METReco/interface/PFMETCollection.h"
#include "DataFormats/MuonReco/interface/MuonMETCorrectionData.h"
#include "DataFormats/Common/interface/ValueMap.h"  
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/MuonMETCorrectionData.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include <vector>
#include <utility>
#include <ostream>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <memory>
#include "DQMServices/Core/interface/DQMStore.h"
#include "TMath.h"

METTester::METTester(const edm::ParameterSet& iConfig)
{
  METType_                 = iConfig.getUntrackedParameter<std::string>("METType");
  inputMETLabel_           = iConfig.getParameter<edm::InputTag>("InputMETLabel");
  if(METType_ == "TCMET") {
       inputCaloMETLabel_       = iConfig.getParameter<edm::InputTag>("InputCaloMETLabel");     
  inputTrackLabel_         = iConfig.getParameter<edm::InputTag>("InputTrackLabel");    
  inputMuonLabel_          = iConfig.getParameter<edm::InputTag>("InputMuonLabel");
  inputElectronLabel_      = iConfig.getParameter<edm::InputTag>("InputElectronLabel");
  inputBeamSpotLabel_      = iConfig.getParameter<edm::InputTag>("InputBeamSpotLabel");
  minhits_                 = iConfig.getParameter<int>("minhits");
  maxd0_                   = iConfig.getParameter<double>("maxd0");
  maxchi2_                 = iConfig.getParameter<double>("maxchi2");
  maxeta_                  = iConfig.getParameter<double>("maxeta");
  maxpt_                   = iConfig.getParameter<double>("maxpt");
  maxPtErr_                = iConfig.getParameter<double>("maxPtErr");
  trkQuality_              = iConfig.getParameter<std::vector<int> >("trkQuality");
  trkAlgos_                = iConfig.getParameter<std::vector<int> >("trkAlgos");
  }
  finebinning_             = iConfig.getUntrackedParameter<bool>("FineBinning");
  FolderName_              = iConfig.getUntrackedParameter<std::string>("FolderName");
}


void METTester::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup)
{
  // get ahold of back-end interface
  dbe_ = edm::Service<DQMStore>().operator->();
  
  if (dbe_) {
    //    TString dirName = "RecoMETV/METTask/MET/";
    //TString dirName = "JetMET/EventInfo/CertificationSummary/MET_Global/";
    //    TString dirName = "RecoMETV/MET_Global/";
    TString dirName(FolderName_.c_str()); 
    TString label(inputMETLabel_.label());
    dirName += label;
    dbe_->setCurrentFolder((std::string)dirName);
    
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
	    me["hCaloMET"]                = dbe_->book1D("METTask_CaloMET","METTask_CaloMET",1000,-0.5,1999.5);
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


	    
	    me["hCaloMETResolution_GenMETTrue"]      = dbe_->book1D("METTask_CaloMETResolution_GenMETTrue","METTask_CaloMETResolution_GenMETTrue", 500,-500,500); 
	    me["hCaloMETResolution_GenMETCalo"]      = dbe_->book1D("METTask_CaloMETResolution_GenMETCalo","METTask_CaloMETResolution_GenMETCalo", 500,-500,500); 

	    me["hCaloMETPhiResolution_GenMETTrue"] = dbe_->book1D("METTask_CaloMETPhiResolution_GenMETTrue","METTask_CaloMETPhiResolution_GenMETTrue", 80,0,4); 
	    me["hCaloMETPhiResolution_GenMETCalo"] = dbe_->book1D("METTask_CaloMETPhiResolution_GenMETCalo","METTask_CaloMETPhiResolution_GenMETCalo", 80,0,4); 
	    
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

	    me["hCaloMETResolution_GenMETTrue"]      = dbe_->book1D("METTask_CaloMETResolution_GenMETTrue","METTask_CaloMETResolution_GenMETTrue", 2000,-1000,1000); 
	    me["hCaloMETResolution_GenMETCalo"]      = dbe_->book1D("METTask_CaloMETResolution_GenMETCalo","METTask_CaloMETResolution_GenMETCalo", 2000,-1000,1000); 
	    
	    me["hCaloMETPhiResolution_GenMETTrue"] = dbe_->book1D("METTask_CaloMETPhiResolution_GenMETTrue","METTask_CaloMETPhiResolution_GenMETTrue", 80,0,4); 
	    me["hCaloMETPhiResolution_GenMETCalo"] = dbe_->book1D("METTask_CaloMETPhiResolution_GenMETCalo","METTask_CaloMETPhiResolution_GenMETCalo", 80,0,4); 
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

	    me["hNeutralEMEtFraction"]    = dbe_->book1D("METTask_GenNeutralEMEtFraction", "METTask_GenNeutralEMEtFraction", 120, 0.0, 1.2 );
	    me["hNeutralHadEtFraction"]   = dbe_->book1D("METTask_GenNeutralHadEtFraction", "METTask_GenNeutralHadEtFraction", 120, 0.0, 1.2 );
	    me["hChargedEMEtFraction"]    = dbe_->book1D("METTask_GenChargedEMEtFraction", "METTask_GenChargedEMEtFraction", 120, 0.0, 1.2);
	    me["hChargedHadEtFraction"]   = dbe_->book1D("METTask_GenChargedHadEtFraction", "METTask_GenChargedHadEtFraction", 120, 0.0,1.2);
	    me["hMuonEtFraction"]         = dbe_->book1D("METTask_GenMuonEtFraction", "METTask_GenMuonEtFraction", 120, 0.0, 1.2 );
	    me["hInvisibleEtFraction"]    = dbe_->book1D("METTask_GenInvisibleEtFraction", "METTask_GenInvisibleEtFraction", 120, 0.0, 1.2 );
	    
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
	       me["hNeutralEMEtFraction"]    = dbe_->book1D("METTask_GenNeutralEMEtFraction", "METTask_GenNeutralEMEtFraction", 120, 0.0, 1.2 );
	       me["hNeutralHadEtFraction"]   = dbe_->book1D("METTask_GenNeutralHadEtFraction", "METTask_GenNeutralHadEtFraction", 120, 0.0, 1.2 );
	       me["hChargedEMEtFraction"]    = dbe_->book1D("METTask_GenChargedEMEtFraction", "METTask_GenChargedEMEtFraction", 120, 0.0, 1.2);
	       me["hChargedHadEtFraction"]   = dbe_->book1D("METTask_GenChargedHadEtFraction", "METTask_GenChargedHadEtFraction", 120, 0.0,1.2);
	       me["hMuonEtFraction"]         = dbe_->book1D("METTask_GenMuonEtFraction", "METTask_GenMuonEtFraction", 120, 0.0, 1.2 );
	       me["hInvisibleEtFraction"]    = dbe_->book1D("METTask_GenInvisibleEtFraction", "METTask_GenInvisibleEtFraction", 120, 0.0, 1.2 );

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
	    me["hMET"]                = dbe_->book1D("METTask_MET","METTask_MET",1000,-0.5,1999.5);
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
	    me["hMET"]                = dbe_->book1D("METTask_MET","METTask_MET",1000,-0.5,1999.5);
	    me["hMETPhi"]             = dbe_->book1D("METTask_METPhi","METTask_METPhi",80,-4,4);
	    me["hSumET"]              = dbe_->book1D("METTask_SumET","METTask_SumET",1000,-0.50,9999.5);     
	    
	    me["hMETResolution_GenMETTrue"]      = dbe_->book1D("METTask_METResolution_GenMETTrue","METTask_METResolution_GenMETTrue", 500,-500,500); 
	    me["hMETResolution_GenMETCalo"]      = dbe_->book1D("METTask_METResolution_GenMETCalo","METTask_METResolution_GenMETCalo", 500,-500,500); 

	    me["hMETPhiResolution_GenMETTrue"] = dbe_->book1D("METTask_METPhiResolution_GenMETTrue","METTask_METPhiResolution_GenMETTrue", 80,0,4); 
	    me["hMETPhiResolution_GenMETCalo"] = dbe_->book1D("METTask_METPhiResolution_GenMETCalo","METTask_METPhiResolution_GenMETCalo", 80,0,4); 

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

	    me["hMETResolution_GenMETTrue"]      = dbe_->book1D("METTask_METResolution_GenMETTrue","METTask_METResolution_GenMETTrue",2000,-1000,1000);
	    me["hMETResolution_GenMETCalo"]      = dbe_->book1D("METTask_METResolution_GenMETCalo","METTask_METResolution_GenMETCalo",2000,-1000,1000);

	    me["hMETPhiResolution_GenMETTrue"] = dbe_->book1D("METTask_METPhiResolution_GenMETTrue","METTask_METPhiResolution_GenMETTrue", 80,0,4); 
	    me["hMETPhiResolution_GenMETCalo"] = dbe_->book1D("METTask_METPhiResolution_GenMETCalo","METTask_METPhiResolution_GenMETCalo", 80,0,4); 

	    
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
	    me["hMET"]                = dbe_->book1D("METTask_MET","METTask_MET",1000,-0.5,1999.5);
	    me["hMETPhi"]             = dbe_->book1D("METTask_METPhi","METTask_METPhi",80,-4,4);
	    me["hSumET"]              = dbe_->book1D("METTask_SumET","METTask_SumET",1000,-0.50,9999.5);     
	   
	    me["hMExCorrection"]       = dbe_->book1D("METTask_MExCorrection","METTask_MExCorrection", 1000, -500.0,500.0);
	    me["hMEyCorrection"]       = dbe_->book1D("METTask_MEyCorrection","METTask_MEyCorrection", 1000, -500.0,500.0);
	    me["hMuonCorrectionFlag"]      = dbe_->book1D("METTask_CorrectionFlag", "METTask_CorrectionFlag", 6, -0.5, 5.5);

	    me["hMETResolution_GenMETTrue"]      = dbe_->book1D("METTask_METResolution_GenMETTrue","METTask_METResolution_GenMETTrue", 500,-500,500); 
	    me["hMETResolution_GenMETCalo"]      = dbe_->book1D("METTask_METResolution_GenMETCalo","METTask_METResolution_GenMETCalo", 500,-500,500); 

	    me["hMETPhiResolution_GenMETTrue"] = dbe_->book1D("METTask_METPhiResolution_GenMETTrue","METTask_METPhiResolution_GenMETTrue", 80,0,4); 
	    me["hMETPhiResolution_GenMETCalo"] = dbe_->book1D("METTask_METPhiResolution_GenMETCalo","METTask_METPhiResolution_GenMETCalo", 80,0,4); 

	    if( METType_ == "TCMET" ) {
	      me["htrkPt"] = dbe_->book1D("METTask_trackPt", "METTask_trackPt", 50, 0, 500);
	      me["htrkEta"] = dbe_->book1D("METTask_trackEta", "METTask_trackEta", 50, -2.5, 2.5);
	      me["htrkNhits"] = dbe_->book1D("METTask_trackNhits", "METTask_trackNhits", 50, 0, 50);
	      me["htrkChi2"] = dbe_->book1D("METTask_trackNormalizedChi2", "METTask_trackNormalizedChi2", 20, 0, 20);
	      me["htrkD0"] = dbe_->book1D("METTask_trackD0", "METTask_trackd0", 50, -1, 1);
	      me["htrkQuality"] = dbe_->book1D("METTask_trackQuality", "METTask_trackQuality", 30, -0.5, 29.5);
	      me["htrkAlgo"] = dbe_->book1D("METTask_trackAlgo", "METTask_trackAlgo", 6, 3.5, 9.5);
	      me["htrkPtErr"] = dbe_->book1D("METTask_trackPtErr", "METTask_trackPtErr", 200, 0, 2);
	      me["helePt"] = dbe_->book1D("METTask_electronPt", "METTask_electronPt", 50, 0, 500);
	      me["heleEta"] = dbe_->book1D("METTask_electronEta", "METTask_electronEta", 50, -2.5, 2.5);
	      me["heleHoE"] = dbe_->book1D("METTask_electronHoverE", "METTask_electronHoverE", 25, 0, 0.5);
	      me["hmuPt"] = dbe_->book1D("METTask_muonPt", "METTask_muonPt", 50, 0, 500);
	      me["hmuEta"] = dbe_->book1D("METTask_muonEta", "METTask_muonEta", 50, -2.5, 2.5);
	      me["hmuNhits"] = dbe_->book1D("METTask_muonNhits", "METTask_muonNhits", 50, 0, 50);
	      me["hmuChi2"] = dbe_->book1D("METTask_muonNormalizedChi2", "METTask_muonNormalizedChi2", 20, 0, 20);
	      me["hmuD0"] = dbe_->book1D("METTask_muonD0", "METTask_muonD0", 50, -1, 1);
	      me["hnMus"] = dbe_->book1D("METTask_nMus", "METTask_nMus", 5, -0.5, 4.5);
	      me["hnMusPis"] = dbe_->book1D("METTask_nMusAsPis", "METTask_nMusAsPis", 5, -0.5, 4.5);
	      me["hmuSAhits"] = dbe_->book1D("METTask_muonSAhits", "METTask_muonSAhits", 51, -0.5, 50.5);
	      me["hnEls"] = dbe_->book1D("METTask_nEls", "METTask_nEls", 5, -0.5, 4.5);
	      me["hfracTrks"] = dbe_->book1D("METTask_fracTracks", "METTask_fracTracks", 100, 0, 1);
	      me["hdMETx"] = dbe_->book1D("METTask_dMETx", "METTask_dMETx", 500, -250, 250);
	      me["hdMETy"] = dbe_->book1D("METTask_dMETy", "METTask_dMETy", 500, -250, 250);
	      me["hdMET"] = dbe_->book1D("METTask_dMET", "METTask_dMET", 500, -250, 250);
	      me["hdMUx"] = dbe_->book1D("METTask_dMUx", "METTask_dMUx", 500, -250, 250);
	      me["hdMUy"] = dbe_->book1D("METTask_dMUy", "METTask_dMUy", 500, -250, 250);
	    }
	    else if( inputMETLabel_.label() == "corMetGlobalMuons" ) {
	      me["hmuPt"] = dbe_->book1D("METTask_muonPt", "METTask_muonPt", 50, 0, 500);
	      me["hmuEta"] = dbe_->book1D("METTask_muonEta", "METTask_muonEta", 50, -2.5, 2.5);
	      me["hmuNhits"] = dbe_->book1D("METTask_muonNhits", "METTask_muonNhits", 50, 0, 50);
	      me["hmuChi2"] = dbe_->book1D("METTask_muonNormalizedChi2", "METTask_muonNormalizedChi2", 20, 0, 20);
	      me["hmuD0"] = dbe_->book1D("METTask_muonD0", "METTask_muonD0", 50, -1, 1);
	      me["hmuSAhits"] = dbe_->book1D("METTask_muonSAhits", "METTask_muonSAhits", 51, -0.5, 50.5);
	    }
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
	    me["hMuonCorrectionFlag"]     = dbe_->book1D("METTask_CorrectionFlag", "METTask_CorrectionFlag", 6, -0.5, 5.5);	  	   

	    me["hMETResolution_GenMETTrue"]      = dbe_->book1D("METTask_METResolution_GenMETTrue","METTask_METResolution_GenMETTrue",2000,-1000,1000);
	    me["hMETResolution_GenMETCalo"]      = dbe_->book1D("METTask_METResolution_GenMETCalo","METTask_METResolution_GenMETCalo",2000,-1000,1000);
	    
	    me["hMETPhiResolution_GenMETTrue"] = dbe_->book1D("METTask_METPhiResolution_GenMETTrue","METTask_METPhiResolution_GenMETTrue", 80,0,4); 
	    me["hMETPhiResolution_GenMETCalo"] = dbe_->book1D("METTask_METPhiResolution_GenMETCalo","METTask_METPhiResolution_GenMETCalo", 80,0,4); 

	    if( METType_ == "TCMET" ) {
	      me["htrkPt"] = dbe_->book1D("METTask_trackPt", "METTask_trackPt", 250, 0, 500);
	      me["htrkEta"] = dbe_->book1D("METTask_trackEta", "METTask_trackEta", 250, -2.5, 2.5);
	      me["htrkNhits"] = dbe_->book1D("METTask_trackNhits", "METTask_trackNhits", 50, 0, 50);
	      me["htrkChi2"] = dbe_->book1D("METTask_trackNormalizedChi2", "METTask_trackNormalizedChi2", 100, 0, 20);
	      me["htrkD0"] = dbe_->book1D("METTask_trackD0", "METTask_trackd0", 200, -1, 1);
	      me["htrkQuality"] = dbe_->book1D("METTask_trackQuality", "METTask_trackQuality", 30, -0.5, 29.5);
	      me["htrkAlgo"] = dbe_->book1D("METTask_trackAlgo", "METTask_trackAlgo", 6, 3.5, 9.5);
	      me["htrkPtErr"] = dbe_->book1D("METTask_trackPtErr", "METTask_trackPtErr", 200, 0, 2);
	      me["helePt"] = dbe_->book1D("METTask_electronPt", "METTask_electronPt", 250, 0, 500);
	      me["heleEta"] = dbe_->book1D("METTask_electronEta", "METTask_electronEta", 250, -2.5, 2.5);
	      me["heleHoE"] = dbe_->book1D("METTask_electronHoverE", "METTask_electronHoverE", 100, 0, 0.5);
	      me["hmuPt"] = dbe_->book1D("METTask_muonPt", "METTask_muonPt", 250, 0, 500);
	      me["hmuEta"] = dbe_->book1D("METTask_muonEta", "METTask_muonEta", 250, -2.5, 2.5);
	      me["hmuNhits"] = dbe_->book1D("METTask_muonNhits", "METTask_muonNhits", 50, 0, 50);
	      me["hmuChi2"] = dbe_->book1D("METTask_muonNormalizedChi2", "METTask_muonNormalizedChi2", 100, 0, 20);
	      me["hmuD0"] = dbe_->book1D("METTask_muonD0", "METTask_muonD0", 200, -1, 1);
	      me["hnMus"] = dbe_->book1D("METTask_nMus", "METTask_nMus", 5, -0.5, 4.5);
	      me["hnMusPis"] = dbe_->book1D("METTask_nMusAsPis", "METTask_nMusAsPis", 5, -0.5, 4.5);
	      me["hmuSAhits"] = dbe_->book1D("METTask_muonSAhits", "METTask_muonSAhits", 51, -0.5, 50.5);
	      me["hnEls"] = dbe_->book1D("METTask_nEls", "METTask_nEls", 5, -0.5, 4.5);
	      me["hfracTrks"] = dbe_->book1D("METTask_fracTracks", "METTask_fracTracks", 100, 0, 1);
	      me["hdMETx"] = dbe_->book1D("METTask_dMETx", "METTask_dMETx", 500, -250, 250);
	      me["hdMETy"] = dbe_->book1D("METTask_dMETy", "METTask_dMETy", 500, -250, 250);
	      me["hdMET"] = dbe_->book1D("METTask_dMET", "METTask_dMET", 500, -250, 250);
	      me["hdMUx"] = dbe_->book1D("METTask_dMUx", "METTask_dMUx", 500, -250, 250);
	      me["hdMUy"] = dbe_->book1D("METTask_dMUy", "METTask_dMUy", 500, -250, 250);
	    }
	    else if( inputMETLabel_.label() == "corMetGlobalMuons" ) {
	      me["hmuPt"] = dbe_->book1D("METTask_muonPt", "METTask_muonPt", 250, 0, 500);
	      me["hmuEta"] = dbe_->book1D("METTask_muonEta", "METTask_muonEta", 250, -2.5, 2.5);
	      me["hmuNhits"] = dbe_->book1D("METTask_muonNhits", "METTask_muonNhits", 50, 0, 50);
	      me["hmuChi2"] = dbe_->book1D("METTask_muonNormalizedChi2", "METTask_muonNormalizedChi2", 100, 0, 20);
	      me["hmuD0"] = dbe_->book1D("METTask_muonD0", "METTask_muonD0", 200, -1, 1);
	      me["hmuSAhits"] = dbe_->book1D("METTask_muonSAhits", "METTask_muonSAhits", 51, -0.5, 50.5);
	    }
	  }
	
	if(METType_ == "TCMET")
	  {
	    me["hMuonCorrectionFlag"]->setBinLabel(1,"Not Corrected");
	    me["hMuonCorrectionFlag"]->setBinLabel(2,"Global Fit");
	    me["hMuonCorrectionFlag"]->setBinLabel(3,"Tracker Fit");
	    me["hMuonCorrectionFlag"]->setBinLabel(4,"SA Fit");
	    me["hMuonCorrectionFlag"]->setBinLabel(5,"Treated as Pion");
	    me["hMuonCorrectionFlag"]->setBinLabel(6,"Default fit");
	  }
	else if( inputMETLabel_.label() == "corMetGlobalMuons") 
	  {
	    me["hMuonCorrectionFlag"]->setBinLabel(1,"Not Corrected");
	    me["hMuonCorrectionFlag"]->setBinLabel(2,"Global Fit");
	    me["hMuonCorrectionFlag"]->setBinLabel(3,"Tracker Fit");
	    me["hMuonCorrectionFlag"]->setBinLabel(4,"SA Fit");
	    me["hMuonCorrectionFlag"]->setBinLabel(5,"Treated as Pion");
	    me["hMuonCorrectionFlag"]->setBinLabel(6,"Default fit");
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
    using namespace reco;
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

      edm::LogInfo("OutputInfo") << caloMET << " " << caloSumET << std::endl;
      me["hNevents"]->Fill(0.5);
      me["hCaloMEx"]->Fill(caloMEx);
      me["hCaloMEy"]->Fill(caloMEy);
      me["hCaloMET"]->Fill(caloMET);
      me["hCaloMETPhi"]->Fill(caloMETPhi);
      me["hCaloSumET"]->Fill(caloSumET);
      me["hCaloMETSig"]->Fill(caloMETSig);
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

      
      // Get Generated MET for Resolution plots

      edm::Handle<GenMETCollection> genTrue;
      iEvent.getByLabel("genMetTrue", genTrue);
      if (genTrue.isValid()) {
	const GenMETCollection *genmetcol = genTrue.product();
	const GenMET *genMetTrue = &(genmetcol->front());
	double genMET = genMetTrue->pt();
	double genMETPhi = genMetTrue->phi();
	
	me["hCaloMETResolution_GenMETTrue"]->Fill( caloMET - genMET );
	me["hCaloMETPhiResolution_GenMETTrue"]->Fill( TMath::ACos( TMath::Cos( caloMETPhi - genMETPhi ) ) );
      } else {
	edm::LogInfo("OutputInfo") << " failed to retrieve data required by MET Task:  genMetTrue";
      }    


      edm::Handle<GenMETCollection> genCalo;
      iEvent.getByLabel("genMetCalo", genCalo);
      if (genCalo.isValid()) {
	const GenMETCollection *genmetcol = genCalo.product();
	const GenMET  *genMetCalo = &(genmetcol->front());
	double genMET = genMetCalo->pt();
	double genMETPhi = genMetCalo->phi();
	
	me["hCaloMETResolution_GenMETCalo"]->Fill( caloMET - genMET );
	me["hCaloMETPhiResolution_GenMETCalo"]->Fill( TMath::ACos( TMath::Cos( caloMETPhi - genMETPhi ) ) );
      } else {
	edm::LogInfo("OutputInfo") << " failed to retrieve data required by MET Task:  genMetCalo";
      }    


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
      /*
      double genEmEnergy = genmet->emEnergy();
      double genHadEnergy = genmet->hadEnergy();
      double genInvisibleEnergy= genmet->invisibleEnergy();
      double genAuxiliaryEnergy= genmet->auxiliaryEnergy();
      */

      double NeutralEMEtFraction = genmet->NeutralEMEtFraction() ;
      double NeutralHadEtFraction = genmet->NeutralHadEtFraction() ;
      double ChargedEMEtFraction = genmet->ChargedEMEtFraction () ;
      double ChargedHadEtFraction = genmet->ChargedHadEtFraction();
      double MuonEtFraction = genmet->MuonEtFraction() ;
      double InvisibleEtFraction = genmet->InvisibleEtFraction() ;

      me["hNevents"]->Fill(0);
      me["hGenMEx"]->Fill(genMEx);
      me["hGenMEy"]->Fill(genMEy);
      me["hGenMET"]->Fill(genMET);
      me["hGenMETPhi"]->Fill(genMETPhi);
      me["hGenSumET"]->Fill(genSumET);
      me["hGenMETSig"]->Fill(genMETSig);
      //me["hGenEz"]->Fill(genEz);

      me["hNeutralEMEtFraction"]->Fill( NeutralEMEtFraction );
      me["hNeutralHadEtFraction"]->Fill( NeutralHadEtFraction );
      me["hChargedEMEtFraction"]->Fill( ChargedEMEtFraction );
      me["hChargedHadEtFraction"]->Fill( ChargedHadEtFraction );
      me["hMuonEtFraction"]->Fill( MuonEtFraction );
      me["hInvisibleEtFraction"]->Fill( InvisibleEtFraction );

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

      edm::Handle<GenMETCollection> genTrue;
      iEvent.getByLabel("genMetTrue", genTrue);
      if (genTrue.isValid()) {
	const GenMETCollection *genmetcol = genTrue.product();
	const GenMET *genMetTrue = &(genmetcol->front());
	double genMET = genMetTrue->pt();
	double genMETPhi = genMetTrue->phi();
	
	me["hMETResolution_GenMETTrue"]->Fill( MET - genMET );
	me["hMETPhiResolution_GenMETTrue"]->Fill( TMath::ACos( TMath::Cos( METPhi - genMETPhi ) ) );
      } else {
	edm::LogInfo("OutputInfo") << " failed to retrieve data required by MET Task:  genMetTrue";
      }    


      edm::Handle<GenMETCollection> genCalo;
      iEvent.getByLabel("genMetCalo", genCalo);
      if (genCalo.isValid()) {
	const GenMETCollection *genmetcol = genCalo.product();
	const GenMET  *genMetCalo = &(genmetcol->front());
	double genMET = genMetCalo->pt();
	double genMETPhi = genMetCalo->phi();
	
	me["hMETResolution_GenMETCalo"]->Fill( MET - genMET );
	me["hMETPhiResolution_GenMETCalo"]->Fill( TMath::ACos( TMath::Cos( METPhi - genMETPhi ) ) );
      } else {
	edm::LogInfo("OutputInfo") << " failed to retrieve data required by MET Task:  genMetCalo";
      }    
      


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
      iEvent.getByLabel(inputMETLabel_, htcMetcol);

      const CaloMET *caloMet;
      edm::Handle<CaloMETCollection> hcaloMetcol;
      iEvent.getByLabel(inputCaloMETLabel_, hcaloMetcol);

      edm::Handle< reco::MuonCollection > muon_h;
      iEvent.getByLabel(inputMuonLabel_, muon_h);
      
//      edm::Handle< edm::View<reco::Track> > track_h;
      edm::Handle<reco::TrackCollection> track_h;
      iEvent.getByLabel(inputTrackLabel_, track_h);
      
      edm::Handle< edm::View<reco::GsfElectron > > electron_h;
      iEvent.getByLabel(inputElectronLabel_, electron_h);
      
      edm::Handle< reco::BeamSpot > beamSpot_h;
      iEvent.getByLabel(inputBeamSpotLabel_, beamSpot_h);
      
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

      if(!hcaloMetcol.isValid()){
	edm::LogInfo("OutputInfo") << "falied to retrieve data require by MET Task";
	edm::LogInfo("OutputInfo") << "MET Taks cannot continue...!";
	return;
      }
      else
	{
	  const CaloMETCollection *caloMetcol = hcaloMetcol.product();
	  caloMet = &(caloMetcol->front());
	}
      
      if(!muon_h.isValid()){
	edm::LogInfo("OutputInfo") << "falied to retrieve muon data require by MET Task";
	edm::LogInfo("OutputInfo") << "MET Taks cannot continue...!";
	return;
      }
      
      if(!track_h.isValid()){
	edm::LogInfo("OutputInfo") << "falied to retrieve track data require by MET Task";
	edm::LogInfo("OutputInfo") << "MET Taks cannot continue...!";
	return;
      }
      
      if(!electron_h.isValid()){
	edm::LogInfo("OutputInfo") << "falied to retrieve electron data require by MET Task";
	edm::LogInfo("OutputInfo") << "MET Taks cannot continue...!";
	return;
      }
      
      if(!beamSpot_h.isValid()){
	edm::LogInfo("OutputInfo") << "falied to retrieve beam spot data require by MET Task";
	edm::LogInfo("OutputInfo") << "MET Taks cannot continue...!";
	return;
      }

      math::XYZPoint bspot = ( beamSpot_h.isValid() ) ? beamSpot_h->position() : math::XYZPoint(0, 0, 0);
      
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

      double caloMET = caloMet->pt();
      double caloMEx = caloMet->px();
      double caloMEy = caloMet->py();

      me["hdMETx"]->Fill(caloMEx-MEx);
      me["hdMETy"]->Fill(caloMEy-MEy);
      me["hdMET"]->Fill(caloMET-MET);
      
      unsigned int nTracks = track_h->size();
      unsigned int nCorrTracks = 0;
      unsigned int trackCount = 0;
//      for( edm::View<reco::Track>::const_iterator trkit = track_h->begin(); trkit != track_h->end(); trkit++ ) {
      for( reco::TrackCollection::const_iterator trkit = track_h->begin(); trkit != track_h->end(); trkit++ ) {
	   ++trackCount;
	me["htrkPt"]->Fill( trkit->pt() );
	me["htrkEta"]->Fill( trkit->eta() );
	me["htrkNhits"]->Fill( trkit->numberOfValidHits() );
	me["htrkChi2"]->Fill( trkit->chi2() / trkit->ndof() );

	double d0 = -1 * trkit->dxy( bspot );
 
	me["htrkD0"]->Fill( d0 );

	me["htrkQuality"]->Fill( trkit->qualityMask() );
	me["htrkAlgo"]->Fill( trkit->algo() );
	me["htrkPtErr"]->Fill( trkit->ptError() / trkit->pt() );

	reco::TrackRef trkref( track_h, trackCount );

	if( isGoodTrack( trkref, d0) ) ++nCorrTracks;
      }

      float frac = (float)nCorrTracks / (float)nTracks;
      me["hfracTrks"]->Fill(frac);

      int nEls = 0;
      for( edm::View<reco::GsfElectron>::const_iterator eleit = electron_h->begin(); eleit != electron_h->end(); eleit++ ) {
	me["helePt"]->Fill( eleit->p4().pt() );  
	me["heleEta"]->Fill( eleit->p4().eta() );
	me["heleHoE"]->Fill( eleit->hadronicOverEm() );

	reco::TrackRef el_track = eleit->closestCtfTrackRef();

	unsigned int ele_idx = el_track.isNonnull() ? el_track.key() : 99999;

	if( eleit->hadronicOverEm() < 0.1 && ele_idx < nTracks )
	     ++nEls;
      }
      
      me["hnEls"]->Fill(nEls);

      for( reco::MuonCollection::const_iterator muonit = muon_h->begin(); muonit != muon_h->end(); muonit++ ) {

	const reco::TrackRef siTrack = muonit->innerTrack();

	me["hmuPt"]->Fill( muonit->p4().pt() );
	me["hmuEta"]->Fill( muonit->p4().eta() );
	me["hmuNhits"]->Fill( siTrack.isNonnull() ? siTrack->numberOfValidHits() : -999 );
	me["hmuChi2"]->Fill( siTrack.isNonnull() ? siTrack->chi2()/siTrack->ndof() : -999 );

	double d0 = siTrack.isNonnull() ? -1 * siTrack->dxy( bspot) : -999;

	me["hmuD0"]->Fill( d0 );
      }

      edm::Handle< edm::ValueMap<reco::MuonMETCorrectionData> > tcMet_ValueMap_Handle;
      iEvent.getByLabel("muonTCMETValueMapProducer" , "muCorrData", tcMet_ValueMap_Handle);

      edm::Handle< edm::ValueMap<reco::MuonMETCorrectionData> > muon_ValueMap_Handle;
      iEvent.getByLabel("muonMETValueMapProducer" , "muCorrData", muon_ValueMap_Handle);
      
      const unsigned int nMuons = muon_h->size();      

      int nMus = 0;
      int nMusPis = 0;
      double muDx = 0;
      double muDy = 0;
      for( unsigned int mus = 0; mus < nMuons; mus++ ) 
	{
	  reco::MuonRef muref( muon_h, mus);
	  reco::MuonMETCorrectionData muCorrData = (*tcMet_ValueMap_Handle)[muref];
	  reco::MuonMETCorrectionData muonCorrData = (*muon_ValueMap_Handle)[muref];

	  me["hMExCorrection"] -> Fill(muCorrData.corrX());
	  me["hMEyCorrection"] -> Fill(muCorrData.corrY());

	  int type = muCorrData.type();
	  me["hMuonCorrectionFlag"]-> Fill(type);

	  if( type == 1 || type == 2 || type == 5 ) {
	       ++nMus;
	       
	       if( type == 1 ) {
		    muDx += muonCorrData.corrX() - muref->globalTrack()->px();
		    muDy += muonCorrData.corrY() - muref->globalTrack()->py();
	       }
	       else if( type == 2 ) {
		    muDx += muonCorrData.corrX() - muref->innerTrack()->px();
		    muDy += muonCorrData.corrY() - muref->innerTrack()->py();
	       }
	       else if( type == 5 ) {
		    muDx += muonCorrData.corrX() - muref->px();
		    muDy += muonCorrData.corrY() - muref->py();
	       }
	  }
	  else if( type == 4 )
	       ++nMusPis;
	}

      me["hnMus"]->Fill(nMus);
      me["hnMusPis"]->Fill(nMusPis);
      me["hdMUx"]->Fill(muDx);
      me["hdMUy"]->Fill(muDy);

      edm::Handle<GenMETCollection> genTrue;
      iEvent.getByLabel("genMetTrue", genTrue);
      if (genTrue.isValid()) {
	const GenMETCollection *genmetcol = genTrue.product();
	const GenMET *genMetTrue = &(genmetcol->front());
	double genMET = genMetTrue->pt();
	double genMETPhi = genMetTrue->phi();
	
	me["hMETResolution_GenMETTrue"]->Fill( MET - genMET );
	me["hMETPhiResolution_GenMETTrue"]->Fill( TMath::ACos( TMath::Cos( METPhi - genMETPhi ) ) );
      } else {
	edm::LogInfo("OutputInfo") << " failed to retrieve data required by MET Task:  genMetTrue";
      }    


      edm::Handle<GenMETCollection> genCalo;
      iEvent.getByLabel("genMetCalo", genCalo);
      if (genCalo.isValid()) {
	const GenMETCollection *genmetcol = genCalo.product();
	const GenMET  *genMetCalo = &(genmetcol->front());
	double genMET = genMetCalo->pt();
	double genMETPhi = genMetCalo->phi();
	
	me["hMETResolution_GenMETCalo"]->Fill( MET - genMET );
	me["hMETPhiResolution_GenMETCalo"]->Fill( TMath::ACos( TMath::Cos( METPhi - genMETPhi ) ) );
      } else {
	edm::LogInfo("OutputInfo") << " failed to retrieve data required by MET Task:  genMetCalo";
      }          
    }

  else if( inputMETLabel_.label() == "corMetGlobalMuons" )
    {
      const CaloMET *corMetGlobalMuons = 0;
      edm::Handle<CaloMETCollection> hcorMetGlobalMuonscol;
      iEvent.getByLabel(inputMETLabel_, hcorMetGlobalMuonscol );
      if(! hcorMetGlobalMuonscol.isValid()){
	edm::LogInfo("OutputInfo") << "hcorMetGlobalMuonscol is NOT Valid";
	edm::LogInfo("OutputInfo") << "MET Taks continues anyway...!";
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

      edm::Handle< reco::BeamSpot > beamSpot_h;
      iEvent.getByLabel(inputBeamSpotLabel_, beamSpot_h);

      if(!beamSpot_h.isValid()){
	edm::LogInfo("OutputInfo") << "beamSpot is NOT Valid";
	edm::LogInfo("OutputInfo") << "MET Taks continues anyway...!";
      }

      math::XYZPoint bspot = ( beamSpot_h.isValid() ) ? beamSpot_h->position() : math::XYZPoint(0, 0, 0);

      for( reco::MuonCollection::const_iterator muonit = muon_Handle->begin(); muonit != muon_Handle->end(); muonit++ ) {

	const reco::TrackRef siTrack = muonit->innerTrack();
	const reco::TrackRef globalTrack = muonit->globalTrack();

	me["hmuPt"]->Fill( muonit->p4().pt() );
	me["hmuEta"]->Fill( muonit->p4().eta() );
	me["hmuNhits"]->Fill( siTrack.isNonnull() ? siTrack->numberOfValidHits() : -999 );
	me["hmuChi2"]->Fill( siTrack.isNonnull() ? siTrack->chi2()/siTrack->ndof() : -999 );

	double d0 = siTrack.isNonnull() ? -1 * siTrack->dxy( bspot) : -999;

	me["hmuD0"]->Fill( d0 );

	int nHits = globalTrack.isNonnull() ? globalTrack->hitPattern().numberOfValidMuonHits() : -999;

	me["hmuSAhits"]->Fill( nHits );
      }

      const unsigned int nMuons = muon_Handle->size();      
      for( unsigned int mus = 0; mus < nMuons; mus++ ) 
	{
	  reco::MuonRef muref( muon_Handle, mus);
	  reco::MuonMETCorrectionData muCorrData = (*corMetGlobalMuons_ValueMap_Handle)[muref];
	  
	  me["hMExCorrection"] -> Fill(muCorrData.corrY());
	  me["hMEyCorrection"] -> Fill(muCorrData.corrX());
	  me["hMuonCorrectionFlag"]-> Fill(muCorrData.type());
	}

           edm::Handle<GenMETCollection> genTrue;
      iEvent.getByLabel("genMetTrue", genTrue);
      if (genTrue.isValid()) {
	const GenMETCollection *genmetcol = genTrue.product();
	const GenMET *genMetTrue = &(genmetcol->front());
	double genMET = genMetTrue->pt();
	double genMETPhi = genMetTrue->phi();
	
	me["hMETResolution_GenMETTrue"]->Fill( MET - genMET );
	me["hMETPhiResolution_GenMETTrue"]->Fill( TMath::ACos( TMath::Cos( METPhi - genMETPhi ) ) );
      } else {
	edm::LogInfo("OutputInfo") << " failed to retrieve data required by MET Task:  genMetTrue";
      }    


      edm::Handle<GenMETCollection> genCalo;
      iEvent.getByLabel("genMetCalo", genCalo);
      if (genCalo.isValid()) {
	const GenMETCollection *genmetcol = genCalo.product();
	const GenMET  *genMetCalo = &(genmetcol->front());
	double genMET = genMetCalo->pt();
	double genMETPhi = genMetCalo->phi();
	
	me["hMETResolution_GenMETCalo"]->Fill( MET - genMET );
	me["hMETPhiResolution_GenMETCalo"]->Fill( TMath::ACos( TMath::Cos( METPhi - genMETPhi ) ) );
      } else {
	edm::LogInfo("OutputInfo") << " failed to retrieve data required by MET Task:  genMetCalo";
      }    
      


    }

}

void METTester::endJob() 
{ 
}

//determines if track is "good" - i.e. passes quality and kinematic cuts
bool METTester::isGoodTrack( const reco::TrackRef track, float d0corr ) {

     if( fabs( d0corr ) > maxd0_ ) return false;
     if( track->numberOfValidHits() < minhits_ ) return false;
     if( track->normalizedChi2() > maxchi2_ ) return false;
     if( fabs( track->eta() ) > maxeta_ ) return false;
     if( track->pt() > maxpt_ ) return false;
     if( (track->ptError() / track->pt()) > maxPtErr_ ) return false;

     int cut = 0;
     for( unsigned int i = 0; i < trkQuality_.size(); i++ ) {
       
          cut |= (1 << trkQuality_.at(i));
     }

     if( !( ( track->qualityMask() & cut ) == cut ) ) return false;

     bool isGoodAlgo = false;
     if( trkAlgos_.size() == 0 ) isGoodAlgo = true;
     for( unsigned int i = 0; i < trkAlgos_.size(); i++ ) {
       
          if( track->algo() == trkAlgos_.at(i) ) isGoodAlgo = true;
     }

     if( !isGoodAlgo ) return false;

     return true;
}
