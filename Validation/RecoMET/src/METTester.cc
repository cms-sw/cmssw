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
//
// modification: Samantha Hewamanage, Florida International University
// date: 01.30.2012
// note: Added few hists for various nvtx ranges to study PU effects.
//       Cleaned up the code by making it readable and const'ing the
//       variables that should be changed.
//       Changed the number of bins from odd to even. Odd number of bins
//       makes it impossible to rebin a hist.

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
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

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
		sample_                  = iConfig.getUntrackedParameter<std::string>("sample");
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
				me["hNevents"]                = dbe_->book1D("METTask_Nevents","METTask_Nevents",1,0,1); 
				me["hNvertex"]                = dbe_->book1D("METTask_Nvertex","METTask_Nvertex",80,0,80);
				me["hCaloMEx"]                = dbe_->book1D("METTask_CaloMEx","METTask_CaloMEx",500,-1000,500); 
				me["hCaloMEy"]                = dbe_->book1D("METTask_CaloMEy","METTask_CaloMEy",500,-1000,500);
				//        me["hCaloEz"]                 = dbe_->book1D("METTask_CaloEz","METTask_CaloEz",2000,-500,501);
				me["hCaloMETSig"]             = dbe_->book1D("METTask_CaloMETSig","METTask_CaloMETSig",25,0,24.5);
				me["hCaloMET"]                = dbe_->book1D("METTask_CaloMET"            , "METTask_CaloMET"            , 1000,0,2000);
				me["hCaloMET_Nvtx0to5"]       = dbe_->book1D("METTask_CaloMET_Nvtx0to5"   , "METTask_CaloMET_Nvtx0to5"   , 1000,0,2000);
				me["hCaloMET_Nvtx6to10"]      = dbe_->book1D("METTask_CaloMET_Nvtx6to10"  , "METTask_CaloMET_Nvtx6to10"  , 1000,0,2000);
				me["hCaloMET_Nvtx11to15"]     = dbe_->book1D("METTask_CaloMET_Nvtx11to15" , "METTask_CaloMET_Nvtx11to15" , 1000,0,2000);
				me["hCaloMET_Nvtx16to20"]     = dbe_->book1D("METTask_CaloMET_Nvtx16to20" , "METTask_CaloMET_Nvtx16to20" , 1000,0,2000);
				me["hCaloMET_Nvtx21to30"]     = dbe_->book1D("METTask_CaloMET_Nvtx21to30" , "METTask_CaloMET_Nvtx21to30" , 1000,0,2000);
				me["hCaloMET_Nvtx30toInf"]    = dbe_->book1D("METTask_CaloMET_Nvtx30toInf", "METTask_CaloMET_Nvtx30toInf", 1000,0,2000);
				me["hCaloMETPhi"]             = dbe_->book1D("METTask_CaloMETPhi","METTask_CaloMETPhi",40,-4,4);
				me["hCaloSumET"]              = dbe_->book1D("METTask_CaloSumET"            , "METTask_CaloSumET"            , 800,0,8000);   //10GeV
				me["hCaloSumET_Nvtx0to5"]     = dbe_->book1D("METTask_CaloSumET_Nvtx0to5"   , "METTask_CaloSumET_Nvtx0to5"   , 800,0,8000);
				me["hCaloSumET_Nvtx6to10"]    = dbe_->book1D("METTask_CaloSumET_Nvtx6to10"  , "METTask_CaloSumET_Nvtx6to10"  , 800,0,8000);
				me["hCaloSumET_Nvtx11to15"]   = dbe_->book1D("METTask_CaloSumET_Nvtx11to15" , "METTask_CaloSumET_Nvtx11to15" , 800,0,8000);
				me["hCaloSumET_Nvtx16to20"]   = dbe_->book1D("METTask_CaloSumET_Nvtx16to20" , "METTask_CaloSumET_Nvtx16to20" , 800,0,8000);
				me["hCaloSumET_Nvtx21to30"]   = dbe_->book1D("METTask_CaloSumET_Nvtx21to30" , "METTask_CaloSumET_Nvtx21to30" , 800,0,8000);
				me["hCaloSumET_Nvtx30toInf"]  = dbe_->book1D("METTask_CaloSumET_Nvtx30toInf", "METTask_CaloSumET_Nvtx30toInf", 800,0,8000);

				me["hCaloMaxEtInEmTowers"]    = dbe_->book1D("METTask_CaloMaxEtInEmTowers","METTask_CaloMaxEtInEmTowers",600,0,3000);   //5GeV
				me["hCaloMaxEtInHadTowers"]   = dbe_->book1D("METTask_CaloMaxEtInHadTowers","METTask_CaloMaxEtInHadTowers",600,0,3000);  //5GeV
				me["hCaloEtFractionHadronic"] = dbe_->book1D("METTask_CaloEtFractionHadronic","METTask_CaloEtFractionHadronic",100,0,1);
				me["hCaloEmEtFraction"]       = dbe_->book1D("METTask_CaloEmEtFraction","METTask_CaloEmEtFraction",100,0,1);
				me["hCaloHadEtInHB"]          = dbe_->book1D("METTask_CaloHadEtInHB","METTask_CaloHadEtInHB",1000, 0, 5000);  //5GeV  
				me["hCaloHadEtInHO"]          = dbe_->book1D("METTask_CaloHadEtInHO","METTask_CaloHadEtInHO", 250, 0, 500);  //5GeV
				me["hCaloHadEtInHE"]          = dbe_->book1D("METTask_CaloHadEtInHE","METTask_CaloHadEtInHE", 200, 0, 400);  //5GeV
				me["hCaloHadEtInHF"]          = dbe_->book1D("METTask_CaloHadEtInHF","METTask_CaloHadEtInHF", 100, 0, 200);  //5GeV
				me["hCaloEmEtInHF"]           = dbe_->book1D("METTask_CaloEmEtInHF","METTask_CaloEmEtInHF",100, 0, 100);   //5GeV
				me["hCaloSETInpHF"]           = dbe_->book1D("METTask_CaloSETInpHF","METTask_CaloSETInpHF",500, 0, 1000);
				me["hCaloSETInmHF"]           = dbe_->book1D("METTask_CaloSETInmHF","METTask_CaloSETInmHF",500, 0, 1000);
				me["hCaloEmEtInEE"]           = dbe_->book1D("METTask_CaloEmEtInEE","METTask_CaloEmEtInEE",100, 0, 200);    //5GeV
				me["hCaloEmEtInEB"]           = dbe_->book1D("METTask_CaloEmEtInEB","METTask_CaloEmEtInEB",1200, 0, 6000);   //5GeV

				me["hCaloMETResolution_GenMETTrue"]    = dbe_->book1D("METTask_CaloMETResolution_GenMETTrue","METTask_CaloMETResolution_GenMETTrue", 500,-500,500); 
				me["hCaloMETResolution_GenMETCalo"]    = dbe_->book1D("METTask_CaloMETResolution_GenMETCalo","METTask_CaloMETResolution_GenMETCalo", 500,-500,500); 

				me["hCaloMETPhiResolution_GenMETTrue"] = dbe_->book1D("METTask_CaloMETPhiResolution_GenMETTrue","METTask_CaloMETPhiResolution_GenMETTrue", 80,0,4); 
				me["hCaloMETPhiResolution_GenMETCalo"] = dbe_->book1D("METTask_CaloMETPhiResolution_GenMETCalo","METTask_CaloMETPhiResolution_GenMETCalo", 80,0,4); 

			} else 
			{

				//FineBinnning
				me["hNevents"]                = dbe_->book1D("METTask_Nevents","METTask_Nevents",1,0,1);
				me["hNvertex"]                = dbe_->book1D("METTask_Nvertex","METTask_Nvertex",80,0,80);
				me["hCaloMEx"]                = dbe_->book1D("METTask_CaloMEx","METTask_CaloMEx",4000,-1000,1000);
				me["hCaloMEy"]                = dbe_->book1D("METTask_CaloMEy","METTask_CaloMEy",4000,-1000,1000);
				//me["hCaloEz"]                 = dbe_->book1D("METTask_CaloEz","METTask_CaloEz",2000,-500,501);
				me["hCaloMETSig"]             = dbe_->book1D("METTask_CaloMETSig","METTask_CaloMETSig",50,0,50);
				me["hCaloMET"]                = dbe_->book1D("METTask_CaloMET","METTask_CaloMET",2000,0,2000);
				me["hCaloMET_Nvtx0to5"]       = dbe_->book1D("METTask_CaloMET_Nvtx0to5"   , "METTask_CaloMET_Nvtx0to5"   , 2000,0,2000);
				me["hCaloMET_Nvtx6to10"]      = dbe_->book1D("METTask_CaloMET_Nvtx6to10"  , "METTask_CaloMET_Nvtx6to10"  , 2000,0,2000);
				me["hCaloMET_Nvtx11to15"]     = dbe_->book1D("METTask_CaloMET_Nvtx11to15" , "METTask_CaloMET_Nvtx11to15" , 2000,0,2000);
				me["hCaloMET_Nvtx16to20"]     = dbe_->book1D("METTask_CaloMET_Nvtx16to20" , "METTask_CaloMET_Nvtx16to20" , 2000,0,2000);
				me["hCaloMET_Nvtx21to30"]     = dbe_->book1D("METTask_CaloMET_Nvtx21to30" , "METTask_CaloMET_Nvtx21to30" , 2000,0,2000);
				me["hCaloMET_Nvtx30toInf"]    = dbe_->book1D("METTask_CaloMET_Nvtx30toInf", "METTask_CaloMET_Nvtx30toInf", 2000,0,2000);
				me["hCaloMETPhi"]             = dbe_->book1D("METTask_CaloMETPhi","METTask_CaloMETPhi",40,-4,4);
				me["hCaloSumET"]              = dbe_->book1D("METTask_CaloSumET","METTask_CaloSumET",10000,0,10000);
				me["hCaloSumET_Nvtx0to5"]     = dbe_->book1D("METTask_CaloSumET_Nvtx0to5"   , "METTask_CaloSumET_Nvtx0to5"   , 10000,0,10000);
				me["hCaloSumET_Nvtx6to10"]    = dbe_->book1D("METTask_CaloSumET_Nvtx6to10"  , "METTask_CaloSumET_Nvtx6to10"  , 10000,0,10000);
				me["hCaloSumET_Nvtx11to15"]   = dbe_->book1D("METTask_CaloSumET_Nvtx11to15" , "METTask_CaloSumET_Nvtx11to15" , 10000,0,10000);
				me["hCaloSumET_Nvtx16to20"]   = dbe_->book1D("METTask_CaloSumET_Nvtx16to20" , "METTask_CaloSumET_Nvtx16to20" , 10000,0,10000);
				me["hCaloSumET_Nvtx21to30"]   = dbe_->book1D("METTask_CaloSumET_Nvtx21to30" , "METTask_CaloSumET_Nvtx21to30" , 10000,0,10000);
				me["hCaloSumET_Nvtx30toInf"]  = dbe_->book1D("METTask_CaloSumET_Nvtx30toInf", "METTask_CaloSumET_Nvtx30toInf", 10000,0,10000);
				me["hCaloMaxEtInEmTowers"]    = dbe_->book1D("METTask_CaloMaxEtInEmTowers","METTask_CaloMaxEtInEmTowers",4000,0,4000);
				me["hCaloMaxEtInHadTowers"]   = dbe_->book1D("METTask_CaloMaxEtInHadTowers","METTask_CaloMaxEtInHadTowers",4000,0,4000);
				me["hCaloEtFractionHadronic"] = dbe_->book1D("METTask_CaloEtFractionHadronic","METTask_CaloEtFractionHadronic",100,0,1);
				me["hCaloEmEtFraction"]       = dbe_->book1D("METTask_CaloEmEtFraction","METTask_CaloEmEtFraction",100,0,1);
				me["hCaloHadEtInHB"]          = dbe_->book1D("METTask_CaloHadEtInHB","METTask_CaloHadEtInHB",8000,0,8000);
				me["hCaloHadEtInHO"]          = dbe_->book1D("METTask_CaloHadEtInHO","METTask_CaloHadEtInHO",4000,0,4000);
				me["hCaloHadEtInHE"]          = dbe_->book1D("METTask_CaloHadEtInHE","METTask_CaloHadEtInHE",4000,0,4000);
				me["hCaloHadEtInHF"]          = dbe_->book1D("METTask_CaloHadEtInHF","METTask_CaloHadEtInHF",4000,0,4000);
				me["hCaloHadEtInEB"]          = dbe_->book1D("METTask_CaloHadEtInEB","METTask_CaloHadEtInEB",8000,0,8000);
				me["hCaloHadEtInEE"]          = dbe_->book1D("METTask_CaloHadEtInEE","METTask_CaloHadEtInEE",4000,0,4000);
				me["hCaloEmEtInHF"]           = dbe_->book1D("METTask_CaloEmEtInHF","METTask_CaloEmEtInHF",4000,0,4000);
				me["hCaloSETInpHF"]           = dbe_->book1D("METTask_CaloSETInpHF","METTask_CaloSETInpHF",4000,0,4000);
				me["hCaloSETInmHF"]           = dbe_->book1D("METTask_CaloSETInmHF","METTask_CaloSETInmHF",4000,0,4000);
				me["hCaloEmEtInEE"]           = dbe_->book1D("METTask_CaloEmEtInEE","METTask_CaloEmEtInEE",4000,0,4000);
				me["hCaloEmEtInEB"]           = dbe_->book1D("METTask_CaloEmEtInEB","METTask_CaloEmEtInEB",8000,0,8000);

				me["hCaloMETResolution_GenMETTrue"]    = dbe_->book1D("METTask_CaloMETResolution_GenMETTrue","METTask_CaloMETResolution_GenMETTrue", 2000,-1000,1000); 
				me["hCaloMETResolution_GenMETCalo"]    = dbe_->book1D("METTask_CaloMETResolution_GenMETCalo","METTask_CaloMETResolution_GenMETCalo", 2000,-1000,1000); 

				me["hCaloMETPhiResolution_GenMETTrue"] = dbe_->book1D("METTask_CaloMETPhiResolution_GenMETTrue","METTask_CaloMETPhiResolution_GenMETTrue", 80,0,4); 
				me["hCaloMETPhiResolution_GenMETCalo"] = dbe_->book1D("METTask_CaloMETPhiResolution_GenMETCalo","METTask_CaloMETPhiResolution_GenMETCalo", 80,0,4); 
			}
		
		
		} else if (METType_ == "GenMET") 
		{

			// GenMET Histograms

			if(!finebinning_)
			{
				me["hNevents"]                = dbe_->book1D("METTask_Nevents","METTask_Nevents",1,0,1); 
				me["hGenMEx"]                 = dbe_->book1D("METTask_GenMEx","METTask_GenMEx",1000,-999.5,999.5);
				me["hGenMEy"]                 = dbe_->book1D("METTask_GenMEy","METTask_GenMEy",1000,-999.5,999.5);
				//        me["hGenEz"]                  = dbe_->book1D("METTask_GenEz","METTask_GenEz",2000,-500,501);
				me["hGenMETSig"]              = dbe_->book1D("METTask_GenMETSig","METTask_GenMETSig",51,0,51);
				me["hGenMET"]                 = dbe_->book1D("METTask_GenMET","METTask_GenMET", 2000,-0.5,1999.5);
				me["hGenMETPhi"]              = dbe_->book1D("METTask_GenMETPhi","METTask_GenMETPhi",40,-4,4);
				me["hGenSumET"]               = dbe_->book1D("METTask_GenSumET","METTask_GenSumET",1000,-0.5,9999.5);

				me["hNeutralEMEtFraction"]    = dbe_->book1D("METTask_GenNeutralEMEtFraction", "METTask_GenNeutralEMEtFraction", 120, 0.0, 1.2 );
				me["hNeutralHadEtFraction"]   = dbe_->book1D("METTask_GenNeutralHadEtFraction", "METTask_GenNeutralHadEtFraction", 120, 0.0, 1.2 );
				me["hChargedEMEtFraction"]    = dbe_->book1D("METTask_GenChargedEMEtFraction", "METTask_GenChargedEMEtFraction", 120, 0.0, 1.2);
				me["hChargedHadEtFraction"]   = dbe_->book1D("METTask_GenChargedHadEtFraction", "METTask_GenChargedHadEtFraction", 120, 0.0,1.2);
				me["hMuonEtFraction"]         = dbe_->book1D("METTask_GenMuonEtFraction", "METTask_GenMuonEtFraction", 120, 0.0, 1.2 );
				me["hInvisibleEtFraction"]    = dbe_->book1D("METTask_GenInvisibleEtFraction", "METTask_GenInvisibleEtFraction", 120, 0.0, 1.2 );

			} else 
			{
				me["hNevents"]                = dbe_->book1D("METTask_Nevents","METTask_Nevents",1,0,1);
				me["hGenMEx"]                 = dbe_->book1D("METTask_GenMEx","METTask_GenMEx",4000,-1000,1000);
				me["hGenMEy"]                 = dbe_->book1D("METTask_GenMEy","METTask_GenMEy",4000,-1000,1000);
				//me["hGenEz"]                  = dbe_->book1D("METTask_GenEz","METTask_GenEz",2000,-500,500);
				me["hGenMETSig"]              = dbe_->book1D("METTask_GenMETSig","METTask_GenMETSig",51,0,51);
				me["hGenMET"]                 = dbe_->book1D("METTask_GenMET","METTask_GenMET",2000,0,2000);
				me["hGenMETPhi"]              = dbe_->book1D("METTask_GenMETPhi","METTask_GenMETPhi",40,-4,4);
				me["hGenSumET"]               = dbe_->book1D("METTask_GenSumET","METTask_GenSumET",10000,0,10000);
				me["hNeutralEMEtFraction"]    = dbe_->book1D("METTask_GenNeutralEMEtFraction", "METTask_GenNeutralEMEtFraction", 120, 0.0, 1.2 );
				me["hNeutralHadEtFraction"]   = dbe_->book1D("METTask_GenNeutralHadEtFraction", "METTask_GenNeutralHadEtFraction", 120, 0.0, 1.2 );
				me["hChargedEMEtFraction"]    = dbe_->book1D("METTask_GenChargedEMEtFraction", "METTask_GenChargedEMEtFraction", 120, 0.0, 1.2);
				me["hChargedHadEtFraction"]   = dbe_->book1D("METTask_GenChargedHadEtFraction", "METTask_GenChargedHadEtFraction", 120, 0.0,1.2);
				me["hMuonEtFraction"]         = dbe_->book1D("METTask_GenMuonEtFraction", "METTask_GenMuonEtFraction", 120, 0.0, 1.2 );
				me["hInvisibleEtFraction"]    = dbe_->book1D("METTask_GenInvisibleEtFraction", "METTask_GenInvisibleEtFraction", 120, 0.0, 1.2 );

			}
		
		} else if (METType_ == "MET")
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
				me["hMETPhi"]             = dbe_->book1D("METTask_METPhi","METTask_METPhi",40,-4,4);
				me["hSumET"]              = dbe_->book1D("METTask_SumET","METTask_SumET",1000,0,9999.5);   

			} else
			{
				me["hNevents"]                = dbe_->book1D("METTask_Nevents","METTask_Nevents",1,0,1);
				me["hMEx"]                = dbe_->book1D("METTask_MEx","METTask_MEx",2000,-500,500);
				me["hMEy"]                = dbe_->book1D("METTask_MEy","METTask_MEy",2000,-500,500);
				//me["hEz"]                 = dbe_->book1D("METTask_Ez","METTask_Ez",2000,-500,500);
				me["hMETSig"]             = dbe_->book1D("METTask_METSig","METTask_METSig",51,0,51);
				me["hMET"]                = dbe_->book1D("METTask_MET","METTask_MET",2000,0,2000);
				me["hMETPhi"]             = dbe_->book1D("METTask_METPhi","METTask_METPhi",40,-4,4);
				me["hSumET"]              = dbe_->book1D("METTask_SumET","METTask_SumET",4000,0,4000);

			}
		
		} else if (METType_ == "PFMET")
		{
			// PFMET Histograms                                                                                                                  
			if(!finebinning_)
			{
				me["hNevents"]            = dbe_->book1D("METTask_Nevents","METTask_Nevents",1,0,1);   
				me["hNvertex"]            = dbe_->book1D("METTask_Nvertex","METTask_Nvertex",80,0,80);
				me["hMEx"]                = dbe_->book1D("METTask_MEx","METTask_MEx",1000,-1000,1000);
				me["hMEy"]                = dbe_->book1D("METTask_MEy","METTask_MEy",1000,-1000,1000);
				//        me["hEz"]                 = dbe_->book1D("METTask_Ez","METTask_Ez",2000,-500,500);
				me["hMETSig"]             = dbe_->book1D("METTask_METSig","METTask_METSig",51,0,51);
				me["hMET"]                = dbe_->book1D("METTask_MET","METTask_MET",1000,0,2000);
				me["hMET_Nvtx0to5"]       = dbe_->book1D("METTask_MET_Nvtx0to5"   , "METTask_MET_Nvtx0to5"   , 1000,0,2000);
				me["hMET_Nvtx6to10"]      = dbe_->book1D("METTask_MET_Nvtx6to10"  , "METTask_MET_Nvtx6to10"  , 1000,0,2000);
				me["hMET_Nvtx11to15"]     = dbe_->book1D("METTask_MET_Nvtx11to15" , "METTask_MET_Nvtx11to15" , 1000,0,2000);
				me["hMET_Nvtx16to20"]     = dbe_->book1D("METTask_MET_Nvtx16to20" , "METTask_MET_Nvtx16to20" , 1000,0,2000);
				me["hMET_Nvtx21to30"]     = dbe_->book1D("METTask_MET_Nvtx21to30" , "METTask_MET_Nvtx21to30" , 1000,0,2000);
				me["hMET_Nvtx30toInf"]    = dbe_->book1D("METTask_MET_Nvtx30toInf", "METTask_MET_Nvtx30toInf", 1000,0,2000);
				me["hMETPhi"]             = dbe_->book1D("METTask_METPhi","METTask_METPhi",40,-4,4);
				me["hSumET"]              = dbe_->book1D("METTask_SumET","METTask_SumET",1000,0,10000);     
				me["hSumET_Nvtx0to5"]     = dbe_->book1D("METTask_SumET_Nvtx0to5"   , "METTask_SumET_Nvtx0to5"   , 1000,0,2000);
				me["hSumET_Nvtx6to10"]    = dbe_->book1D("METTask_SumET_Nvtx6to10"  , "METTask_SumET_Nvtx6to10"  , 1000,0,2000);
				me["hSumET_Nvtx11to15"]   = dbe_->book1D("METTask_SumET_Nvtx11to15" , "METTask_SumET_Nvtx11to15" , 1000,0,2000);
				me["hSumET_Nvtx16to20"]   = dbe_->book1D("METTask_SumET_Nvtx16to20" , "METTask_SumET_Nvtx16to20" , 1000,0,2000);
				me["hSumET_Nvtx21to30"]   = dbe_->book1D("METTask_SumET_Nvtx21to30" , "METTask_SumET_Nvtx21to30" , 1000,0,2000);
				me["hSumET_Nvtx30toInf"]  = dbe_->book1D("METTask_SumET_Nvtx30toInf", "METTask_SumET_Nvtx30toInf", 1000,0,2000);


				me["hMETResolution_GenMETTrue"]      = dbe_->book1D("METTask_METResolution_GenMETTrue","METTask_METResolution_GenMETTrue", 500,-500,500); 
				me["hMETResolution_GenMETCalo"]      = dbe_->book1D("METTask_METResolution_GenMETCalo","METTask_METResolution_GenMETCalo", 500,-500,500); 

				me["hMETPhiResolution_GenMETTrue"] = dbe_->book1D("METTask_METPhiResolution_GenMETTrue","METTask_METPhiResolution_GenMETTrue", 80,0,4); 
				me["hMETPhiResolution_GenMETCalo"] = dbe_->book1D("METTask_METPhiResolution_GenMETCalo","METTask_METPhiResolution_GenMETCalo", 80,0,4); 


				me["hMETResolution_GenMETTrue_MET0to20"]    = dbe_->book1D("METTask_METResolution_GenMETTrue_MET0to20"   , "METTask_METResolution_GenMETTrue_MET0to20"   , 500,-500,500); 
				me["hMETResolution_GenMETTrue_MET20to40"]   = dbe_->book1D("METTask_METResolution_GenMETTrue_MET20to40"  , "METTask_METResolution_GenMETTrue_MET20to40"  , 500,-500,500); 
				me["hMETResolution_GenMETTrue_MET40to60"]   = dbe_->book1D("METTask_METResolution_GenMETTrue_MET40to60"  , "METTask_METResolution_GenMETTrue_MET40to60"  , 500,-500,500); 
				me["hMETResolution_GenMETTrue_MET60to80"]   = dbe_->book1D("METTask_METResolution_GenMETTrue_MET60to80"  , "METTask_METResolution_GenMETTrue_MET60to80"  , 500,-500,500); 
				me["hMETResolution_GenMETTrue_MET80to100"]  = dbe_->book1D("METTask_METResolution_GenMETTrue_MET80to100" , "METTask_METResolution_GenMETTrue_MET80to100" , 500,-500,500); 
				me["hMETResolution_GenMETTrue_MET100to150"] = dbe_->book1D("METTask_METResolution_GenMETTrue_MET100to150", "METTask_METResolution_GenMETTrue_MET100to150", 500,-500,500); 
				me["hMETResolution_GenMETTrue_MET150to200"] = dbe_->book1D("METTask_METResolution_GenMETTrue_MET150to200", "METTask_METResolution_GenMETTrue_MET150to200", 500,-500,500); 
				me["hMETResolution_GenMETTrue_MET200to300"] = dbe_->book1D("METTask_METResolution_GenMETTrue_MET200to300", "METTask_METResolution_GenMETTrue_MET200to300", 500,-500,500); 
				me["hMETResolution_GenMETTrue_MET300to400"] = dbe_->book1D("METTask_METResolution_GenMETTrue_MET300to400", "METTask_METResolution_GenMETTrue_MET300to400", 500,-500,500); 
				me["hMETResolution_GenMETTrue_MET400to500"] = dbe_->book1D("METTask_METResolution_GenMETTrue_MET400to500", "METTask_METResolution_GenMETTrue_MET400to500", 500,-500,500); 
				//this will be filled at the end of the job using info from above hists
				int nBins = 10;
				float bins[] = {0.,20.,40.,60.,80.,100.,150.,200.,300.,400.,500.};
				me["hMETResolution_GenMETTrue_METResolution"]     = dbe_->book1D("METTask_METResolution_GenMETTrue_InMETBins","METTask_METResolution_GenMETTrue_InMETBins",nBins, bins);


			} else 
			{

				//FineBin
				me["hNevents"]            = dbe_->book1D("METTask_Nevents","METTask_Nevents",1,0,1);
				me["hNvertex"]            = dbe_->book1D("METTask_Nvertex","METTask_Nvertex",80,0,80);
				me["hMEx"]                = dbe_->book1D("METTask_MEx","METTask_MEx",2000,-500,500);
				me["hMEy"]                = dbe_->book1D("METTask_MEy","METTask_MEy",2000,-500,500);
				//me["hEz"]               = dbe_->book1D("METTask_Ez","METTask_Ez",2000,-500,500);
				me["hMETSig"]             = dbe_->book1D("METTask_METSig","METTask_METSig",51,0,51);
				me["hMET"]                = dbe_->book1D("METTask_MET","METTask_MET", 2000,0,2000);
				me["hMET_Nvtx0to5"]       = dbe_->book1D("METTask_MET_Nvtx0to5"   , "METTask_MET_Nvtx0to5"   , 2000,0,2000);
				me["hMET_Nvtx6to10"]      = dbe_->book1D("METTask_MET_Nvtx6to10"  , "METTask_MET_Nvtx6to10"  , 2000,0,2000);
				me["hMET_Nvtx11to15"]     = dbe_->book1D("METTask_MET_Nvtx11to15" , "METTask_MET_Nvtx11to15" , 2000,0,2000);
				me["hMET_Nvtx16to20"]     = dbe_->book1D("METTask_MET_Nvtx16to20" , "METTask_MET_Nvtx16to20" , 2000,0,2000);
				me["hMET_Nvtx21to30"]     = dbe_->book1D("METTask_MET_Nvtx21to30" , "METTask_MET_Nvtx21to30" , 2000,0,2000);
				me["hMET_Nvtx30toInf"]    = dbe_->book1D("METTask_MET_Nvtx30toInf", "METTask_MET_Nvtx30toInf", 2000,0,2000);
				me["hMETPhi"]             = dbe_->book1D("METTask_METPhi","METTask_METPhi",40,-4,4);
				me["hSumET"]              = dbe_->book1D("METTask_SumET","METTask_SumET",4000,0,4000);
				me["hSumET_Nvtx0to5"]     = dbe_->book1D("METTask_SumET_Nvtx0to5"   , "METTask_SumET_Nvtx0to5"   , 4000,0,4000);
				me["hSumET_Nvtx6to10"]    = dbe_->book1D("METTask_SumET_Nvtx6to10"  , "METTask_SumET_Nvtx6to10"  , 4000,0,4000);
				me["hSumET_Nvtx11to15"]   = dbe_->book1D("METTask_SumET_Nvtx11to15" , "METTask_SumET_Nvtx11to15" , 4000,0,4000);
				me["hSumET_Nvtx16to20"]   = dbe_->book1D("METTask_SumET_Nvtx16to20" , "METTask_SumET_Nvtx16to20" , 4000,0,4000);
				me["hSumET_Nvtx21to30"]   = dbe_->book1D("METTask_SumET_Nvtx21to30" , "METTask_SumET_Nvtx21to30" , 4000,0,4000);
				me["hSumET_Nvtx30toInf"]  = dbe_->book1D("METTask_SumET_Nvtx30toInf", "METTask_SumET_Nvtx30toInf", 4000,0,4000);

				me["hMETResolution_GenMETTrue"]      = dbe_->book1D("METTask_METResolution_GenMETTrue","METTask_METResolution_GenMETTrue",2000,-1000,1000);
				me["hMETResolution_GenMETCalo"]      = dbe_->book1D("METTask_METResolution_GenMETCalo","METTask_METResolution_GenMETCalo",2000,-1000,1000);

				me["hMETPhiResolution_GenMETTrue"] = dbe_->book1D("METTask_METPhiResolution_GenMETTrue","METTask_METPhiResolution_GenMETTrue", 80,0,4); 
				me["hMETPhiResolution_GenMETCalo"] = dbe_->book1D("METTask_METPhiResolution_GenMETCalo","METTask_METPhiResolution_GenMETCalo", 80,0,4); 


				me["hMETResolution_GenMETTrue_MET0to20"]    = dbe_->book1D("METTask_METResolution_GenMETTrue_MET0to20","METTask_METResolution_GenMETTrue_MET0to20"      , 2000,-1000,1000); 
				me["hMETResolution_GenMETTrue_MET20to40"]   = dbe_->book1D("METTask_METResolution_GenMETTrue_MET20to40","METTask_METResolution_GenMETTrue_MET20to40"    , 2000,-1000,1000); 
				me["hMETResolution_GenMETTrue_MET40to60"]   = dbe_->book1D("METTask_METResolution_GenMETTrue_MET40to60","METTask_METResolution_GenMETTrue_MET40to60"    , 2000,-1000,1000); 
				me["hMETResolution_GenMETTrue_MET60to80"]   = dbe_->book1D("METTask_METResolution_GenMETTrue_MET60to80","METTask_METResolution_GenMETTrue_MET60to80"    , 2000,-1000,1000); 
				me["hMETResolution_GenMETTrue_MET80to100"]  = dbe_->book1D("METTask_METResolution_GenMETTrue_MET80to100","METTask_METResolution_GenMETTrue_MET80to100"  , 2000,-1000,1000); 
				me["hMETResolution_GenMETTrue_MET100to150"] = dbe_->book1D("METTask_METResolution_GenMETTrue_MET100to150","METTask_METResolution_GenMETTrue_MET100to150", 2000,-1000,1000); 
				me["hMETResolution_GenMETTrue_MET150to200"] = dbe_->book1D("METTask_METResolution_GenMETTrue_MET150to200","METTask_METResolution_GenMETTrue_MET150to200", 2000,-1000,1000); 
				me["hMETResolution_GenMETTrue_MET200to300"] = dbe_->book1D("METTask_METResolution_GenMETTrue_MET200to300","METTask_METResolution_GenMETTrue_MET200to300", 2000,-1000,1000); 
				me["hMETResolution_GenMETTrue_MET300to400"] = dbe_->book1D("METTask_METResolution_GenMETTrue_MET300to400","METTask_METResolution_GenMETTrue_MET300to400", 2000,-1000,1000); 
				me["hMETResolution_GenMETTrue_MET400to500"] = dbe_->book1D("METTask_METResolution_GenMETTrue_MET400to500","METTask_METResolution_GenMETTrue_MET400to500", 2000,-1000,1000); 

				//this will be filled at the end of the job using info from above hists
				int nBins = 10;
				float bins[] = {0.,20.,40.,60.,80.,100.,150.,200.,300.,400.,500.};
				me["hMETResolution_GenMETTrue_METResolution"]     = dbe_->book1D("METTask_METResolution_GenMETTrue_InMETBins","METTask_METResolution_GenMETTrue_InMETBins",nBins, bins);
			}
		
		} else if (METType_ == "TCMET" || inputMETLabel_.label() == "corMetGlobalMuons") 
		{
			//TCMET or MuonCorrectedCaloMET Histograms                                                                                                                  
			if(!finebinning_)
			{
				me["hNevents"]                = dbe_->book1D("METTask_Nevents","METTask_Nevents",1,0,1);   
				me["hMEx"]                = dbe_->book1D("METTask_MEx","METTask_MEx",1000,-999.5,999.5);
				me["hMEy"]                = dbe_->book1D("METTask_MEy","METTask_MEy",1000,-999.5,999.5);
				me["hMETSig"]             = dbe_->book1D("METTask_METSig","METTask_METSig",51,0,51);
				me["hMET"]                = dbe_->book1D("METTask_MET","METTask_MET",1000,-0.5,1999.5);
				me["hMETPhi"]             = dbe_->book1D("METTask_METPhi","METTask_METPhi",40,-4,4);
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
				me["hMEx"]                = dbe_->book1D("METTask_MEx","METTask_MEx",2000,-500,500);
				me["hMEy"]                = dbe_->book1D("METTask_MEy","METTask_MEy",2000,-500,500);
				me["hMETSig"]             = dbe_->book1D("METTask_METSig","METTask_METSig",51,0,51);
				me["hMET"]                = dbe_->book1D("METTask_MET","METTask_MET",2000,0,2002);
				me["hMETPhi"]             = dbe_->book1D("METTask_METPhi","METTask_METPhi",40,-4,4);
				me["hSumET"]              = dbe_->book1D("METTask_SumET","METTask_SumET",4000,0,4000);
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

	edm::Handle<reco::VertexCollection> vertexHandle;
	iEvent.getByLabel("offlinePrimaryVertices", vertexHandle);
   if (! vertexHandle.isValid())
	{
		std::cout << __FUNCTION__ << ":" << __LINE__ << ":vertexHandle handle not found!" << std::endl;
		assert(false);
	}
	const int nvtx = vertexHandle->size();

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
		const double caloSumET = calomet->sumEt();
		const double caloMETSig = calomet->mEtSig();
		const double caloMET = calomet->pt();
		const double caloMEx = calomet->px();
		const double caloMEy = calomet->py();
		const double caloMETPhi = calomet->phi();
		const double caloMaxEtInEMTowers = calomet->maxEtInEmTowers();
		const double caloMaxEtInHadTowers = calomet->maxEtInHadTowers();
		const double caloEtFractionHadronic = calomet->etFractionHadronic();
		const double caloEmEtFraction = calomet->emEtFraction();
		const double caloHadEtInHB = calomet->hadEtInHB();
		const double caloHadEtInHO = calomet->hadEtInHO();
		const double caloHadEtInHE = calomet->hadEtInHE();
		const double caloHadEtInHF = calomet->hadEtInHF();
		const double caloEmEtInEB = calomet->emEtInEB();
		const double caloEmEtInEE = calomet->emEtInEE();
		const double caloEmEtInHF = calomet->emEtInHF();
		const double caloSETInpHF = calomet->CaloSETInpHF();
		const double caloSETInmHF = calomet->CaloSETInmHF();

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
		me["hCaloSETInpHF"]->Fill(caloSETInpHF);
		me["hCaloSETInmHF"]->Fill(caloSETInmHF);
		
		/******************************************
		 * For PU Studies
		 * ****************************************/
		me["hNvertex"]->Fill(nvtx);

		if (nvtx <= 5)
		{
			me["hCaloMET_Nvtx0to5"]->Fill(caloMET);
			me["hCaloSumET_Nvtx0to5"]->Fill(caloSumET);
		} else if (nvtx >= 6 && nvtx <= 10)
		{
			me["hCaloMET_Nvtx6to10"]->Fill(caloMET);
			me["hCaloSumET_Nvtx6to10"]->Fill(caloSumET);
		} else if (nvtx >= 11 && nvtx <= 15)
		{
			me["hCaloMET_Nvtx11to15"]->Fill(caloMET);
			me["hCaloSumET_Nvtx11to15"]->Fill(caloSumET);
		} else if (nvtx >= 16 && nvtx <= 20)
		{
			me["hCaloMET_Nvtx16to20"]->Fill(caloMET);
			me["hCaloSumET_Nvtx16to20"]->Fill(caloSumET);
		} else if (nvtx >= 21 && nvtx <= 30)
		{
			me["hCaloMET_Nvtx21to30"]->Fill(caloMET);
			me["hCaloSumET_Nvtx21to30"]->Fill(caloSumET);
		} else if (nvtx >= 31)
		{
			me["hCaloMET_Nvtx30toInf"]->Fill(caloMET);
			me["hCaloSumET_Nvtx30toInf"]->Fill(caloSumET);
		}


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
			const double genMET = genMetCalo->pt();
			const double genMETPhi = genMetCalo->phi();

			me["hCaloMETResolution_GenMETCalo"]->Fill( caloMET - genMET );
			me["hCaloMETPhiResolution_GenMETCalo"]->Fill( TMath::ACos( TMath::Cos( caloMETPhi - genMETPhi ) ) );
		} else {
			edm::LogInfo("OutputInfo") << " failed to retrieve data required by MET Task:  genMetCalo";
		}    


	}	else if (METType_ == "GenMET")
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
		const double genSumET = genmet->sumEt();
		const double genMET = genmet->pt();
		const double genMEx = genmet->px();
		const double genMEy = genmet->py();
		const double genMETPhi = genmet->phi();
		const double genMETSig = genmet->mEtSig();
		/*
			double genEmEnergy = genmet->emEnergy();
			double genHadEnergy = genmet->hadEnergy();
			double genInvisibleEnergy= genmet->invisibleEnergy();
			double genAuxiliaryEnergy= genmet->auxiliaryEnergy();
			*/

		const double NeutralEMEtFraction = genmet->NeutralEMEtFraction() ;
		const double NeutralHadEtFraction = genmet->NeutralHadEtFraction() ;
		const double ChargedEMEtFraction = genmet->ChargedEMEtFraction () ;
		const double ChargedHadEtFraction = genmet->ChargedHadEtFraction();
		const double MuonEtFraction = genmet->MuonEtFraction() ;
		const double InvisibleEtFraction = genmet->InvisibleEtFraction() ;

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
	
	} else if( METType_ == "PFMET")
	{
		const PFMET *pfmet;
		edm::Handle<PFMETCollection> hpfmetcol;
		iEvent.getByLabel(inputMETLabel_,hpfmetcol);
		if (!hpfmetcol.isValid()){
			edm::LogInfo("OutputInfo") << "falied to retrieve data require by MET Task";
			edm::LogInfo("OutputInfo") << "MET Taks cannot continue...!";
			return;
		} else
		{
			const PFMETCollection *pfmetcol = hpfmetcol.product();
			pfmet = &(pfmetcol->front());
		}
		// Reconstructed MET Information                                                                                                     
		const double SumET = pfmet->sumEt();
		const double MET = pfmet->pt();
		const double MEx = pfmet->px();
		const double MEy = pfmet->py();
		const double METPhi = pfmet->phi();
		const double METSig = pfmet->mEtSig();
		me["hMEx"]->Fill(MEx);
		me["hMEy"]->Fill(MEy);
		me["hMET"]->Fill(MET);
		me["hMETPhi"]->Fill(METPhi);
		me["hSumET"]->Fill(SumET);
		me["hMETSig"]->Fill(METSig);
		me["hNevents"]->Fill(0.5);

		/******************************************
		 * For PU Studies
		 * ****************************************/
		
		me["hNvertex"]->Fill(nvtx);
		if (nvtx <= 5)
		{
			me["hMET_Nvtx0to5"]->Fill(MET);
			me["hSumET_Nvtx0to5"]->Fill(SumET);
		} else if (nvtx >= 6 && nvtx <= 10)
		{
			me["hMET_Nvtx6to10"]->Fill(MET);
			me["hSumET_Nvtx6to10"]->Fill(SumET);
		} else if (nvtx >= 11 && nvtx <= 15)
		{
			me["hMET_Nvtx11to15"]->Fill(MET);
			me["hSumET_Nvtx11to15"]->Fill(SumET);
		} else if (nvtx >= 16 && nvtx <= 20)
		{
			me["hMET_Nvtx16to20"]->Fill(MET);
			me["hSumET_Nvtx16to20"]->Fill(SumET);
		} else if (nvtx >= 21 && nvtx <= 30)
		{
			me["hMET_Nvtx21to30"]->Fill(MET);
			me["hSumET_Nvtx21to30"]->Fill(SumET);
		} else if (nvtx >= 31)
		{
			me["hMET_Nvtx30toInf"]->Fill(MET);
			me["hSumET_Nvtx30toInf"]->Fill(SumET);
		}


		edm::Handle<GenMETCollection> genTrue;
		iEvent.getByLabel("genMetTrue", genTrue);
		if (genTrue.isValid()) {
			const GenMETCollection *genmetcol = genTrue.product();
			const GenMET *genMetTrue = &(genmetcol->front());
			const double genMET = genMetTrue->pt();
			const double genMETPhi = genMetTrue->phi();

			me["hMETResolution_GenMETTrue"]->Fill( MET - genMET );
			me["hMETPhiResolution_GenMETTrue"]->Fill( TMath::ACos( TMath::Cos( METPhi - genMETPhi ) ) );

			//pfMET resolution in pfMET bins : Sam, Feb, 2012

			if (MET > 0 && MET < 20) me["hMETResolution_GenMETTrue_MET0to20"]->Fill( MET - genMET );
			else if (MET > 20 && MET < 40) me["hMETResolution_GenMETTrue_MET20to40"]->Fill( MET - genMET );
			else if (MET > 40 && MET < 60) me["hMETResolution_GenMETTrue_MET40to60"]->Fill( MET - genMET );
			else if (MET > 60 && MET < 80) me["hMETResolution_GenMETTrue_MET60to80"]->Fill( MET - genMET );
			else if (MET > 80 && MET <100) me["hMETResolution_GenMETTrue_MET80to100"]->Fill( MET - genMET );
			else if (MET >100 && MET <150) me["hMETResolution_GenMETTrue_MET100to150"]->Fill( MET - genMET );
			else if (MET >150 && MET <200) me["hMETResolution_GenMETTrue_MET150to200"]->Fill( MET - genMET );
			else if (MET >200 && MET <300) me["hMETResolution_GenMETTrue_MET200to300"]->Fill( MET - genMET );
			else if (MET >300 && MET <400) me["hMETResolution_GenMETTrue_MET300to400"]->Fill( MET - genMET );
			else if (MET >400 && MET <500) me["hMETResolution_GenMETTrue_MET400to500"]->Fill( MET - genMET );

			//This is an ugly hack I had to do.
			//I wanted to fill just one histogram at the endo the job. I tried
			//to do this in endjob() but it seems the file is closed before this
			//step.
			FillpfMETRes();

		} else {
			edm::LogInfo("OutputInfo") << " failed to retrieve data required by MET Task:  genMetTrue";
		}    


		edm::Handle<GenMETCollection> genCalo;
		iEvent.getByLabel("genMetCalo", genCalo);
		if (genCalo.isValid()) {
			const GenMETCollection *genmetcol = genCalo.product();
			const GenMET  *genMetCalo = &(genmetcol->front());
			const double genMET = genMetCalo->pt();
			const double genMETPhi = genMetCalo->phi();

			me["hMETResolution_GenMETCalo"]->Fill( MET - genMET );
			me["hMETPhiResolution_GenMETCalo"]->Fill( TMath::ACos( TMath::Cos( METPhi - genMETPhi ) ) );
		} else {
			edm::LogInfo("OutputInfo") << " failed to retrieve data required by MET Task:  genMetCalo";
		}    



	} else if (METType_ == "MET")
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
		const double SumET = met->sumEt();
		const double MET = met->pt();
		const double MEx = met->px();
		const double MEy = met->py();
		const double METPhi = met->phi();
		const double METSig = met->mEtSig();

		me["hMEx"]->Fill(MEx);
		me["hMEy"]->Fill(MEy);
		me["hMET"]->Fill(MET);
		me["hMETPhi"]->Fill(METPhi);
		me["hSumET"]->Fill(SumET);
		me["hMETSig"]->Fill(METSig);
		me["hNevents"]->Fill(0.5);

	} else if( METType_ == "TCMET" )
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

		//Event selection-----------------------------------------------------------------------

		edm::Handle< edm::ValueMap<reco::MuonMETCorrectionData> > tcMet_ValueMap_Handle;
		iEvent.getByLabel("muonTCMETValueMapProducer" , "muCorrData", tcMet_ValueMap_Handle);

		//count muons
		int nM = 0;

		for( unsigned int mus = 0; mus < muon_h->size() ; mus++ ) {

			reco::MuonRef muref( muon_h, mus);
			if( muref->pt() < 20 ) continue;

			reco::MuonMETCorrectionData muCorrData = (*tcMet_ValueMap_Handle)[muref];
			int type = muCorrData.type();

			if( type == 1 || type == 2 || type == 5 )  ++nM;
		}

		//count electrons
		int nE = 0;

		for( edm::View<reco::GsfElectron>::const_iterator eleit = electron_h->begin(); eleit != electron_h->end(); eleit++ ) {
			if( eleit->p4().pt() < 20 ) continue;  
			++nE;
		}

		if( strcmp( sample_.c_str() , "zmm" ) == 0 && nM != 2 ) return;

		if( strcmp( sample_.c_str() , "zee" ) == 0 && nE != 2 ) return;

		if( strcmp( sample_.c_str() , "ttbar" ) == 0 && ( nE + nM ) == 0 ) return;

		// Reconstructed TCMET Information                                                                                                     
		const double SumET = tcMet->sumEt();
		const double MET = tcMet->pt();
		const double MEx = tcMet->px();
		const double MEy = tcMet->py();
		const double METPhi = tcMet->phi();
		const double METSig = tcMet->mEtSig();

		me["hMEx"]->Fill(MEx);
		me["hMEy"]->Fill(MEy);
		me["hMET"]->Fill(MET);
		me["hMETPhi"]->Fill(METPhi);
		me["hSumET"]->Fill(SumET);
		me["hMETSig"]->Fill(METSig);
		me["hNevents"]->Fill(0.5);

		const double caloMET = caloMet->pt();
		const double caloMEx = caloMet->px();
		const double caloMEy = caloMet->py();

		me["hdMETx"]->Fill(caloMEx-MEx);
		me["hdMETy"]->Fill(caloMEy-MEy);
		me["hdMET"]->Fill(caloMET-MET);

		const unsigned int nTracks = track_h->size();
		unsigned int nCorrTracks = 0;
		unsigned int trackCount = 0;
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

		const float frac = (float)nCorrTracks / (float)nTracks;
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

		//edm::Handle< edm::ValueMap<reco::MuonMETCorrectionData> > tcMet_ValueMap_Handle;
		//iEvent.getByLabel("muonTCMETValueMapProducer" , "muCorrData", tcMet_ValueMap_Handle);

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
			const double genMET = genMetTrue->pt();
			const double genMETPhi = genMetTrue->phi();

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
			const double genMET = genMetCalo->pt();
			const double genMETPhi = genMetCalo->phi();

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
		const double SumET = corMetGlobalMuons->sumEt();
		const double MET = corMetGlobalMuons->pt();
		const double MEx = corMetGlobalMuons->px();
		const double MEy = corMetGlobalMuons->py();
		const double METPhi = corMetGlobalMuons->phi();
		const double METSig = corMetGlobalMuons->mEtSig();
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
			const double genMET = genMetTrue->pt();
			const double genMETPhi = genMetTrue->phi();

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
			const double genMET = genMetCalo->pt();
			const double genMETPhi = genMetCalo->phi();

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

//void METTester::endRun(const edm::Run& iRun, const edm::EventSetup& iSetup)
void METTester::FillpfMETRes()
{
	me["hMETResolution_GenMETTrue_METResolution"]->setBinContent(1, me["hMETResolution_GenMETTrue_MET0to20"]->getMean());
	me["hMETResolution_GenMETTrue_METResolution"]->setBinContent(2, me["hMETResolution_GenMETTrue_MET20to40"]->getMean());
	me["hMETResolution_GenMETTrue_METResolution"]->setBinContent(3, me["hMETResolution_GenMETTrue_MET40to60"]->getMean());
	me["hMETResolution_GenMETTrue_METResolution"]->setBinContent(4, me["hMETResolution_GenMETTrue_MET60to80"]->getMean());
	me["hMETResolution_GenMETTrue_METResolution"]->setBinContent(5, me["hMETResolution_GenMETTrue_MET80to100"]->getMean());
	me["hMETResolution_GenMETTrue_METResolution"]->setBinContent(6, me["hMETResolution_GenMETTrue_MET100to150"]->getMean());
	me["hMETResolution_GenMETTrue_METResolution"]->setBinContent(7, me["hMETResolution_GenMETTrue_MET150to200"]->getMean());
	me["hMETResolution_GenMETTrue_METResolution"]->setBinContent(8, me["hMETResolution_GenMETTrue_MET200to300"]->getMean());
	me["hMETResolution_GenMETTrue_METResolution"]->setBinContent(9, me["hMETResolution_GenMETTrue_MET300to400"]->getMean());
	me["hMETResolution_GenMETTrue_METResolution"]->setBinContent(10, me["hMETResolution_GenMETTrue_MET400to500"]->getMean());
	me["hMETResolution_GenMETTrue_METResolution"]->setBinError(1, me["hMETResolution_GenMETTrue_MET0to20"]->getRMS());
	me["hMETResolution_GenMETTrue_METResolution"]->setBinError(2, me["hMETResolution_GenMETTrue_MET20to40"]->getRMS());
	me["hMETResolution_GenMETTrue_METResolution"]->setBinError(3, me["hMETResolution_GenMETTrue_MET40to60"]->getRMS());
	me["hMETResolution_GenMETTrue_METResolution"]->setBinError(4, me["hMETResolution_GenMETTrue_MET60to80"]->getRMS());
	me["hMETResolution_GenMETTrue_METResolution"]->setBinError(5, me["hMETResolution_GenMETTrue_MET80to100"]->getRMS());
	me["hMETResolution_GenMETTrue_METResolution"]->setBinError(6, me["hMETResolution_GenMETTrue_MET100to150"]->getRMS());
	me["hMETResolution_GenMETTrue_METResolution"]->setBinError(7, me["hMETResolution_GenMETTrue_MET150to200"]->getRMS());
	me["hMETResolution_GenMETTrue_METResolution"]->setBinError(8, me["hMETResolution_GenMETTrue_MET200to300"]->getRMS());
	me["hMETResolution_GenMETTrue_METResolution"]->setBinError(9, me["hMETResolution_GenMETTrue_MET300to400"]->getRMS());
	me["hMETResolution_GenMETTrue_METResolution"]->setBinError(10, me["hMETResolution_GenMETTrue_MET400to500"]->getRMS());

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
