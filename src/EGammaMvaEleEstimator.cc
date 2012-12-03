#include <TFile.h>
#include "EgammaAnalysis/ElectronTools/interface/EGammaMvaEleEstimator.h"
#include <cmath>
#include <vector>
using namespace std;

#ifndef STANDALONE
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/Common/interface/RefToPtr.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/IPTools/interface/IPTools.h"
#include "EgammaAnalysis/ElectronTools/interface/ElectronEffectiveArea.h"
#include "DataFormats/Common/interface/RefToPtr.h"
using namespace reco;
#endif

//--------------------------------------------------------------------------------------------------
EGammaMvaEleEstimator::EGammaMvaEleEstimator() :
fMethodname("BDTG method"),
fisInitialized(kFALSE),
fMVAType(kTrig),
fUseBinnedVersion(kTRUE),
fNMVABins(0)
{
  // Constructor.  
}

//--------------------------------------------------------------------------------------------------
EGammaMvaEleEstimator::~EGammaMvaEleEstimator()
{
  for (unsigned int i=0;i<fTMVAReader.size(); ++i) {
    if (fTMVAReader[i]) delete fTMVAReader[i];
  }
}

//--------------------------------------------------------------------------------------------------
void EGammaMvaEleEstimator::initialize( std::string methodName,
                                       std::string weightsfile,
                                       EGammaMvaEleEstimator::MVAType type)
{
  
  std::vector<std::string> tempWeightFileVector;
  tempWeightFileVector.push_back(weightsfile);
  initialize(methodName,type,kFALSE,tempWeightFileVector);
}


//--------------------------------------------------------------------------------------------------
void EGammaMvaEleEstimator::initialize( std::string methodName,
                                       EGammaMvaEleEstimator::MVAType type,
                                       Bool_t useBinnedVersion,
				       std::vector<std::string> weightsfiles
  ) {

  //clean up first
  for (unsigned int i=0;i<fTMVAReader.size(); ++i) {
    if (fTMVAReader[i]) delete fTMVAReader[i];
  }
  fTMVAReader.clear();

  //initialize
  fisInitialized = kTRUE;
  fMVAType = type;
  fMethodname = methodName;
  fUseBinnedVersion = useBinnedVersion;

  //Define expected number of bins
  UInt_t ExpectedNBins = 0;
  if (!fUseBinnedVersion) {
    ExpectedNBins = 1;
  } else if (type == kTrig) {
    ExpectedNBins = 6;
  } else if (type == kNonTrig) {
    ExpectedNBins = 6;
  } else if (type == kIsoRings) {
    ExpectedNBins = 4;
  } else if (type == kTrigIDIsoCombined) {
    ExpectedNBins = 6;
  } else if (type == kTrigIDIsoCombinedPUCorrected) {
    ExpectedNBins = 6;
  }

  fNMVABins = ExpectedNBins;
  
  //Check number of weight files given
  if (fNMVABins != weightsfiles.size() ) {
    std::cout << "Error: Expected Number of bins = " << fNMVABins << " does not equal to weightsfiles.size() = " 
              << weightsfiles.size() << std::endl; 
 
   #ifndef STANDALONE
    assert(fNMVABins == weightsfiles.size());
   #endif 
  }

  //Loop over all bins
  for (unsigned int i=0;i<fNMVABins; ++i) {
  
    TMVA::Reader *tmpTMVAReader = new TMVA::Reader( "!Color:!Silent:Error" );  
    tmpTMVAReader->SetVerbose(kTRUE);
  
    if (type == kTrig) {
      // Pure tracking variables
      tmpTMVAReader->AddVariable("fbrem",           &fMVAVar_fbrem);
      tmpTMVAReader->AddVariable("kfchi2",          &fMVAVar_kfchi2);
      tmpTMVAReader->AddVariable("kfhits",          &fMVAVar_kfhits);
      tmpTMVAReader->AddVariable("gsfchi2",         &fMVAVar_gsfchi2);

      // Geometrical matchings
      tmpTMVAReader->AddVariable("deta",            &fMVAVar_deta);
      tmpTMVAReader->AddVariable("dphi",            &fMVAVar_dphi);
      tmpTMVAReader->AddVariable("detacalo",        &fMVAVar_detacalo);
    
      // Pure ECAL -> shower shapes
      tmpTMVAReader->AddVariable("see",             &fMVAVar_see);
      tmpTMVAReader->AddVariable("spp",             &fMVAVar_spp);
      tmpTMVAReader->AddVariable("etawidth",        &fMVAVar_etawidth);
      tmpTMVAReader->AddVariable("phiwidth",        &fMVAVar_phiwidth);
      tmpTMVAReader->AddVariable("e1x5e5x5",        &fMVAVar_OneMinusE1x5E5x5);
      tmpTMVAReader->AddVariable("R9",              &fMVAVar_R9);

      // Energy matching
      tmpTMVAReader->AddVariable("HoE",             &fMVAVar_HoE);
      tmpTMVAReader->AddVariable("EoP",             &fMVAVar_EoP); 
      tmpTMVAReader->AddVariable("IoEmIoP",         &fMVAVar_IoEmIoP);
      tmpTMVAReader->AddVariable("eleEoPout",       &fMVAVar_eleEoPout);
      if(i == 2 || i == 5) 
	tmpTMVAReader->AddVariable("PreShowerOverRaw",&fMVAVar_PreShowerOverRaw);
      
      if(!fUseBinnedVersion)
	tmpTMVAReader->AddVariable("PreShowerOverRaw",&fMVAVar_PreShowerOverRaw);

      // IP
      tmpTMVAReader->AddVariable("d0",              &fMVAVar_d0);
      tmpTMVAReader->AddVariable("ip3d",            &fMVAVar_ip3d);
    
      tmpTMVAReader->AddSpectator("eta",            &fMVAVar_eta);
      tmpTMVAReader->AddSpectator("pt",             &fMVAVar_pt);
    }
  
    if (type == kNonTrig) {
      // Pure tracking variables
      tmpTMVAReader->AddVariable("fbrem",           &fMVAVar_fbrem);
      tmpTMVAReader->AddVariable("kfchi2",          &fMVAVar_kfchi2);
      tmpTMVAReader->AddVariable("kfhits",          &fMVAVar_kfhits);
      tmpTMVAReader->AddVariable("gsfchi2",         &fMVAVar_gsfchi2);

      // Geometrical matchings
      tmpTMVAReader->AddVariable("deta",            &fMVAVar_deta);
      tmpTMVAReader->AddVariable("dphi",            &fMVAVar_dphi);
      tmpTMVAReader->AddVariable("detacalo",        &fMVAVar_detacalo);
    
      // Pure ECAL -> shower shapes
      tmpTMVAReader->AddVariable("see",             &fMVAVar_see);
      tmpTMVAReader->AddVariable("spp",             &fMVAVar_spp);
      tmpTMVAReader->AddVariable("etawidth",        &fMVAVar_etawidth);
      tmpTMVAReader->AddVariable("phiwidth",        &fMVAVar_phiwidth);
      tmpTMVAReader->AddVariable("e1x5e5x5",        &fMVAVar_OneMinusE1x5E5x5);
      tmpTMVAReader->AddVariable("R9",              &fMVAVar_R9);

      // Energy matching
      tmpTMVAReader->AddVariable("HoE",             &fMVAVar_HoE);
      tmpTMVAReader->AddVariable("EoP",             &fMVAVar_EoP); 
      tmpTMVAReader->AddVariable("IoEmIoP",         &fMVAVar_IoEmIoP);
      tmpTMVAReader->AddVariable("eleEoPout",       &fMVAVar_eleEoPout);
      if(i == 2 || i == 5) 
	tmpTMVAReader->AddVariable("PreShowerOverRaw",&fMVAVar_PreShowerOverRaw);
    
      if(!fUseBinnedVersion)
	tmpTMVAReader->AddVariable("PreShowerOverRaw",&fMVAVar_PreShowerOverRaw);

      tmpTMVAReader->AddSpectator("eta",            &fMVAVar_eta);
      tmpTMVAReader->AddSpectator("pt",             &fMVAVar_pt);
    }

    if (type == kIsoRings) {
      tmpTMVAReader->AddVariable( "ChargedIso_DR0p0To0p1",         &fMVAVar_ChargedIso_DR0p0To0p1        );
      tmpTMVAReader->AddVariable( "ChargedIso_DR0p1To0p2",         &fMVAVar_ChargedIso_DR0p1To0p2        );
      tmpTMVAReader->AddVariable( "ChargedIso_DR0p2To0p3",         &fMVAVar_ChargedIso_DR0p2To0p3        );
      tmpTMVAReader->AddVariable( "ChargedIso_DR0p3To0p4",         &fMVAVar_ChargedIso_DR0p3To0p4        );
      tmpTMVAReader->AddVariable( "ChargedIso_DR0p4To0p5",         &fMVAVar_ChargedIso_DR0p4To0p5        );
      tmpTMVAReader->AddVariable( "GammaIso_DR0p0To0p1",           &fMVAVar_GammaIso_DR0p0To0p1          );
      tmpTMVAReader->AddVariable( "GammaIso_DR0p1To0p2",           &fMVAVar_GammaIso_DR0p1To0p2          );
      tmpTMVAReader->AddVariable( "GammaIso_DR0p2To0p3",           &fMVAVar_GammaIso_DR0p2To0p3          );
      tmpTMVAReader->AddVariable( "GammaIso_DR0p3To0p4",           &fMVAVar_GammaIso_DR0p3To0p4          );
      tmpTMVAReader->AddVariable( "GammaIso_DR0p4To0p5",           &fMVAVar_GammaIso_DR0p4To0p5          );
      tmpTMVAReader->AddVariable( "NeutralHadronIso_DR0p0To0p1",   &fMVAVar_NeutralHadronIso_DR0p0To0p1  );
      tmpTMVAReader->AddVariable( "NeutralHadronIso_DR0p1To0p2",   &fMVAVar_NeutralHadronIso_DR0p1To0p2  );
      tmpTMVAReader->AddVariable( "NeutralHadronIso_DR0p2To0p3",   &fMVAVar_NeutralHadronIso_DR0p2To0p3  );
      tmpTMVAReader->AddVariable( "NeutralHadronIso_DR0p3To0p4",   &fMVAVar_NeutralHadronIso_DR0p3To0p4  );
      tmpTMVAReader->AddVariable( "NeutralHadronIso_DR0p4To0p5",   &fMVAVar_NeutralHadronIso_DR0p4To0p5  );
      tmpTMVAReader->AddSpectator("eta",            &fMVAVar_eta);
      tmpTMVAReader->AddSpectator("pt",             &fMVAVar_pt);
    }
  

    if (type == kTrigIDIsoCombinedPUCorrected) {

      // Pure tracking variables
      tmpTMVAReader->AddVariable("fbrem",                      &fMVAVar_fbrem);
      tmpTMVAReader->AddVariable("kfchi2",                     &fMVAVar_kfchi2);
      tmpTMVAReader->AddVariable("kflayers",                   &fMVAVar_kfhits);
      tmpTMVAReader->AddVariable("gsfchi2",                    &fMVAVar_gsfchi2);

      // Geometrical matchings
      tmpTMVAReader->AddVariable("deta",                       &fMVAVar_deta);
      tmpTMVAReader->AddVariable("dphi",                       &fMVAVar_dphi);
      tmpTMVAReader->AddVariable("detacalo",                   &fMVAVar_detacalo);
    
      // Pure ECAL -> shower shapes
      tmpTMVAReader->AddVariable("see",                        &fMVAVar_see);
      tmpTMVAReader->AddVariable("spp",                        &fMVAVar_spp);
      tmpTMVAReader->AddVariable("etawidth",                   &fMVAVar_etawidth);
      tmpTMVAReader->AddVariable("phiwidth",                   &fMVAVar_phiwidth);
      tmpTMVAReader->AddVariable("OneMinusSeedE1x5OverE5x5",   &fMVAVar_OneMinusE1x5E5x5);
      tmpTMVAReader->AddVariable("R9",                         &fMVAVar_R9);

      // Energy matching
      tmpTMVAReader->AddVariable("HoE",                        &fMVAVar_HoE);
      tmpTMVAReader->AddVariable("EoP",                        &fMVAVar_EoP); 
      tmpTMVAReader->AddVariable("IoEmIoP",                    &fMVAVar_IoEmIoP);
      tmpTMVAReader->AddVariable("EEleoPout",                  &fMVAVar_eleEoPout);
      if(i == 2 || i == 5) {
	tmpTMVAReader->AddVariable("PreShowerOverRaw",&fMVAVar_PreShowerOverRaw);
      }
      if(!fUseBinnedVersion) {
	tmpTMVAReader->AddVariable("PreShowerOverRaw",&fMVAVar_PreShowerOverRaw);
      }

      // IP
      tmpTMVAReader->AddVariable("d0",              &fMVAVar_d0);
      tmpTMVAReader->AddVariable("ip3d",            &fMVAVar_ip3d);

      //isolation variables
      tmpTMVAReader->AddVariable( "ChargedIso_DR0p0To0p1",         &fMVAVar_ChargedIso_DR0p0To0p1        );
      tmpTMVAReader->AddVariable( "ChargedIso_DR0p1To0p2",         &fMVAVar_ChargedIso_DR0p1To0p2        );
      tmpTMVAReader->AddVariable( "ChargedIso_DR0p2To0p3",         &fMVAVar_ChargedIso_DR0p2To0p3        );
      tmpTMVAReader->AddVariable( "ChargedIso_DR0p3To0p4",         &fMVAVar_ChargedIso_DR0p3To0p4        );
      tmpTMVAReader->AddVariable( "ChargedIso_DR0p4To0p5",         &fMVAVar_ChargedIso_DR0p4To0p5        );
      tmpTMVAReader->AddVariable( "GammaIso_DR0p0To0p1",           &fMVAVar_GammaIso_DR0p0To0p1          );
      tmpTMVAReader->AddVariable( "GammaIso_DR0p1To0p2",           &fMVAVar_GammaIso_DR0p1To0p2          );
      tmpTMVAReader->AddVariable( "GammaIso_DR0p2To0p3",           &fMVAVar_GammaIso_DR0p2To0p3          );
      tmpTMVAReader->AddVariable( "GammaIso_DR0p3To0p4",           &fMVAVar_GammaIso_DR0p3To0p4          );
      tmpTMVAReader->AddVariable( "GammaIso_DR0p4To0p5",           &fMVAVar_GammaIso_DR0p4To0p5          );
      tmpTMVAReader->AddVariable( "NeutralHadronIso_DR0p0To0p1",   &fMVAVar_NeutralHadronIso_DR0p0To0p1  );
      tmpTMVAReader->AddVariable( "NeutralHadronIso_DR0p1To0p2",   &fMVAVar_NeutralHadronIso_DR0p1To0p2  );
      tmpTMVAReader->AddVariable( "NeutralHadronIso_DR0p2To0p3",   &fMVAVar_NeutralHadronIso_DR0p2To0p3  );
      tmpTMVAReader->AddVariable( "NeutralHadronIso_DR0p3To0p4",   &fMVAVar_NeutralHadronIso_DR0p3To0p4  );
      tmpTMVAReader->AddVariable( "NeutralHadronIso_DR0p4To0p5",   &fMVAVar_NeutralHadronIso_DR0p4To0p5  );

      //spectators
      tmpTMVAReader->AddSpectator("eta",            &fMVAVar_eta);
      tmpTMVAReader->AddSpectator("pt",             &fMVAVar_pt);

    }

    if (type == kTrigIDIsoCombined) {

      // Pure tracking variables
      tmpTMVAReader->AddVariable("fbrem",                      &fMVAVar_fbrem);
      tmpTMVAReader->AddVariable("kfchi2",                     &fMVAVar_kfchi2);
      tmpTMVAReader->AddVariable("kflayers",                   &fMVAVar_kfhits);
      tmpTMVAReader->AddVariable("gsfchi2",                    &fMVAVar_gsfchi2);
 
      // Geometrical matchings
      tmpTMVAReader->AddVariable("deta",                       &fMVAVar_deta);
      tmpTMVAReader->AddVariable("dphi",                       &fMVAVar_dphi);
      tmpTMVAReader->AddVariable("detacalo",                   &fMVAVar_detacalo);
    
      // Pure ECAL -> shower shapes
      tmpTMVAReader->AddVariable("see",                        &fMVAVar_see);
      tmpTMVAReader->AddVariable("spp",                        &fMVAVar_spp);
      tmpTMVAReader->AddVariable("etawidth",                   &fMVAVar_etawidth);
      tmpTMVAReader->AddVariable("phiwidth",                   &fMVAVar_phiwidth);
      tmpTMVAReader->AddVariable("OneMinusSeedE1x5OverE5x5",   &fMVAVar_OneMinusE1x5E5x5);
      tmpTMVAReader->AddVariable("R9",                         &fMVAVar_R9); 

      // Energy matching
      tmpTMVAReader->AddVariable("HoE",                        &fMVAVar_HoE);
      tmpTMVAReader->AddVariable("EoP",                        &fMVAVar_EoP); 
      tmpTMVAReader->AddVariable("IoEmIoP",                    &fMVAVar_IoEmIoP);
      tmpTMVAReader->AddVariable("EEleoPout",                  &fMVAVar_eleEoPout);
      if(i == 2 || i == 5) {
	tmpTMVAReader->AddVariable("PreShowerOverRaw",&fMVAVar_PreShowerOverRaw);
      }
      if(!fUseBinnedVersion) {
	tmpTMVAReader->AddVariable("PreShowerOverRaw",&fMVAVar_PreShowerOverRaw);
      }

      // IP
      tmpTMVAReader->AddVariable("d0",              &fMVAVar_d0);
      tmpTMVAReader->AddVariable("ip3d",            &fMVAVar_ip3d);

      //isolation variables
      tmpTMVAReader->AddVariable( "ChargedIso_DR0p0To0p1",         &fMVAVar_ChargedIso_DR0p0To0p1        );
      tmpTMVAReader->AddVariable( "ChargedIso_DR0p1To0p2",         &fMVAVar_ChargedIso_DR0p1To0p2        );
      tmpTMVAReader->AddVariable( "ChargedIso_DR0p2To0p3",         &fMVAVar_ChargedIso_DR0p2To0p3        );
      tmpTMVAReader->AddVariable( "ChargedIso_DR0p3To0p4",         &fMVAVar_ChargedIso_DR0p3To0p4        );
      tmpTMVAReader->AddVariable( "ChargedIso_DR0p4To0p5",         &fMVAVar_ChargedIso_DR0p4To0p5        );
      tmpTMVAReader->AddVariable( "GammaIso_DR0p0To0p1",           &fMVAVar_GammaIso_DR0p0To0p1          );
      tmpTMVAReader->AddVariable( "GammaIso_DR0p1To0p2",           &fMVAVar_GammaIso_DR0p1To0p2          );
      tmpTMVAReader->AddVariable( "GammaIso_DR0p2To0p3",           &fMVAVar_GammaIso_DR0p2To0p3          );
      tmpTMVAReader->AddVariable( "GammaIso_DR0p3To0p4",           &fMVAVar_GammaIso_DR0p3To0p4          );
      tmpTMVAReader->AddVariable( "GammaIso_DR0p4To0p5",           &fMVAVar_GammaIso_DR0p4To0p5          );
      tmpTMVAReader->AddVariable( "NeutralHadronIso_DR0p0To0p1",   &fMVAVar_NeutralHadronIso_DR0p0To0p1  );
      tmpTMVAReader->AddVariable( "NeutralHadronIso_DR0p1To0p2",   &fMVAVar_NeutralHadronIso_DR0p1To0p2  );
      tmpTMVAReader->AddVariable( "NeutralHadronIso_DR0p2To0p3",   &fMVAVar_NeutralHadronIso_DR0p2To0p3  );
      tmpTMVAReader->AddVariable( "NeutralHadronIso_DR0p3To0p4",   &fMVAVar_NeutralHadronIso_DR0p3To0p4  );
      tmpTMVAReader->AddVariable( "NeutralHadronIso_DR0p4To0p5",   &fMVAVar_NeutralHadronIso_DR0p4To0p5  );
      tmpTMVAReader->AddVariable( "rho",   &fMVAVar_rho );

      //spectators
      tmpTMVAReader->AddSpectator("eta",            &fMVAVar_eta);
      tmpTMVAReader->AddSpectator("pt",             &fMVAVar_pt);

    }



    tmpTMVAReader->BookMVA(fMethodname , weightsfiles[i]);
    std::cout << "MVABin " << i << " : MethodName = " << fMethodname 
              << " , type == " << type << " , "
              << "Load weights file : " << weightsfiles[i] 
              << std::endl;
    fTMVAReader.push_back(tmpTMVAReader);
  }
  std::cout << "Electron ID MVA Completed\n";

}


//--------------------------------------------------------------------------------------------------
UInt_t EGammaMvaEleEstimator::GetMVABin( double eta, double pt) const {
  
    //Default is to return the first bin
    unsigned int bin = 0;

    if (fMVAType == EGammaMvaEleEstimator::kIsoRings) {
      if (pt < 10 && fabs(eta) < 1.479) bin = 0;
      if (pt < 10 && fabs(eta) >= 1.479) bin = 1;
      if (pt >= 10 && fabs(eta) < 1.479) bin = 2;
      if (pt >= 10 && fabs(eta) >= 1.479) bin = 3;
    }

    if (fMVAType == EGammaMvaEleEstimator::kNonTrig ) {
      bin = 0;
      if (pt < 10 && fabs(eta) < 0.8) bin = 0;
      if (pt < 10 && fabs(eta) >= 0.8 && fabs(eta) < 1.479 ) bin = 1;
      if (pt < 10 && fabs(eta) >= 1.479) bin = 2;
      if (pt >= 10 && fabs(eta) < 0.8) bin = 3;
      if (pt >= 10 && fabs(eta) >= 0.8 && fabs(eta) < 1.479 ) bin = 4;
      if (pt >= 10 && fabs(eta) >= 1.479) bin = 5;
    }


    if (fMVAType == EGammaMvaEleEstimator::kTrig || 
        fMVAType == EGammaMvaEleEstimator::kTrigIDIsoCombined || 
        fMVAType == EGammaMvaEleEstimator::kTrigIDIsoCombinedPUCorrected
      ) {
      bin = 0;
      if (pt < 20 && fabs(eta) < 0.8) bin = 0;
      if (pt < 20 && fabs(eta) >= 0.8 && fabs(eta) < 1.479 ) bin = 1;
      if (pt < 20 && fabs(eta) >= 1.479) bin = 2;
      if (pt >= 20 && fabs(eta) < 0.8) bin = 3;
      if (pt >= 20 && fabs(eta) >= 0.8 && fabs(eta) < 1.479 ) bin = 4;
      if (pt >= 20 && fabs(eta) >= 1.479) bin = 5;
    }

 

    return bin;
}


//--------------------------------------------------------------------------------------------------
Double_t EGammaMvaEleEstimator::mvaValue(Double_t fbrem, 
					Double_t kfchi2,
					Int_t    kfhits,
					Double_t gsfchi2,
					Double_t deta,
					Double_t dphi,
					Double_t detacalo,
					//Double_t dphicalo,
					Double_t see,
					Double_t spp,
					Double_t etawidth,
					Double_t phiwidth,
					Double_t OneMinusE1x5E5x5,
					Double_t R9,
					//Int_t    nbrems,
					Double_t HoE,
					Double_t EoP,
					Double_t IoEmIoP,
					Double_t eleEoPout,
					Double_t PreShowerOverRaw,
					//Double_t EoPout,
					Double_t d0,
					Double_t ip3d,
					Double_t eta,
					Double_t pt,
					Bool_t printDebug) {
  
  if (!fisInitialized) { 
    std::cout << "Error: EGammaMvaEleEstimator not properly initialized.\n"; 
    return -9999;
  }

  fMVAVar_fbrem           = fbrem; 
  fMVAVar_kfchi2          = kfchi2;
  fMVAVar_kfhits          = float(kfhits);   // BTD does not support int variables
  fMVAVar_gsfchi2         = gsfchi2;

  fMVAVar_deta            = deta;
  fMVAVar_dphi            = dphi;
  fMVAVar_detacalo        = detacalo;


  fMVAVar_see             = see;
  fMVAVar_spp             = spp;
  fMVAVar_etawidth        = etawidth;
  fMVAVar_phiwidth        = phiwidth;
  fMVAVar_OneMinusE1x5E5x5        = OneMinusE1x5E5x5;
  fMVAVar_R9              = R9;


  fMVAVar_HoE             = HoE;
  fMVAVar_EoP             = EoP;
  fMVAVar_IoEmIoP         = IoEmIoP;
  fMVAVar_eleEoPout       = eleEoPout;
  fMVAVar_PreShowerOverRaw= PreShowerOverRaw;

  fMVAVar_d0              = d0;
  fMVAVar_ip3d            = ip3d;
  fMVAVar_eta             = eta;
  fMVAVar_pt              = pt;


  bindVariables();
  Double_t mva = -9999;  
  if (fUseBinnedVersion) {
    mva = fTMVAReader[GetMVABin(fMVAVar_eta,fMVAVar_pt)]->EvaluateMVA(fMethodname);
  } else {
    mva = fTMVAReader[0]->EvaluateMVA(fMethodname);
  }

  if(printDebug) {
    cout << " *** Inside the class fMethodname " << fMethodname << endl;
    cout << " fbrem " <<  fMVAVar_fbrem  
      	 << " kfchi2 " << fMVAVar_kfchi2  
	 << " mykfhits " << fMVAVar_kfhits  
	 << " gsfchi2 " << fMVAVar_gsfchi2  
	 << " deta " <<  fMVAVar_deta  
	 << " dphi " << fMVAVar_dphi  
      	 << " detacalo " << fMVAVar_detacalo  
	 << " see " << fMVAVar_see  
	 << " spp " << fMVAVar_spp  
	 << " etawidth " << fMVAVar_etawidth  
	 << " phiwidth " << fMVAVar_phiwidth  
	 << " OneMinusE1x5E5x5 " << fMVAVar_OneMinusE1x5E5x5  
	 << " R9 " << fMVAVar_R9  
	 << " HoE " << fMVAVar_HoE  
	 << " EoP " << fMVAVar_EoP  
	 << " IoEmIoP " << fMVAVar_IoEmIoP  
	 << " eleEoPout " << fMVAVar_eleEoPout  
	 << " PreShowerOverRaw " << fMVAVar_PreShowerOverRaw  
	 << " d0 " << fMVAVar_d0  
	 << " ip3d " << fMVAVar_ip3d  
	 << " eta " << fMVAVar_eta  
	 << " pt " << fMVAVar_pt << endl;
    cout << " ### MVA " << mva << endl;
  }


  return mva;
}
//--------------------------------------------------------------------------------------------------
Double_t EGammaMvaEleEstimator::mvaValue(Double_t fbrem, 
					Double_t kfchi2,
					Int_t    kfhits,
					Double_t gsfchi2,
					Double_t deta,
					Double_t dphi,
					Double_t detacalo,
					//Double_t dphicalo,
					Double_t see,
					Double_t spp,
					Double_t etawidth,
					Double_t phiwidth,
					Double_t OneMinusE1x5E5x5,
					Double_t R9,
					//Int_t    nbrems,
					Double_t HoE,
					Double_t EoP,
					Double_t IoEmIoP,
					Double_t eleEoPout,
					Double_t PreShowerOverRaw,
					//Double_t EoPout,
					Double_t eta,
					Double_t pt,
					Bool_t printDebug) {
  
  if (!fisInitialized) { 
    std::cout << "Error: EGammaMvaEleEstimator not properly initialized.\n"; 
    return -9999;
  }

  fMVAVar_fbrem           = fbrem; 
  fMVAVar_kfchi2          = kfchi2;
  fMVAVar_kfhits          = float(kfhits);   // BTD does not support int variables
  fMVAVar_gsfchi2         = gsfchi2;

  fMVAVar_deta            = deta;
  fMVAVar_dphi            = dphi;
  fMVAVar_detacalo        = detacalo;


  fMVAVar_see             = see;
  fMVAVar_spp             = spp;
  fMVAVar_etawidth        = etawidth;
  fMVAVar_phiwidth        = phiwidth;
  fMVAVar_OneMinusE1x5E5x5        = OneMinusE1x5E5x5;
  fMVAVar_R9              = R9;


  fMVAVar_HoE             = HoE;
  fMVAVar_EoP             = EoP;
  fMVAVar_IoEmIoP         = IoEmIoP;
  fMVAVar_eleEoPout       = eleEoPout;
  fMVAVar_PreShowerOverRaw= PreShowerOverRaw;

  fMVAVar_eta             = eta;
  fMVAVar_pt              = pt;


  bindVariables();
  Double_t mva = -9999;  
  if (fUseBinnedVersion) {
    mva = fTMVAReader[GetMVABin(fMVAVar_eta,fMVAVar_pt)]->EvaluateMVA(fMethodname);
  } else {
    mva = fTMVAReader[0]->EvaluateMVA(fMethodname);
  }



  if(printDebug) {
    cout << " *** Inside the class fMethodname " << fMethodname << endl;
    cout << " fbrem " <<  fMVAVar_fbrem  
      	 << " kfchi2 " << fMVAVar_kfchi2  
	 << " mykfhits " << fMVAVar_kfhits  
	 << " gsfchi2 " << fMVAVar_gsfchi2  
	 << " deta " <<  fMVAVar_deta  
	 << " dphi " << fMVAVar_dphi  
      	 << " detacalo " << fMVAVar_detacalo  
	 << " see " << fMVAVar_see  
	 << " spp " << fMVAVar_spp  
	 << " etawidth " << fMVAVar_etawidth  
	 << " phiwidth " << fMVAVar_phiwidth  
	 << " OneMinusE1x5E5x5 " << fMVAVar_OneMinusE1x5E5x5  
	 << " R9 " << fMVAVar_R9  
	 << " HoE " << fMVAVar_HoE  
	 << " EoP " << fMVAVar_EoP  
	 << " IoEmIoP " << fMVAVar_IoEmIoP  
	 << " eleEoPout " << fMVAVar_eleEoPout  
	 << " PreShowerOverRaw " << fMVAVar_PreShowerOverRaw  
	 << " eta " << fMVAVar_eta  
	 << " pt " << fMVAVar_pt << endl;
    cout << " ### MVA " << mva << endl;
  }


  return mva;
}



//--------------------------------------------------------------------------------------------------
Double_t EGammaMvaEleEstimator::IDIsoCombinedMvaValue(Double_t fbrem, 
                                                      Double_t kfchi2,
                                                      Int_t    kfhits,
                                                      Double_t gsfchi2,
                                                      Double_t deta,
                                                      Double_t dphi,
                                                      Double_t detacalo,
                                                      Double_t see,
                                                      Double_t spp,
                                                      Double_t etawidth,
                                                      Double_t phiwidth,
                                                      Double_t OneMinusE1x5E5x5,
                                                      Double_t R9,
                                                      Double_t HoE,
                                                      Double_t EoP,
                                                      Double_t IoEmIoP,
                                                      Double_t eleEoPout,
                                                      Double_t PreShowerOverRaw,
                                                      Double_t d0,
                                                      Double_t ip3d,
                                                      Double_t ChargedIso_DR0p0To0p1,
                                                      Double_t ChargedIso_DR0p1To0p2,
                                                      Double_t ChargedIso_DR0p2To0p3,
                                                      Double_t ChargedIso_DR0p3To0p4,
                                                      Double_t ChargedIso_DR0p4To0p5,
                                                      Double_t GammaIso_DR0p0To0p1,
                                                      Double_t GammaIso_DR0p1To0p2,
                                                      Double_t GammaIso_DR0p2To0p3,
                                                      Double_t GammaIso_DR0p3To0p4,
                                                      Double_t GammaIso_DR0p4To0p5,
                                                      Double_t NeutralHadronIso_DR0p0To0p1,
                                                      Double_t NeutralHadronIso_DR0p1To0p2,
                                                      Double_t NeutralHadronIso_DR0p2To0p3,
                                                      Double_t NeutralHadronIso_DR0p3To0p4,
                                                      Double_t NeutralHadronIso_DR0p4To0p5,
                                                      Double_t Rho,
                                                      Double_t eta,
                                                      Double_t pt,
                                                      Bool_t printDebug) {
  
  if (!fisInitialized) { 
    std::cout << "Error: EGammaMvaEleEstimator not properly initialized.\n"; 
    return -9999;
  }

  fMVAVar_fbrem           = ( fbrem < -1.0 ) ? -1.0 : fbrem; 
  fMVAVar_kfchi2          = ( kfchi2 > 10 ) ? 10 : kfchi2;
  fMVAVar_kfhits          = float(kfhits);   // BTD does not support int variables
  fMVAVar_gsfchi2         = ( gsfchi2 > 200 ) ? 200 : gsfchi2;
  fMVAVar_deta            = ( fabs(deta) > 0.06 ) ? 0.06 : fabs(deta);
  fMVAVar_dphi            = dphi;
  fMVAVar_detacalo        = detacalo;

  fMVAVar_see             = see;
  fMVAVar_spp             = spp;
  fMVAVar_etawidth        = etawidth;
  fMVAVar_phiwidth        = phiwidth;
  fMVAVar_OneMinusE1x5E5x5= max(min(double(OneMinusE1x5E5x5),2.0),-1.0);
  fMVAVar_R9              = (R9 > 5) ? 5: R9;

  fMVAVar_HoE             = HoE;
  fMVAVar_EoP             = (EoP > 20) ? 20 : EoP;
  fMVAVar_IoEmIoP         = IoEmIoP;
  fMVAVar_eleEoPout       = (eleEoPout > 20) ? 20 : eleEoPout;
  fMVAVar_PreShowerOverRaw= PreShowerOverRaw;

  fMVAVar_d0              = d0;
  fMVAVar_ip3d            = ip3d;

  fMVAVar_ChargedIso_DR0p0To0p1 = ChargedIso_DR0p0To0p1;
  fMVAVar_ChargedIso_DR0p1To0p2 = ChargedIso_DR0p1To0p2;
  fMVAVar_ChargedIso_DR0p2To0p3 = ChargedIso_DR0p2To0p3;
  fMVAVar_ChargedIso_DR0p3To0p4 = ChargedIso_DR0p3To0p4;
  fMVAVar_ChargedIso_DR0p4To0p5 = ChargedIso_DR0p4To0p5;
  fMVAVar_GammaIso_DR0p0To0p1 = GammaIso_DR0p0To0p1;
  fMVAVar_GammaIso_DR0p1To0p2 = GammaIso_DR0p1To0p2;
  fMVAVar_GammaIso_DR0p2To0p3 = GammaIso_DR0p2To0p3;
  fMVAVar_GammaIso_DR0p3To0p4 = GammaIso_DR0p3To0p4;
  fMVAVar_GammaIso_DR0p4To0p5 = GammaIso_DR0p4To0p5;
  fMVAVar_NeutralHadronIso_DR0p0To0p1 = NeutralHadronIso_DR0p0To0p1;
  fMVAVar_NeutralHadronIso_DR0p1To0p2 = NeutralHadronIso_DR0p1To0p2;
  fMVAVar_NeutralHadronIso_DR0p2To0p3 = NeutralHadronIso_DR0p2To0p3;
  fMVAVar_NeutralHadronIso_DR0p3To0p4 = NeutralHadronIso_DR0p3To0p4;
  fMVAVar_NeutralHadronIso_DR0p4To0p5 = NeutralHadronIso_DR0p4To0p5;

  fMVAVar_rho             = Rho;
  fMVAVar_eta             = eta;
  fMVAVar_pt              = pt;

  Double_t mva = -9999;  
  if (fUseBinnedVersion) {
    mva = fTMVAReader[GetMVABin(fMVAVar_eta,fMVAVar_pt)]->EvaluateMVA(fMethodname);
  } else {
    mva = fTMVAReader[0]->EvaluateMVA(fMethodname);
  }

  if(printDebug) {
    cout << " *** Inside the class fMethodname " << fMethodname << endl;
    cout << " fbrem " <<  fMVAVar_fbrem  
      	 << " kfchi2 " << fMVAVar_kfchi2  
	 << " mykfhits " << fMVAVar_kfhits  
	 << " gsfchi2 " << fMVAVar_gsfchi2  
	 << " deta " <<  fMVAVar_deta  
	 << " dphi " << fMVAVar_dphi  
      	 << " detacalo " << fMVAVar_detacalo  
	 << " see " << fMVAVar_see  
	 << " spp " << fMVAVar_spp  
	 << " etawidth " << fMVAVar_etawidth  
	 << " phiwidth " << fMVAVar_phiwidth  
	 << " OneMinusE1x5E5x5 " << fMVAVar_OneMinusE1x5E5x5  
	 << " R9 " << fMVAVar_R9  
	 << " HoE " << fMVAVar_HoE  
	 << " EoP " << fMVAVar_EoP  
	 << " IoEmIoP " << fMVAVar_IoEmIoP  
	 << " eleEoPout " << fMVAVar_eleEoPout  
	 << " PreShowerOverRaw " << fMVAVar_PreShowerOverRaw  
	 << " d0 " << fMVAVar_d0  
	 << " ip3d " << fMVAVar_ip3d  
         << " ChargedIso_DR0p0To0p1 " <<  ChargedIso_DR0p0To0p1
         << " ChargedIso_DR0p1To0p2 " <<  ChargedIso_DR0p1To0p2
         << " ChargedIso_DR0p2To0p3 " <<  ChargedIso_DR0p2To0p3
         << " ChargedIso_DR0p3To0p4 " <<  ChargedIso_DR0p3To0p4
         << " ChargedIso_DR0p4To0p5 " <<  ChargedIso_DR0p4To0p5
         << " GammaIso_DR0p0To0p1 " <<  GammaIso_DR0p0To0p1
         << " GammaIso_DR0p1To0p2 " <<  GammaIso_DR0p1To0p2
         << " GammaIso_DR0p2To0p3 " <<  GammaIso_DR0p2To0p3
         << " GammaIso_DR0p3To0p4 " <<  GammaIso_DR0p3To0p4
         << " GammaIso_DR0p4To0p5 " <<  GammaIso_DR0p4To0p5
         << " NeutralHadronIso_DR0p0To0p1 " <<  NeutralHadronIso_DR0p0To0p1
         << " NeutralHadronIso_DR0p1To0p2 " <<  NeutralHadronIso_DR0p1To0p2
         << " NeutralHadronIso_DR0p2To0p3 " <<  NeutralHadronIso_DR0p2To0p3
         << " NeutralHadronIso_DR0p3To0p4 " <<  NeutralHadronIso_DR0p3To0p4
         << " NeutralHadronIso_DR0p4To0p5 " <<  NeutralHadronIso_DR0p4To0p5
         << " Rho " <<  Rho
	 << " eta " << fMVAVar_eta  
	 << " pt " << fMVAVar_pt << endl;
    cout << " ### MVA " << mva << endl;
  }

  return mva;
}





#ifndef STANDALONE
Double_t EGammaMvaEleEstimator::isoMvaValue(Double_t Pt,
                                            Double_t Eta,
                                            Double_t Rho,
                                            ElectronEffectiveArea::ElectronEffectiveAreaTarget EATarget,
                                            Double_t ChargedIso_DR0p0To0p1,
                                            Double_t ChargedIso_DR0p1To0p2,
                                            Double_t ChargedIso_DR0p2To0p3,
                                            Double_t ChargedIso_DR0p3To0p4,
                                            Double_t ChargedIso_DR0p4To0p5,
                                            Double_t GammaIso_DR0p0To0p1,
                                            Double_t GammaIso_DR0p1To0p2,
                                            Double_t GammaIso_DR0p2To0p3,
                                            Double_t GammaIso_DR0p3To0p4,
                                            Double_t GammaIso_DR0p4To0p5,
                                            Double_t NeutralHadronIso_DR0p0To0p1,
                                            Double_t NeutralHadronIso_DR0p1To0p2,
                                            Double_t NeutralHadronIso_DR0p2To0p3,
                                            Double_t NeutralHadronIso_DR0p3To0p4,
                                            Double_t NeutralHadronIso_DR0p4To0p5,
                                            Bool_t printDebug) {

  if (!fisInitialized) { 
    std::cout << "Error: EGammaMvaEleEstimator not properly initialized.\n"; 
    return -9999;
  }

  fMVAVar_ChargedIso_DR0p0To0p1   = TMath::Min((ChargedIso_DR0p0To0p1)/Pt, 2.5);
  fMVAVar_ChargedIso_DR0p1To0p2   = TMath::Min((ChargedIso_DR0p1To0p2)/Pt, 2.5);
  fMVAVar_ChargedIso_DR0p2To0p3 = TMath::Min((ChargedIso_DR0p2To0p3)/Pt, 2.5);
  fMVAVar_ChargedIso_DR0p3To0p4 = TMath::Min((ChargedIso_DR0p3To0p4)/Pt, 2.5);
  fMVAVar_ChargedIso_DR0p4To0p5 = TMath::Min((ChargedIso_DR0p4To0p5)/Pt, 2.5); 
  fMVAVar_GammaIso_DR0p0To0p1 = TMath::Max(TMath::Min((GammaIso_DR0p0To0p1 - Rho*ElectronEffectiveArea::GetElectronEffectiveArea(ElectronEffectiveArea::kEleGammaIsoDR0p0To0p1, Eta, EATarget))/Pt, 2.5), 0.0);
  fMVAVar_GammaIso_DR0p1To0p2 = TMath::Max(TMath::Min((GammaIso_DR0p1To0p2 - Rho*ElectronEffectiveArea::GetElectronEffectiveArea(ElectronEffectiveArea::kEleGammaIsoDR0p1To0p2, Eta, EATarget))/Pt, 2.5), 0.0);
  fMVAVar_GammaIso_DR0p2To0p3 = TMath::Max(TMath::Min((GammaIso_DR0p2To0p3 - Rho*ElectronEffectiveArea::GetElectronEffectiveArea(ElectronEffectiveArea::kEleGammaIsoDR0p2To0p3, Eta, EATarget))/Pt, 2.5), 0.0);
  fMVAVar_GammaIso_DR0p3To0p4 = TMath::Max(TMath::Min((GammaIso_DR0p3To0p4 - Rho*ElectronEffectiveArea::GetElectronEffectiveArea(ElectronEffectiveArea::kEleGammaIsoDR0p3To0p4, Eta, EATarget))/Pt, 2.5), 0.0);
  fMVAVar_GammaIso_DR0p4To0p5 = TMath::Max(TMath::Min((GammaIso_DR0p4To0p5 - Rho*ElectronEffectiveArea::GetElectronEffectiveArea(ElectronEffectiveArea::kEleGammaIsoDR0p4To0p5, Eta, EATarget))/Pt, 2.5), 0.0);
  fMVAVar_NeutralHadronIso_DR0p0To0p1 = TMath::Max(TMath::Min((NeutralHadronIso_DR0p0To0p1 - Rho*ElectronEffectiveArea::GetElectronEffectiveArea(ElectronEffectiveArea::kEleNeutralHadronIsoDR0p0To0p1, Eta, EATarget))/Pt, 2.5), 0.0);
  fMVAVar_NeutralHadronIso_DR0p1To0p2 = TMath::Max(TMath::Min((NeutralHadronIso_DR0p1To0p2 - Rho*ElectronEffectiveArea::GetElectronEffectiveArea(ElectronEffectiveArea::kEleNeutralHadronIsoDR0p1To0p2, Eta, EATarget))/Pt, 2.5), 0.0);
  fMVAVar_NeutralHadronIso_DR0p2To0p3 = TMath::Max(TMath::Min((NeutralHadronIso_DR0p2To0p3 - Rho*ElectronEffectiveArea::GetElectronEffectiveArea(ElectronEffectiveArea::kEleNeutralHadronIsoDR0p2To0p3, Eta, EATarget))/Pt, 2.5), 0.0);
  fMVAVar_NeutralHadronIso_DR0p3To0p4 = TMath::Max(TMath::Min((NeutralHadronIso_DR0p3To0p4 - Rho*ElectronEffectiveArea::GetElectronEffectiveArea(ElectronEffectiveArea::kEleNeutralHadronIsoDR0p3To0p4, Eta, EATarget))/Pt, 2.5), 0.0);
  fMVAVar_NeutralHadronIso_DR0p4To0p5 = TMath::Max(TMath::Min((NeutralHadronIso_DR0p4To0p5 - Rho*ElectronEffectiveArea::GetElectronEffectiveArea(ElectronEffectiveArea::kEleNeutralHadronIsoDR0p4To0p5, Eta, EATarget))/Pt, 2.5), 0.0);

  // evaluate
  Double_t mva = fTMVAReader[GetMVABin(Eta,Pt)]->EvaluateMVA(fMethodname);

  if(printDebug) {
    cout << " *** Inside the class fMethodname " << fMethodname << " fMVAType " << fMVAType << endl;
    cout  << "ChargedIso ( 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 ): " 
          << fMVAVar_ChargedIso_DR0p0To0p1   << " "
          << fMVAVar_ChargedIso_DR0p1To0p2   << " "
          << fMVAVar_ChargedIso_DR0p2To0p3 << " "
          << fMVAVar_ChargedIso_DR0p3To0p4 << " "
          << fMVAVar_ChargedIso_DR0p4To0p5 << endl;
    cout  << "PF Gamma Iso ( 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 ): " 
          << fMVAVar_GammaIso_DR0p0To0p1 << " "
          << fMVAVar_GammaIso_DR0p1To0p2 << " "
          << fMVAVar_GammaIso_DR0p2To0p3 << " "
          << fMVAVar_GammaIso_DR0p3To0p4 << " "
          << fMVAVar_GammaIso_DR0p4To0p5 << endl;
    cout  << "PF Neutral Hadron Iso ( 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 ): " 
          << fMVAVar_NeutralHadronIso_DR0p0To0p1 << " "
          << fMVAVar_NeutralHadronIso_DR0p1To0p2 << " "
          << fMVAVar_NeutralHadronIso_DR0p2To0p3 << " "
          << fMVAVar_NeutralHadronIso_DR0p3To0p4 << " "
          << fMVAVar_NeutralHadronIso_DR0p4To0p5 << " "
          << endl;
    cout << " ### MVA " << mva << endl;
  }

  return mva;

}


//--------------------------------------------------------------------------------------------------

Double_t EGammaMvaEleEstimator::mvaValue(const reco::GsfElectron& ele, 
					const reco::Vertex& vertex, 
					const TransientTrackBuilder& transientTrackBuilder,					
					EcalClusterLazyTools myEcalCluster,
					bool printDebug) {
  
  if (!fisInitialized) { 
    std::cout << "Error: EGammaMvaEleEstimator not properly initialized.\n"; 
    return -9999;
  }
  
  bool validKF= false; 
  reco::TrackRef myTrackRef = ele.closestCtfTrackRef();
  validKF = (myTrackRef.isAvailable());
  validKF = (myTrackRef.isNonnull());  

  // Pure tracking variables
  fMVAVar_fbrem           =  ele.fbrem();
  fMVAVar_kfchi2          =  (validKF) ? myTrackRef->normalizedChi2() : 0 ;
  fMVAVar_kfhits          =  (validKF) ? myTrackRef->hitPattern().trackerLayersWithMeasurement() : -1. ; 
  fMVAVar_kfhitsall          =  (validKF) ? myTrackRef->numberOfValidHits() : -1. ;   //  save also this in your ntuple as possible alternative
  fMVAVar_gsfchi2         =  ele.gsfTrack()->normalizedChi2();  

  
  // Geometrical matchings
  fMVAVar_deta            =  ele.deltaEtaSuperClusterTrackAtVtx();
  fMVAVar_dphi            =  ele.deltaPhiSuperClusterTrackAtVtx();
  fMVAVar_detacalo        =  ele.deltaEtaSeedClusterTrackAtCalo();


  // Pure ECAL -> shower shapes
  fMVAVar_see             =  ele.sigmaIetaIeta();    //EleSigmaIEtaIEta
  std::vector<float> vCov = myEcalCluster.localCovariances(*(ele.superCluster()->seed())) ;
  if (!isnan(vCov[2])) fMVAVar_spp = sqrt (vCov[2]);   //EleSigmaIPhiIPhi
  else fMVAVar_spp = 0.;    

  fMVAVar_etawidth        =  ele.superCluster()->etaWidth();
  fMVAVar_phiwidth        =  ele.superCluster()->phiWidth();
  fMVAVar_OneMinusE1x5E5x5        =  (ele.e5x5()) !=0. ? 1.-(ele.e1x5()/ele.e5x5()) : -1. ;
  fMVAVar_R9              =  myEcalCluster.e3x3(*(ele.superCluster()->seed())) / ele.superCluster()->rawEnergy();

  // Energy matching
  fMVAVar_HoE             =  ele.hadronicOverEm();
  fMVAVar_EoP             =  ele.eSuperClusterOverP();
  fMVAVar_IoEmIoP         =  (1.0/ele.ecalEnergy()) - (1.0 / ele.p());  // in the future to be changed with ele.gsfTrack()->p()
  fMVAVar_eleEoPout       =  ele.eEleClusterOverPout();
  fMVAVar_PreShowerOverRaw=  ele.superCluster()->preshowerEnergy() / ele.superCluster()->rawEnergy();


  // Spectators
  fMVAVar_eta             =  ele.superCluster()->eta();         
  fMVAVar_pt              =  ele.pt();                          

 

  // for triggering electrons get the impact parameteres
  if(fMVAType == kTrig) {
    //d0
    if (ele.gsfTrack().isNonnull()) {
      fMVAVar_d0 = (-1.0)*ele.gsfTrack()->dxy(vertex.position()); 
    } else if (ele.closestCtfTrackRef().isNonnull()) {
      fMVAVar_d0 = (-1.0)*ele.closestCtfTrackRef()->dxy(vertex.position()); 
    } else {
      fMVAVar_d0 = -9999.0;
    }
    
    //default values for IP3D
    fMVAVar_ip3d = -999.0; 
    fMVAVar_ip3dSig = 0.0;
    if (ele.gsfTrack().isNonnull()) {
      const double gsfsign   = ( (-ele.gsfTrack()->dxy(vertex.position()))   >=0 ) ? 1. : -1.;
      
      const reco::TransientTrack &tt = transientTrackBuilder.build(ele.gsfTrack()); 
      const std::pair<bool,Measurement1D> &ip3dpv =  IPTools::absoluteImpactParameter3D(tt,vertex);
      if (ip3dpv.first) {
	double ip3d = gsfsign*ip3dpv.second.value();
	double ip3derr = ip3dpv.second.error();  
	fMVAVar_ip3d = ip3d; 
        fMVAVar_ip3dSig = ip3d/ip3derr;
      }
    }
  }
  

  // evaluate
  bindVariables();
  Double_t mva = -9999;  
  if (fUseBinnedVersion) {
    mva = fTMVAReader[GetMVABin(fMVAVar_eta,fMVAVar_pt)]->EvaluateMVA(fMethodname);
  } else {
    mva = fTMVAReader[0]->EvaluateMVA(fMethodname);
  }



  if(printDebug) {
    cout << " *** Inside the class fMethodname " << fMethodname << " fMVAType " << fMVAType << endl;
    cout << " fbrem " <<  fMVAVar_fbrem  
      	 << " kfchi2 " << fMVAVar_kfchi2  
	 << " mykfhits " << fMVAVar_kfhits  
	 << " gsfchi2 " << fMVAVar_gsfchi2  
	 << " deta " <<  fMVAVar_deta  
	 << " dphi " << fMVAVar_dphi  
      	 << " detacalo " << fMVAVar_detacalo  
	 << " see " << fMVAVar_see  
	 << " spp " << fMVAVar_spp  
	 << " etawidth " << fMVAVar_etawidth  
	 << " phiwidth " << fMVAVar_phiwidth  
	 << " OneMinusE1x5E5x5 " << fMVAVar_OneMinusE1x5E5x5  
	 << " R9 " << fMVAVar_R9  
	 << " HoE " << fMVAVar_HoE  
	 << " EoP " << fMVAVar_EoP  
	 << " IoEmIoP " << fMVAVar_IoEmIoP  
	 << " eleEoPout " << fMVAVar_eleEoPout  
	 << " d0 " << fMVAVar_d0  
	 << " ip3d " << fMVAVar_ip3d  
	 << " eta " << fMVAVar_eta  
	 << " pt " << fMVAVar_pt << endl;
    cout << " ### MVA " << mva << endl;
  }



  return mva;
}


Double_t EGammaMvaEleEstimator::isoMvaValue(const reco::GsfElectron& ele, 
                                            const reco::Vertex& vertex, 
                                            const reco::PFCandidateCollection &PFCandidates,
                                            double Rho,
                                            ElectronEffectiveArea::ElectronEffectiveAreaTarget EATarget,
                                            const reco::GsfElectronCollection &IdentifiedElectrons,
                                            const reco::MuonCollection &IdentifiedMuons,
                                            bool printDebug) {
  
  if (!fisInitialized) { 
    std::cout << "Error: EGammaMvaEleEstimator not properly initialized.\n"; 
    return -9999;
  }
  
  // Spectators 	 
  fMVAVar_eta             =  ele.superCluster()->eta(); 	 
  fMVAVar_pt              =  ele.pt();
  
  //**********************************************************
  //Isolation variables
  //**********************************************************
  Double_t tmpChargedIso_DR0p0To0p1  = 0;
  Double_t tmpChargedIso_DR0p1To0p2  = 0;
  Double_t tmpChargedIso_DR0p2To0p3  = 0;
  Double_t tmpChargedIso_DR0p3To0p4  = 0;
  Double_t tmpChargedIso_DR0p4To0p5  = 0;
  Double_t tmpGammaIso_DR0p0To0p1  = 0;
  Double_t tmpGammaIso_DR0p1To0p2  = 0;
  Double_t tmpGammaIso_DR0p2To0p3  = 0;
  Double_t tmpGammaIso_DR0p3To0p4  = 0;
  Double_t tmpGammaIso_DR0p4To0p5  = 0;
  Double_t tmpNeutralHadronIso_DR0p0To0p1  = 0;
  Double_t tmpNeutralHadronIso_DR0p1To0p2  = 0;
  Double_t tmpNeutralHadronIso_DR0p2To0p3  = 0;
  Double_t tmpNeutralHadronIso_DR0p3To0p4  = 0;
  Double_t tmpNeutralHadronIso_DR0p4To0p5  = 0;

  double electronTrackZ = 0;
  if (ele.gsfTrack().isNonnull()) {
    electronTrackZ = ele.gsfTrack()->dz(vertex.position());
  } else if (ele.closestCtfTrackRef().isNonnull()) {
    electronTrackZ = ele.closestCtfTrackRef()->dz(vertex.position());
  }

  for (reco::PFCandidateCollection::const_iterator iP = PFCandidates.begin(); 
       iP != PFCandidates.end(); ++iP) {
      
    //exclude the electron itself
    if(iP->gsfTrackRef().isNonnull() && ele.gsfTrack().isNonnull() &&
       refToPtr(iP->gsfTrackRef()) == refToPtr(ele.gsfTrack())) continue;
    if(iP->trackRef().isNonnull() && ele.closestCtfTrackRef().isNonnull() &&
       refToPtr(iP->trackRef()) == refToPtr(ele.closestCtfTrackRef())) continue;      

    //************************************************************
    // New Isolation Calculations
    //************************************************************
    double dr = sqrt(pow(iP->eta() - ele.eta(),2) + pow(acos(cos(iP->phi() - ele.phi())),2));
    //Double_t deta = (iP->eta() - ele.eta());

    if (dr < 1.0) {
      Bool_t IsLeptonFootprint = kFALSE;
      //************************************************************
      // Lepton Footprint Removal
      //************************************************************   
      for (reco::GsfElectronCollection::const_iterator iE = IdentifiedElectrons.begin(); 
           iE != IdentifiedElectrons.end(); ++iE) {
	//if pf candidate matches an electron passing ID cuts, then veto it
	if(iP->gsfTrackRef().isNonnull() && iE->gsfTrack().isNonnull() &&
	   refToPtr(iP->gsfTrackRef()) == refToPtr(iE->gsfTrack())) IsLeptonFootprint = kTRUE;
        if(iP->trackRef().isNonnull() && iE->closestCtfTrackRef().isNonnull() &&
           refToPtr(iP->trackRef()) == refToPtr(iE->closestCtfTrackRef())) IsLeptonFootprint = kTRUE;

	//if pf candidate lies in veto regions of electron passing ID cuts, then veto it
        double tmpDR = sqrt(pow(iP->eta() - iE->eta(),2) + pow(acos(cos(iP->phi() - iE->phi())),2));
	if(iP->trackRef().isNonnull() && fabs(iE->superCluster()->eta()) >= 1.479 
           && tmpDR < 0.015) IsLeptonFootprint = kTRUE;
	if(iP->particleId() == reco::PFCandidate::gamma && fabs(iE->superCluster()->eta()) >= 1.479 
           && tmpDR < 0.08) IsLeptonFootprint = kTRUE;
      }
      for (reco::MuonCollection::const_iterator iM = IdentifiedMuons.begin(); 
           iM != IdentifiedMuons.end(); ++iM) {
	//if pf candidate matches an muon passing ID cuts, then veto it
	if(iP->trackRef().isNonnull() && iM->innerTrack().isNonnull() &&
	   refToPtr(iP->trackRef()) == refToPtr(iM->innerTrack())) IsLeptonFootprint = kTRUE;

	//if pf candidate lies in veto regions of muon passing ID cuts, then veto it
        double tmpDR = sqrt(pow(iP->eta() - iM->eta(),2) + pow(acos(cos(iP->phi() - iM->phi())),2));
	if(iP->trackRef().isNonnull() && tmpDR < 0.01) IsLeptonFootprint = kTRUE;
      }

     if (!IsLeptonFootprint) {
	Bool_t passVeto = kTRUE;
	//Charged
	 if(iP->trackRef().isNonnull()) {	  	   
	   if (!(fabs(iP->trackRef()->dz(vertex.position()) - electronTrackZ) < 0.2)) passVeto = kFALSE;
	   //************************************************************
	   // Veto any PFmuon, or PFEle
	   if (iP->particleId() == reco::PFCandidate::e || iP->particleId() == reco::PFCandidate::mu) passVeto = kFALSE;
	   //************************************************************
	   //************************************************************
	   // Footprint Veto
	   if (fabs(fMVAVar_eta) > 1.479 && dr < 0.015) passVeto = kFALSE;
	   //************************************************************
	   if (passVeto) {
	     if (dr < 0.1) tmpChargedIso_DR0p0To0p1 += iP->pt();
	     if (dr >= 0.1 && dr < 0.2) tmpChargedIso_DR0p1To0p2 += iP->pt();
	     if (dr >= 0.2 && dr < 0.3) tmpChargedIso_DR0p2To0p3 += iP->pt();
	     if (dr >= 0.3 && dr < 0.4) tmpChargedIso_DR0p3To0p4 += iP->pt();
	     if (dr >= 0.4 && dr < 0.5) tmpChargedIso_DR0p4To0p5 += iP->pt();
	   } //pass veto	   
	 }
	 //Gamma
	 else if (iP->particleId() == reco::PFCandidate::gamma) {
	   //************************************************************
	   // Footprint Veto
	   if (fabs(fMVAVar_eta) > 1.479 && dr < 0.08) passVeto = kFALSE;
	   //************************************************************	
	   if (passVeto) {
	     if (dr < 0.1) tmpGammaIso_DR0p0To0p1 += iP->pt();
	     if (dr >= 0.1 && dr < 0.2) tmpGammaIso_DR0p1To0p2 += iP->pt();
	     if (dr >= 0.2 && dr < 0.3) tmpGammaIso_DR0p2To0p3 += iP->pt();
	     if (dr >= 0.3 && dr < 0.4) tmpGammaIso_DR0p3To0p4 += iP->pt();
	     if (dr >= 0.4 && dr < 0.5) tmpGammaIso_DR0p4To0p5 += iP->pt();
	   }
	 }
	 //NeutralHadron
	 else {
           if (dr < 0.1) tmpNeutralHadronIso_DR0p0To0p1 += iP->pt();
           if (dr >= 0.1 && dr < 0.2) tmpNeutralHadronIso_DR0p1To0p2 += iP->pt();
           if (dr >= 0.2 && dr < 0.3) tmpNeutralHadronIso_DR0p2To0p3 += iP->pt();
           if (dr >= 0.3 && dr < 0.4) tmpNeutralHadronIso_DR0p3To0p4 += iP->pt();
           if (dr >= 0.4 && dr < 0.5) tmpNeutralHadronIso_DR0p4To0p5 += iP->pt();
	 }
      } //not lepton footprint
    } //in 1.0 dr cone
  } //loop over PF candidates

  fMVAVar_ChargedIso_DR0p0To0p1   = TMath::Min((tmpChargedIso_DR0p0To0p1)/ele.pt(), 2.5);
  fMVAVar_ChargedIso_DR0p1To0p2   = TMath::Min((tmpChargedIso_DR0p1To0p2)/ele.pt(), 2.5);
  fMVAVar_ChargedIso_DR0p2To0p3 = TMath::Min((tmpChargedIso_DR0p2To0p3)/ele.pt(), 2.5);
  fMVAVar_ChargedIso_DR0p3To0p4 = TMath::Min((tmpChargedIso_DR0p3To0p4)/ele.pt(), 2.5);
  fMVAVar_ChargedIso_DR0p4To0p5 = TMath::Min((tmpChargedIso_DR0p4To0p5)/ele.pt(), 2.5); 
  fMVAVar_GammaIso_DR0p0To0p1 = TMath::Max(TMath::Min((tmpGammaIso_DR0p0To0p1 - Rho*ElectronEffectiveArea::GetElectronEffectiveArea(ElectronEffectiveArea::kEleGammaIsoDR0p0To0p1, fMVAVar_eta, EATarget))/ele.pt(), 2.5), 0.0);
  fMVAVar_GammaIso_DR0p1To0p2 = TMath::Max(TMath::Min((tmpGammaIso_DR0p1To0p2 - Rho*ElectronEffectiveArea::GetElectronEffectiveArea(ElectronEffectiveArea::kEleGammaIsoDR0p1To0p2, fMVAVar_eta, EATarget))/ele.pt(), 2.5), 0.0);
  fMVAVar_GammaIso_DR0p2To0p3 = TMath::Max(TMath::Min((tmpGammaIso_DR0p2To0p3 - Rho*ElectronEffectiveArea::GetElectronEffectiveArea(ElectronEffectiveArea::kEleGammaIsoDR0p2To0p3, fMVAVar_eta, EATarget))/ele.pt(), 2.5), 0.0);
  fMVAVar_GammaIso_DR0p3To0p4 = TMath::Max(TMath::Min((tmpGammaIso_DR0p3To0p4 - Rho*ElectronEffectiveArea::GetElectronEffectiveArea(ElectronEffectiveArea::kEleGammaIsoDR0p3To0p4, fMVAVar_eta, EATarget))/ele.pt(), 2.5), 0.0);
  fMVAVar_GammaIso_DR0p4To0p5 = TMath::Max(TMath::Min((tmpGammaIso_DR0p4To0p5 - Rho*ElectronEffectiveArea::GetElectronEffectiveArea(ElectronEffectiveArea::kEleGammaIsoDR0p4To0p5, fMVAVar_eta, EATarget))/ele.pt(), 2.5), 0.0);
  fMVAVar_NeutralHadronIso_DR0p0To0p1 = TMath::Max(TMath::Min((tmpNeutralHadronIso_DR0p0To0p1 - Rho*ElectronEffectiveArea::GetElectronEffectiveArea(ElectronEffectiveArea::kEleNeutralHadronIsoDR0p0To0p1, fMVAVar_eta, EATarget))/ele.pt(), 2.5), 0.0);
  fMVAVar_NeutralHadronIso_DR0p1To0p2 = TMath::Max(TMath::Min((tmpNeutralHadronIso_DR0p1To0p2 - Rho*ElectronEffectiveArea::GetElectronEffectiveArea(ElectronEffectiveArea::kEleNeutralHadronIsoDR0p1To0p2, fMVAVar_eta, EATarget))/ele.pt(), 2.5), 0.0);
  fMVAVar_NeutralHadronIso_DR0p2To0p3 = TMath::Max(TMath::Min((tmpNeutralHadronIso_DR0p2To0p3 - Rho*ElectronEffectiveArea::GetElectronEffectiveArea(ElectronEffectiveArea::kEleNeutralHadronIsoDR0p2To0p3, fMVAVar_eta, EATarget))/ele.pt(), 2.5), 0.0);
  fMVAVar_NeutralHadronIso_DR0p3To0p4 = TMath::Max(TMath::Min((tmpNeutralHadronIso_DR0p3To0p4 - Rho*ElectronEffectiveArea::GetElectronEffectiveArea(ElectronEffectiveArea::kEleNeutralHadronIsoDR0p3To0p4, fMVAVar_eta, EATarget))/ele.pt(), 2.5), 0.0);
  fMVAVar_NeutralHadronIso_DR0p4To0p5 = TMath::Max(TMath::Min((tmpNeutralHadronIso_DR0p4To0p5 - Rho*ElectronEffectiveArea::GetElectronEffectiveArea(ElectronEffectiveArea::kEleNeutralHadronIsoDR0p4To0p5, fMVAVar_eta, EATarget))/ele.pt(), 2.5), 0.0);
 
  if (printDebug) {
    cout << "UseBinnedVersion=" << fUseBinnedVersion << " -> BIN: " << fMVAVar_eta << " " << fMVAVar_pt << " : " << GetMVABin(fMVAVar_eta,fMVAVar_pt) << endl;
  }

  // evaluate
  bindVariables();
  Double_t mva = -9999; 
   
//   mva = fTMVAReader[0]->EvaluateMVA(fMethodname);
  if (fUseBinnedVersion) {
    mva = fTMVAReader[GetMVABin(fMVAVar_eta,fMVAVar_pt)]->EvaluateMVA(fMethodname);
  } else {
    mva = fTMVAReader[0]->EvaluateMVA(fMethodname);
  }


  if(printDebug) {
    cout << " *** Inside the class fMethodname " << fMethodname << " fMVAType " << fMVAType << endl;
    cout  << "ChargedIso ( 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 ): " 
          << fMVAVar_ChargedIso_DR0p0To0p1   << " "
          << fMVAVar_ChargedIso_DR0p1To0p2   << " "
          << fMVAVar_ChargedIso_DR0p2To0p3 << " "
          << fMVAVar_ChargedIso_DR0p3To0p4 << " "
          << fMVAVar_ChargedIso_DR0p4To0p5 << endl;
    cout  << "PF Gamma Iso ( 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 ): " 
          << fMVAVar_GammaIso_DR0p0To0p1 << " "
          << fMVAVar_GammaIso_DR0p1To0p2 << " "
          << fMVAVar_GammaIso_DR0p2To0p3 << " "
          << fMVAVar_GammaIso_DR0p3To0p4 << " "
          << fMVAVar_GammaIso_DR0p4To0p5 << endl;
    cout  << "PF Neutral Hadron Iso ( 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 ): " 
          << fMVAVar_NeutralHadronIso_DR0p0To0p1 << " "
          << fMVAVar_NeutralHadronIso_DR0p1To0p2 << " "
          << fMVAVar_NeutralHadronIso_DR0p2To0p3 << " "
          << fMVAVar_NeutralHadronIso_DR0p3To0p4 << " "
          << fMVAVar_NeutralHadronIso_DR0p4To0p5 << " "
          << endl;
    cout << " ### MVA " << mva << endl;
  }
  

  return mva;
}


//--------------------------------------------------------------------------------------------------

Double_t EGammaMvaEleEstimator::IDIsoCombinedMvaValue(const reco::GsfElectron& ele, 
                                                      const reco::Vertex& vertex, 
                                                      const TransientTrackBuilder& transientTrackBuilder,					
                                                      EcalClusterLazyTools myEcalCluster,
                                                      const reco::PFCandidateCollection &PFCandidates,
                                                      double Rho,
                                                      ElectronEffectiveArea::ElectronEffectiveAreaTarget EATarget,
                                                      bool printDebug) {
  
  if (!fisInitialized) { 
    std::cout << "Error: EGammaMvaEleEstimator not properly initialized.\n"; 
    return -9999;
  }
  
  bool validKF= false; 
  reco::TrackRef myTrackRef = ele.closestCtfTrackRef();
  validKF = (myTrackRef.isAvailable());
  validKF = (myTrackRef.isNonnull());  

  // Pure tracking variables
  fMVAVar_fbrem           =  (ele.fbrem() < -1. ) ? -1. : ele.fbrem();
  fMVAVar_kfchi2           =  (validKF) ? myTrackRef->normalizedChi2() : 0 ; 
  if (fMVAVar_kfchi2 > 10) fMVAVar_kfchi2 = 10;
  fMVAVar_kfhits          =  (validKF) ? myTrackRef->hitPattern().trackerLayersWithMeasurement() : -1. ; 
  fMVAVar_kfhitsall          =  (validKF) ? myTrackRef->numberOfValidHits() : -1. ;   //  save also this in your ntuple as possible alternative
  fMVAVar_gsfchi2         =  ele.gsfTrack()->normalizedChi2();  
  if (fMVAVar_gsfchi2 > 200) fMVAVar_gsfchi2 = 200;

  
  // Geometrical matchings
  fMVAVar_deta            =  ( fabs(ele.deltaEtaSuperClusterTrackAtVtx()) > 0.06 ) ? 0.06 : fabs(ele.deltaEtaSuperClusterTrackAtVtx());
  fMVAVar_dphi            =  ele.deltaPhiSuperClusterTrackAtVtx();
  fMVAVar_detacalo        =  ele.deltaEtaSeedClusterTrackAtCalo();


  // Pure ECAL -> shower shapes
  fMVAVar_see             =  ele.sigmaIetaIeta();    //EleSigmaIEtaIEta
  std::vector<float> vCov = myEcalCluster.localCovariances(*(ele.superCluster()->seed())) ;
  if (!isnan(vCov[2])) fMVAVar_spp = sqrt (vCov[2]);   //EleSigmaIPhiIPhi
  else fMVAVar_spp = 0.;    

  fMVAVar_etawidth        =  ele.superCluster()->etaWidth();
  fMVAVar_phiwidth        =  ele.superCluster()->phiWidth();
  fMVAVar_OneMinusE1x5E5x5        =  (ele.e5x5()) !=0. ? 1.-(ele.e1x5()/ele.e5x5()) : -1. ;
  fMVAVar_OneMinusE1x5E5x5 = max(min(double(fMVAVar_OneMinusE1x5E5x5),2.0),-1.0);
  fMVAVar_R9              =  myEcalCluster.e3x3(*(ele.superCluster()->seed())) / ele.superCluster()->rawEnergy();
  if (fMVAVar_R9 > 5) fMVAVar_R9 = 5;

  // Energy matching
  fMVAVar_HoE             =  ele.hadronicOverEm();
  fMVAVar_EoP             =  ( ele.eSuperClusterOverP() > 20 ) ? 20 : ele.eSuperClusterOverP();
  fMVAVar_IoEmIoP         =  (1.0/ele.superCluster()->energy()) - (1.0 / ele.trackMomentumAtVtx().R()); //this is the proper variable
  fMVAVar_eleEoPout       =  ( ele.eEleClusterOverPout() > 20 ) ? 20 : ele.eEleClusterOverPout();
  fMVAVar_PreShowerOverRaw=  ele.superCluster()->preshowerEnergy() / ele.superCluster()->rawEnergy();

  // Spectators
  fMVAVar_eta             =  ele.superCluster()->eta();         
  fMVAVar_pt              =  ele.pt();                          

 

  // for triggering electrons get the impact parameteres
  if(fMVAType == kTrig) {
    //d0
    if (ele.gsfTrack().isNonnull()) {
      fMVAVar_d0 = (-1.0)*ele.gsfTrack()->dxy(vertex.position()); 
    } else if (ele.closestCtfTrackRef().isNonnull()) {
      fMVAVar_d0 = (-1.0)*ele.closestCtfTrackRef()->dxy(vertex.position()); 
    } else {
      fMVAVar_d0 = -9999.0;
    }
    
    //default values for IP3D
    fMVAVar_ip3d = -999.0; 
    fMVAVar_ip3dSig = 0.0;
    if (ele.gsfTrack().isNonnull()) {
      const double gsfsign   = ( (-ele.gsfTrack()->dxy(vertex.position()))   >=0 ) ? 1. : -1.;
      
      const reco::TransientTrack &tt = transientTrackBuilder.build(ele.gsfTrack()); 
      const std::pair<bool,Measurement1D> &ip3dpv =  IPTools::absoluteImpactParameter3D(tt,vertex);
      if (ip3dpv.first) {
	double ip3d = gsfsign*ip3dpv.second.value();
	double ip3derr = ip3dpv.second.error();  
	fMVAVar_ip3d = ip3d; 
        fMVAVar_ip3dSig = ip3d/ip3derr;
      }
    }
  }
  
  //**********************************************************
  //Isolation variables
  //**********************************************************
  Double_t tmpChargedIso_DR0p0To0p1  = 0;
  Double_t tmpChargedIso_DR0p1To0p2  = 0;
  Double_t tmpChargedIso_DR0p2To0p3  = 0;
  Double_t tmpChargedIso_DR0p3To0p4  = 0;
  Double_t tmpChargedIso_DR0p4To0p5  = 0;
  Double_t tmpGammaIso_DR0p0To0p1  = 0;
  Double_t tmpGammaIso_DR0p1To0p2  = 0;
  Double_t tmpGammaIso_DR0p2To0p3  = 0;
  Double_t tmpGammaIso_DR0p3To0p4  = 0;
  Double_t tmpGammaIso_DR0p4To0p5  = 0;
  Double_t tmpNeutralHadronIso_DR0p0To0p1  = 0;
  Double_t tmpNeutralHadronIso_DR0p1To0p2  = 0;
  Double_t tmpNeutralHadronIso_DR0p2To0p3  = 0;
  Double_t tmpNeutralHadronIso_DR0p3To0p4  = 0;
  Double_t tmpNeutralHadronIso_DR0p4To0p5  = 0;

  for (reco::PFCandidateCollection::const_iterator iP = PFCandidates.begin(); 
       iP != PFCandidates.end(); ++iP) {
      
    double dr = sqrt(pow(iP->eta() - ele.eta(),2) + pow(acos(cos(iP->phi() - ele.phi())),2));

    Bool_t passVeto = kTRUE;
    //Charged
    if(iP->trackRef().isNonnull()) {	  	   

      //make sure charged pf candidates pass the PFNoPU condition (assumed)

      //************************************************************
      // Veto any PFmuon, or PFEle
      if (iP->particleId() == reco::PFCandidate::e || iP->particleId() == reco::PFCandidate::mu) passVeto = kFALSE;
      //************************************************************
      //************************************************************
      // Footprint Veto
      if (fabs(fMVAVar_eta) > 1.479 && dr < 0.015) passVeto = kFALSE;
      //************************************************************
      if (passVeto) {
        if (dr < 0.1) tmpChargedIso_DR0p0To0p1 += iP->pt();
        if (dr >= 0.1 && dr < 0.2) tmpChargedIso_DR0p1To0p2 += iP->pt();
        if (dr >= 0.2 && dr < 0.3) tmpChargedIso_DR0p2To0p3 += iP->pt();
        if (dr >= 0.3 && dr < 0.4) tmpChargedIso_DR0p3To0p4 += iP->pt();
        if (dr >= 0.4 && dr < 0.5) tmpChargedIso_DR0p4To0p5 += iP->pt();
      } //pass veto	   
    }
    //Gamma
    else if (iP->particleId() == reco::PFCandidate::gamma) {
      //************************************************************
      // Footprint Veto
      if (fabs(fMVAVar_eta) > 1.479 && dr < 0.08) passVeto = kFALSE;
      //************************************************************	
      if (passVeto) {
        if (dr < 0.1) tmpGammaIso_DR0p0To0p1 += iP->pt();
        if (dr >= 0.1 && dr < 0.2) tmpGammaIso_DR0p1To0p2 += iP->pt();
        if (dr >= 0.2 && dr < 0.3) tmpGammaIso_DR0p2To0p3 += iP->pt();
        if (dr >= 0.3 && dr < 0.4) tmpGammaIso_DR0p3To0p4 += iP->pt();
        if (dr >= 0.4 && dr < 0.5) tmpGammaIso_DR0p4To0p5 += iP->pt();
      }
    }
    //NeutralHadron
    else {
      if (dr < 0.1) tmpNeutralHadronIso_DR0p0To0p1 += iP->pt();
      if (dr >= 0.1 && dr < 0.2) tmpNeutralHadronIso_DR0p1To0p2 += iP->pt();
      if (dr >= 0.2 && dr < 0.3) tmpNeutralHadronIso_DR0p2To0p3 += iP->pt();
      if (dr >= 0.3 && dr < 0.4) tmpNeutralHadronIso_DR0p3To0p4 += iP->pt();
      if (dr >= 0.4 && dr < 0.5) tmpNeutralHadronIso_DR0p4To0p5 += iP->pt();
    }
  } //loop over PF candidates

  if (fMVAType == kTrigIDIsoCombinedPUCorrected) {
    fMVAVar_ChargedIso_DR0p0To0p1   = TMath::Min((tmpChargedIso_DR0p0To0p1)/ele.pt(), 2.5);
    fMVAVar_ChargedIso_DR0p1To0p2   = TMath::Min((tmpChargedIso_DR0p1To0p2)/ele.pt(), 2.5);
    fMVAVar_ChargedIso_DR0p2To0p3 = TMath::Min((tmpChargedIso_DR0p2To0p3)/ele.pt(), 2.5);
    fMVAVar_ChargedIso_DR0p3To0p4 = TMath::Min((tmpChargedIso_DR0p3To0p4)/ele.pt(), 2.5);
    fMVAVar_ChargedIso_DR0p4To0p5 = TMath::Min((tmpChargedIso_DR0p4To0p5)/ele.pt(), 2.5); 
    fMVAVar_GammaIso_DR0p0To0p1 = TMath::Max(TMath::Min((tmpGammaIso_DR0p0To0p1 - Rho*ElectronEffectiveArea::GetElectronEffectiveArea(ElectronEffectiveArea::kEleGammaIsoDR0p0To0p1, fMVAVar_eta, EATarget))/ele.pt(), 2.5), 0.0);
    fMVAVar_GammaIso_DR0p1To0p2 = TMath::Max(TMath::Min((tmpGammaIso_DR0p1To0p2 - Rho*ElectronEffectiveArea::GetElectronEffectiveArea(ElectronEffectiveArea::kEleGammaIsoDR0p1To0p2, fMVAVar_eta, EATarget))/ele.pt(), 2.5), 0.0);
    fMVAVar_GammaIso_DR0p2To0p3 = TMath::Max(TMath::Min((tmpGammaIso_DR0p2To0p3 - Rho*ElectronEffectiveArea::GetElectronEffectiveArea(ElectronEffectiveArea::kEleGammaIsoDR0p2To0p3, fMVAVar_eta, EATarget))/ele.pt(), 2.5), 0.0);
    fMVAVar_GammaIso_DR0p3To0p4 = TMath::Max(TMath::Min((tmpGammaIso_DR0p3To0p4 - Rho*ElectronEffectiveArea::GetElectronEffectiveArea(ElectronEffectiveArea::kEleGammaIsoDR0p3To0p4, fMVAVar_eta, EATarget))/ele.pt(), 2.5), 0.0);
    fMVAVar_GammaIso_DR0p4To0p5 = TMath::Max(TMath::Min((tmpGammaIso_DR0p4To0p5 - Rho*ElectronEffectiveArea::GetElectronEffectiveArea(ElectronEffectiveArea::kEleGammaIsoDR0p4To0p5, fMVAVar_eta, EATarget))/ele.pt(), 2.5), 0.0);
    fMVAVar_NeutralHadronIso_DR0p0To0p1 = TMath::Max(TMath::Min((tmpNeutralHadronIso_DR0p0To0p1 - Rho*ElectronEffectiveArea::GetElectronEffectiveArea(ElectronEffectiveArea::kEleNeutralHadronIsoDR0p0To0p1, fMVAVar_eta, EATarget))/ele.pt(), 2.5), 0.0);
    fMVAVar_NeutralHadronIso_DR0p1To0p2 = TMath::Max(TMath::Min((tmpNeutralHadronIso_DR0p1To0p2 - Rho*ElectronEffectiveArea::GetElectronEffectiveArea(ElectronEffectiveArea::kEleNeutralHadronIsoDR0p1To0p2, fMVAVar_eta, EATarget))/ele.pt(), 2.5), 0.0);
    fMVAVar_NeutralHadronIso_DR0p2To0p3 = TMath::Max(TMath::Min((tmpNeutralHadronIso_DR0p2To0p3 - Rho*ElectronEffectiveArea::GetElectronEffectiveArea(ElectronEffectiveArea::kEleNeutralHadronIsoDR0p2To0p3, fMVAVar_eta, EATarget))/ele.pt(), 2.5), 0.0);
    fMVAVar_NeutralHadronIso_DR0p3To0p4 = TMath::Max(TMath::Min((tmpNeutralHadronIso_DR0p3To0p4 - Rho*ElectronEffectiveArea::GetElectronEffectiveArea(ElectronEffectiveArea::kEleNeutralHadronIsoDR0p3To0p4, fMVAVar_eta, EATarget))/ele.pt(), 2.5), 0.0);
    fMVAVar_NeutralHadronIso_DR0p4To0p5 = TMath::Max(TMath::Min((tmpNeutralHadronIso_DR0p4To0p5 - Rho*ElectronEffectiveArea::GetElectronEffectiveArea(ElectronEffectiveArea::kEleNeutralHadronIsoDR0p4To0p5, fMVAVar_eta, EATarget))/ele.pt(), 2.5), 0.0);
  } else if (fMVAType == kTrigIDIsoCombined) {
    fMVAVar_ChargedIso_DR0p0To0p1   = TMath::Min((tmpChargedIso_DR0p0To0p1)/ele.pt(), 2.5);
    fMVAVar_ChargedIso_DR0p1To0p2   = TMath::Min((tmpChargedIso_DR0p1To0p2)/ele.pt(), 2.5);
    fMVAVar_ChargedIso_DR0p2To0p3 = TMath::Min((tmpChargedIso_DR0p2To0p3)/ele.pt(), 2.5);
    fMVAVar_ChargedIso_DR0p3To0p4 = TMath::Min((tmpChargedIso_DR0p3To0p4)/ele.pt(), 2.5);
    fMVAVar_ChargedIso_DR0p4To0p5 = TMath::Min((tmpChargedIso_DR0p4To0p5)/ele.pt(), 2.5); 
    fMVAVar_GammaIso_DR0p0To0p1 = TMath::Max(TMath::Min((tmpGammaIso_DR0p0To0p1)/ele.pt(), 2.5), 0.0);
    fMVAVar_GammaIso_DR0p1To0p2 = TMath::Max(TMath::Min((tmpGammaIso_DR0p1To0p2)/ele.pt(), 2.5), 0.0);
    fMVAVar_GammaIso_DR0p2To0p3 = TMath::Max(TMath::Min((tmpGammaIso_DR0p2To0p3)/ele.pt(), 2.5), 0.0);
    fMVAVar_GammaIso_DR0p3To0p4 = TMath::Max(TMath::Min((tmpGammaIso_DR0p3To0p4)/ele.pt(), 2.5), 0.0);
    fMVAVar_GammaIso_DR0p4To0p5 = TMath::Max(TMath::Min((tmpGammaIso_DR0p4To0p5)/ele.pt(), 2.5), 0.0);
    fMVAVar_NeutralHadronIso_DR0p0To0p1 = TMath::Max(TMath::Min((tmpNeutralHadronIso_DR0p0To0p1)/ele.pt(), 2.5), 0.0);
    fMVAVar_NeutralHadronIso_DR0p1To0p2 = TMath::Max(TMath::Min((tmpNeutralHadronIso_DR0p1To0p2)/ele.pt(), 2.5), 0.0);
    fMVAVar_NeutralHadronIso_DR0p2To0p3 = TMath::Max(TMath::Min((tmpNeutralHadronIso_DR0p2To0p3)/ele.pt(), 2.5), 0.0);
    fMVAVar_NeutralHadronIso_DR0p3To0p4 = TMath::Max(TMath::Min((tmpNeutralHadronIso_DR0p3To0p4)/ele.pt(), 2.5), 0.0);
    fMVAVar_NeutralHadronIso_DR0p4To0p5 = TMath::Max(TMath::Min((tmpNeutralHadronIso_DR0p4To0p5)/ele.pt(), 2.5), 0.0);
    fMVAVar_rho = Rho;
  } else {
    cout << "Warning: Type " << fMVAType << " is not supported.\n";
  }

  // evaluate
  Double_t mva = -9999;  
  if (fUseBinnedVersion) {
    mva = fTMVAReader[GetMVABin(fMVAVar_eta,fMVAVar_pt)]->EvaluateMVA(fMethodname);
  } else {
    mva = fTMVAReader[0]->EvaluateMVA(fMethodname);
  }



  if(printDebug) {
    cout << " *** Inside the class fMethodname " << fMethodname << " fMVAType " << fMVAType << endl;
    cout << " fbrem " <<  fMVAVar_fbrem  
      	 << " kfchi2 " << fMVAVar_kfchi2  
	 << " mykfhits " << fMVAVar_kfhits  
	 << " gsfchi2 " << fMVAVar_gsfchi2  
	 << " deta " <<  fMVAVar_deta  
	 << " dphi " << fMVAVar_dphi  
      	 << " detacalo " << fMVAVar_detacalo  
	 << " see " << fMVAVar_see  
	 << " spp " << fMVAVar_spp  
	 << " etawidth " << fMVAVar_etawidth  
	 << " phiwidth " << fMVAVar_phiwidth  
	 << " OneMinusE1x5E5x5 " << fMVAVar_OneMinusE1x5E5x5  
	 << " R9 " << fMVAVar_R9  
	 << " HoE " << fMVAVar_HoE  
	 << " EoP " << fMVAVar_EoP  
	 << " IoEmIoP " << fMVAVar_IoEmIoP  
	 << " eleEoPout " << fMVAVar_eleEoPout  
	 << " d0 " << fMVAVar_d0  
	 << " ip3d " << fMVAVar_ip3d  
	 << " eta " << fMVAVar_eta  
	 << " pt " << fMVAVar_pt << endl;
    cout  << "ChargedIso ( 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 ): " 
          << fMVAVar_ChargedIso_DR0p0To0p1   << " "
          << fMVAVar_ChargedIso_DR0p1To0p2   << " "
          << fMVAVar_ChargedIso_DR0p2To0p3 << " "
          << fMVAVar_ChargedIso_DR0p3To0p4 << " "
          << fMVAVar_ChargedIso_DR0p4To0p5 << endl;
    cout  << "PF Gamma Iso ( 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 ): " 
          << fMVAVar_GammaIso_DR0p0To0p1 << " "
          << fMVAVar_GammaIso_DR0p1To0p2 << " "
          << fMVAVar_GammaIso_DR0p2To0p3 << " "
          << fMVAVar_GammaIso_DR0p3To0p4 << " "
          << fMVAVar_GammaIso_DR0p4To0p5 << endl;
    cout  << "PF Neutral Hadron Iso ( 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 ): " 
          << fMVAVar_NeutralHadronIso_DR0p0To0p1 << " "
          << fMVAVar_NeutralHadronIso_DR0p1To0p2 << " "
          << fMVAVar_NeutralHadronIso_DR0p2To0p3 << " "
          << fMVAVar_NeutralHadronIso_DR0p3To0p4 << " "
          << fMVAVar_NeutralHadronIso_DR0p4To0p5 << " "
          << endl;
    cout  << "Rho : " << Rho << endl;
    cout << " ### MVA " << mva << endl;
  }



  return mva;
}




#endif

void EGammaMvaEleEstimator::bindVariables() {

  // this binding is needed for variables that sometime diverge. 


  if(fMVAVar_fbrem < -1.)
    fMVAVar_fbrem = -1.;	
  
  fMVAVar_deta = fabs(fMVAVar_deta);
  if(fMVAVar_deta > 0.06)
    fMVAVar_deta = 0.06;
  
  
  fMVAVar_dphi = fabs(fMVAVar_dphi);
  if(fMVAVar_dphi > 0.6)
    fMVAVar_dphi = 0.6;
  

  if(fMVAVar_EoP > 20.)
    fMVAVar_EoP = 20.;
  
  if(fMVAVar_eleEoPout > 20.)
    fMVAVar_eleEoPout = 20.;
  
  
  fMVAVar_detacalo = fabs(fMVAVar_detacalo);
  if(fMVAVar_detacalo > 0.2)
    fMVAVar_detacalo = 0.2;
  
  if(fMVAVar_OneMinusE1x5E5x5 < -1.)
    fMVAVar_OneMinusE1x5E5x5 = -1;
  
  if(fMVAVar_OneMinusE1x5E5x5 > 2.)
    fMVAVar_OneMinusE1x5E5x5 = 2.; 
  
  
  
  if(fMVAVar_R9 > 5)
    fMVAVar_R9 = 5;
  
  if(fMVAVar_gsfchi2 > 200.)
    fMVAVar_gsfchi2 = 200;
  
  
  if(fMVAVar_kfchi2 > 10.)
    fMVAVar_kfchi2 = 10.;
  
  
  // Needed for a bug in CMSSW_420, fixed in more recent CMSSW versions
  if(std::isnan(fMVAVar_spp))
    fMVAVar_spp = 0.;	
  
  
  return;
}








