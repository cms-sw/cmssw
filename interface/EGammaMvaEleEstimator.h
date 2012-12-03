//--------------------------------------------------------------------------------------------------
// $Id $
//
// EGammaMvaEleEstimator
//
// Helper Class for applying MVA electron ID selection
//
// Authors: D.Benedetti, E.DiMaro, S.Xie
//--------------------------------------------------------------------------------------------------


/// --> NOTE if you want to use this class as standalone without the CMSSW part 
///  you need to uncomment the below line and compile normally with scramv1 b 
///  Then you need just to load it in your root macro the lib with the correct path, eg:
///  gSystem->Load("/data/benedet/CMSSW_5_2_2/lib/slc5_amd64_gcc462/pluginEGammaEGammaAnalysisTools.so");

//#define STANDALONE   // <---- this line

#ifndef EGammaMvaEleEstimator_H
#define EGammaMvaEleEstimator_H

#ifndef STANDALONE
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "EgammaAnalysis/ElectronTools/interface/ElectronEffectiveArea.h"
#endif
#include <vector>
#include <TROOT.h>
#include "TMVA/Factory.h"
#include "TMVA/Tools.h"
#include "TMVA/Reader.h"

class EGammaMvaEleEstimator{
  public:
    EGammaMvaEleEstimator();
    ~EGammaMvaEleEstimator(); 
  
    enum MVAType {
      kTrig = 0,                     // MVA for triggering electrons     
      kNonTrig,                      // MVA for non-triggering electrons     
      kIsoRings,                     // Isolation MVA for non-trigger electrons
      kTrigIDIsoCombined,            // ID+Iso Combined MVA for triggering electrons
      kTrigIDIsoCombinedPUCorrected  // ID+Iso Combined MVA for triggering electrons
    };
  
    void     initialize( std::string methodName,
                         std::string weightsfile,
                         EGammaMvaEleEstimator::MVAType type);
    void     initialize( std::string methodName,
                         EGammaMvaEleEstimator::MVAType type,
                         Bool_t useBinnedVersion,
                         std::vector<std::string> weightsfiles );
    
    Bool_t   isInitialized() const { return fisInitialized; }
    UInt_t   GetMVABin(double eta,double pt ) const;
    
    void bindVariables();
    
#ifndef STANDALONE
    Double_t mvaValue(const reco::GsfElectron& ele, 
                      const reco::Vertex& vertex, 
                      const TransientTrackBuilder& transientTrackBuilder,
                      EcalClusterLazyTools myEcalCluster,
                      bool printDebug = kFALSE);
    
    Double_t isoMvaValue(const reco::GsfElectron& ele, 
                         const reco::Vertex& vertex, 
                         const reco::PFCandidateCollection &PFCandidates,
                         double Rho,
                         ElectronEffectiveArea::ElectronEffectiveAreaTarget EATarget,
                         const reco::GsfElectronCollection &IdentifiedElectrons,
                         const reco::MuonCollection &IdentifiedMuons,
                         bool printDebug = kFALSE);
    Double_t IDIsoCombinedMvaValue(const reco::GsfElectron& ele, 
                                   const reco::Vertex& vertex, 
                                   const TransientTrackBuilder& transientTrackBuilder,
                                   EcalClusterLazyTools myEcalCluster,
                                   const reco::PFCandidateCollection &PFCandidates,
                                   double Rho,
                                   ElectronEffectiveArea::ElectronEffectiveAreaTarget EATarget,
                                   bool printDebug = kFALSE);
    
    Double_t isoMvaValue(Double_t Pt,
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
                         Bool_t printDebug = kFALSE );
#endif
  
    Double_t mvaValue(Double_t fbrem, 
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
                      Double_t e1x5e5x5,
                      Double_t R9,
                      Double_t HoE,
                      Double_t EoP,
                      Double_t IoEmIoP,
                      Double_t eleEoPout,
                      Double_t PreShowerOverRaw,
                      Double_t d0,
                      Double_t ip3d,
                      Double_t eta,
                      Double_t pt,
                      Bool_t printDebug = kFALSE );
 
    Double_t mvaValue(Double_t fbrem, 
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
                      Double_t e1x5e5x5,
                      Double_t R9,
                      Double_t HoE,
                      Double_t EoP,
                      Double_t IoEmIoP,
                      Double_t eleEoPout,
                      Double_t PreShowerOverRaw,
                      Double_t eta,
                      Double_t pt,
                      Bool_t printDebug = kFALSE );
 
    Double_t IDIsoCombinedMvaValue(Double_t fbrem, 
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
                                   Bool_t printDebug = kFALSE);



  private:

    std::vector<TMVA::Reader*> fTMVAReader;
    std::string                fMethodname;
    Bool_t                     fisInitialized;
    MVAType                    fMVAType;
    Bool_t                     fUseBinnedVersion;
    UInt_t                     fNMVABins;

    Float_t                    fMVAVar_fbrem;
    Float_t                    fMVAVar_kfchi2;
    Float_t                    fMVAVar_kfhits;    //number of layers
    Float_t                    fMVAVar_kfhitsall; //number of hits
    Float_t                    fMVAVar_gsfchi2;

    Float_t                    fMVAVar_deta;
    Float_t                    fMVAVar_dphi;
    Float_t                    fMVAVar_detacalo;

    Float_t                    fMVAVar_see;
    Float_t                    fMVAVar_spp;
    Float_t                    fMVAVar_etawidth;
    Float_t                    fMVAVar_phiwidth;
    Float_t                    fMVAVar_OneMinusE1x5E5x5;
    Float_t                    fMVAVar_R9;

    Float_t                    fMVAVar_HoE;
    Float_t                    fMVAVar_EoP;
    Float_t                    fMVAVar_IoEmIoP;
    Float_t                    fMVAVar_eleEoPout;
    Float_t                    fMVAVar_EoPout; 
    Float_t                    fMVAVar_PreShowerOverRaw;

    Float_t                    fMVAVar_d0;
    Float_t                    fMVAVar_ip3d;
    Float_t                    fMVAVar_ip3dSig;

    Float_t                    fMVAVar_eta;
    Float_t                    fMVAVar_pt;
    Float_t                    fMVAVar_rho;
  
    Float_t                    fMVAVar_ChargedIso_DR0p0To0p1;
    Float_t                    fMVAVar_ChargedIso_DR0p1To0p2;
    Float_t                    fMVAVar_ChargedIso_DR0p2To0p3;
    Float_t                    fMVAVar_ChargedIso_DR0p3To0p4;
    Float_t                    fMVAVar_ChargedIso_DR0p4To0p5;
    Float_t                    fMVAVar_GammaIso_DR0p0To0p1;
    Float_t                    fMVAVar_GammaIso_DR0p1To0p2;
    Float_t                    fMVAVar_GammaIso_DR0p2To0p3;
    Float_t                    fMVAVar_GammaIso_DR0p3To0p4;
    Float_t                    fMVAVar_GammaIso_DR0p4To0p5;
    Float_t                    fMVAVar_NeutralHadronIso_DR0p0To0p1;
    Float_t                    fMVAVar_NeutralHadronIso_DR0p1To0p2;
    Float_t                    fMVAVar_NeutralHadronIso_DR0p2To0p3;
    Float_t                    fMVAVar_NeutralHadronIso_DR0p3To0p4;
    Float_t                    fMVAVar_NeutralHadronIso_DR0p4To0p5;
 
};

#endif
