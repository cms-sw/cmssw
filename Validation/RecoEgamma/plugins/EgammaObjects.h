#ifndef EgammaObjects_h
#define EgammaObjects_h

#include <string>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"

#include "TFile.h"
#include "TH1D.h"
#include "TH2D.h"

class EgammaObjects : public edm::EDAnalyzer {
  public:
    explicit EgammaObjects( const edm::ParameterSet& );
    ~EgammaObjects();

    virtual void analyze( const edm::Event&, const edm::EventSetup& );
    virtual void beginJob();
    virtual void endJob();

  private:
    TFile*  rootFile_;

    edm::InputTag MCTruthCollection_;
    edm::InputTag RecoCollection_;

    int particleID; 
    std::string particleString;
    
    int EtCut; 

    double hist_min_Et_;
    double hist_max_Et_;
    int    hist_bins_Et_;
    
    double hist_min_E_;
    double hist_max_E_;
    int    hist_bins_E_;
    
    double hist_min_Eta_;
    double hist_max_Eta_;
    int    hist_bins_Eta_;
    
    double hist_min_Phi_;
    double hist_max_Phi_;
    int    hist_bins_Phi_;
    
    double hist_min_EtOverTruth_;
    double hist_max_EtOverTruth_;
    int    hist_bins_EtOverTruth_;
    
    double hist_min_EOverTruth_;
    double hist_max_EOverTruth_;
    int    hist_bins_EOverTruth_;

    double hist_min_EtaOverTruth_;
    double hist_max_EtaOverTruth_;
    int    hist_bins_EtaOverTruth_;
    
    double hist_min_PhiOverTruth_;
    double hist_max_PhiOverTruth_;
    int    hist_bins_PhiOverTruth_;

    double hist_min_deltaEta_;
    double hist_max_deltaEta_;
    int    hist_bins_deltaEta_;
    
    double hist_min_deltaPhi_;
    double hist_max_deltaPhi_;
    int    hist_bins_deltaPhi_;
    
    double hist_min_recoMass_;
    double hist_max_recoMass_;
    int    hist_bins_recoMass_;
    
    TH1D* hist_Et_;
    TH1D* hist_EtOverTruth_;
    TH1D* hist_EtEfficiency_;
    TH1D* hist_EtNumRecoOverNumTrue_;
    TH1D* hist_EtOverTruthVsEt_;
    TH1D* hist_EtOverTruthVsE_;
    TH1D* hist_EtOverTruthVsEta_;
    TH1D* hist_EtOverTruthVsPhi_;
    TH1D* hist_resolutionEtVsEt_;
    TH1D* hist_resolutionEtVsE_;    
    TH1D* hist_resolutionEtVsEta_;
    TH1D* hist_resolutionEtVsPhi_;
    
    TH1D* hist_E_;
    TH1D* hist_EOverTruth_;
    TH1D* hist_EEfficiency_;
    TH1D* hist_ENumRecoOverNumTrue_;
    TH1D* hist_EOverTruthVsEt_;
    TH1D* hist_EOverTruthVsE_;
    TH1D* hist_EOverTruthVsEta_;
    TH1D* hist_EOverTruthVsPhi_;
    TH1D* hist_resolutionEVsEt_;
    TH1D* hist_resolutionEVsE_;    
    TH1D* hist_resolutionEVsEta_;
    TH1D* hist_resolutionEVsPhi_;
    
    TH1D* hist_Eta_;
    TH1D* hist_EtaOverTruth_;
    TH1D* hist_EtaEfficiency_;
    TH1D* hist_EtaNumRecoOverNumTrue_;
    TH1D* hist_deltaEtaVsEt_;
    TH1D* hist_deltaEtaVsE_;
    TH1D* hist_deltaEtaVsEta_;
    TH1D* hist_deltaEtaVsPhi_;
    TH1D* hist_resolutionEtaVsEt_;
    TH1D* hist_resolutionEtaVsE_;    
    TH1D* hist_resolutionEtaVsEta_;
    TH1D* hist_resolutionEtaVsPhi_;
    
    TH1D* hist_Phi_;
    TH1D* hist_PhiOverTruth_;
    TH1D* hist_PhiEfficiency_;
    TH1D* hist_PhiNumRecoOverNumTrue_;
    TH1D* hist_deltaPhiVsEt_;
    TH1D* hist_deltaPhiVsE_;
    TH1D* hist_deltaPhiVsEta_;
    TH1D* hist_deltaPhiVsPhi_;
    TH1D* hist_resolutionPhiVsEt_;
    TH1D* hist_resolutionPhiVsE_;    
    TH1D* hist_resolutionPhiVsEta_;
    TH1D* hist_resolutionPhiVsPhi_;
    
    TH1D* hist_All_recoMass_;
    TH1D* hist_BarrelOnly_recoMass_;
    TH1D* hist_EndcapOnly_recoMass_;
    TH1D* hist_Mixed_recoMass_;

    TH1D* hist_recoMass_withBackgroud_NoEtCut_;
    TH1D* hist_recoMass_withBackgroud_5EtCut_;
    TH1D* hist_recoMass_withBackgroud_10EtCut_;
    TH1D* hist_recoMass_withBackgroud_20EtCut_;
    
    TH2D* _TEMP_scatterPlot_EtOverTruthVsEt_;
    TH2D* _TEMP_scatterPlot_EtOverTruthVsE_;
    TH2D* _TEMP_scatterPlot_EtOverTruthVsEta_;
    TH2D* _TEMP_scatterPlot_EtOverTruthVsPhi_;
    
    TH2D* _TEMP_scatterPlot_EOverTruthVsEt_;
    TH2D* _TEMP_scatterPlot_EOverTruthVsE_;
    TH2D* _TEMP_scatterPlot_EOverTruthVsEta_;
    TH2D* _TEMP_scatterPlot_EOverTruthVsPhi_;
    
    TH2D* _TEMP_scatterPlot_deltaEtaVsEt_;
    TH2D* _TEMP_scatterPlot_deltaEtaVsE_;    
    TH2D* _TEMP_scatterPlot_deltaEtaVsEta_;
    TH2D* _TEMP_scatterPlot_deltaEtaVsPhi_;
    
    TH2D* _TEMP_scatterPlot_deltaPhiVsEt_;
    TH2D* _TEMP_scatterPlot_deltaPhiVsE_;
    TH2D* _TEMP_scatterPlot_deltaPhiVsEta_;
    TH2D* _TEMP_scatterPlot_deltaPhiVsPhi_;

    void loadCMSSWObjects(const edm::ParameterSet& ps);
    void loadHistoParameters(const edm::ParameterSet& ps);
  
    void createBookedHistoObjects();
    void createTempHistoObjects();
    
    void analyzePhotons( const edm::Event&, const edm::EventSetup& );
    void analyzeElectrons( const edm::Event&, const edm::EventSetup& );    

    void getEfficiencyHistosViaDividing();
    void getDeltaResHistosViaSlicing();
    void fitHistos(); 
    
    void applyLabels(); 
    void setDrawOptions();
    void saveHistos();

    double findRecoMass(reco::Photon pOne, reco::Photon pTwo);
    double findRecoMass(reco::GsfElectron eOne, reco::GsfElectron eTwo);
    
    float ecalEta(float EtaParticle , float Zvertex, float plane_Radius);
};
#endif

