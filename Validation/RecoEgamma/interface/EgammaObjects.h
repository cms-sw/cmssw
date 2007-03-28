#ifndef EgammaObjects_h
#define EgammaObjects_h

#include <string>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"

#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PixelMatchGsfElectron.h"

#include "TFile.h"

class EgammaObjects : public edm::EDAnalyzer {
  public:
    explicit EgammaObjects( const edm::ParameterSet& );
    ~EgammaObjects();

    virtual void analyze( const edm::Event&, const edm::EventSetup& );
    virtual void beginJob(edm::EventSetup const&);
    virtual void endJob();

  private:
    std::string outputFile_;
    TFile*  rootFile_;

    DaqMonitorBEInterface * dbe;

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
    
    TH1F* hist_Et_;
    TH1F* hist_EtOverTruth_;
    TH1F* hist_EtEfficiency_;
    TH1F* hist_EtNumRecoOverNumTrue_;
    TH1F* hist_EtOverTruthVsEt_;
    TH1F* hist_EtOverTruthVsE_;
    TH1F* hist_EtOverTruthVsEta_;
    TH1F* hist_EtOverTruthVsPhi_;
    TH1F* hist_resolutionEtVsEt_;
    TH1F* hist_resolutionEtVsE_;    
    TH1F* hist_resolutionEtVsEta_;
    TH1F* hist_resolutionEtVsPhi_;
    
    TH1F* hist_E_;
    TH1F* hist_EOverTruth_;
    TH1F* hist_EEfficiency_;
    TH1F* hist_ENumRecoOverNumTrue_;
    TH1F* hist_EOverTruthVsEt_;
    TH1F* hist_EOverTruthVsE_;
    TH1F* hist_EOverTruthVsEta_;
    TH1F* hist_EOverTruthVsPhi_;
    TH1F* hist_resolutionEVsEt_;
    TH1F* hist_resolutionEVsE_;    
    TH1F* hist_resolutionEVsEta_;
    TH1F* hist_resolutionEVsPhi_;
    
    TH1F* hist_Eta_;
    TH1F* hist_EtaOverTruth_;
    TH1F* hist_EtaEfficiency_;
    TH1F* hist_EtaNumRecoOverNumTrue_;
    TH1F* hist_deltaEtaVsEt_;
    TH1F* hist_deltaEtaVsE_;
    TH1F* hist_deltaEtaVsEta_;
    TH1F* hist_deltaEtaVsPhi_;
    TH1F* hist_resolutionEtaVsEt_;
    TH1F* hist_resolutionEtaVsE_;    
    TH1F* hist_resolutionEtaVsEta_;
    TH1F* hist_resolutionEtaVsPhi_;
    
    TH1F* hist_Phi_;
    TH1F* hist_PhiOverTruth_;
    TH1F* hist_PhiEfficiency_;
    TH1F* hist_PhiNumRecoOverNumTrue_;
    TH1F* hist_deltaPhiVsEt_;
    TH1F* hist_deltaPhiVsE_;
    TH1F* hist_deltaPhiVsEta_;
    TH1F* hist_deltaPhiVsPhi_;
    TH1F* hist_resolutionPhiVsEt_;
    TH1F* hist_resolutionPhiVsE_;    
    TH1F* hist_resolutionPhiVsEta_;
    TH1F* hist_resolutionPhiVsPhi_;
    
    TH1F* hist_All_recoMass_;
    TH1F* hist_BarrelOnly_recoMass_;
    TH1F* hist_EndcapOnly_recoMass_;
    TH1F* hist_Mixed_recoMass_;

    TH1F* hist_recoMass_withBackgroud_NoEtCut_;
    TH1F* hist_recoMass_withBackgroud_5EtCut_;
    TH1F* hist_recoMass_withBackgroud_10EtCut_;
    TH1F* hist_recoMass_withBackgroud_20EtCut_;
    
    TH2F* _TEMP_scatterPlot_EtOverTruthVsEt_;
    TH2F* _TEMP_scatterPlot_EtOverTruthVsE_;
    TH2F* _TEMP_scatterPlot_EtOverTruthVsEta_;
    TH2F* _TEMP_scatterPlot_EtOverTruthVsPhi_;
    
    TH2F* _TEMP_scatterPlot_EOverTruthVsEt_;
    TH2F* _TEMP_scatterPlot_EOverTruthVsE_;
    TH2F* _TEMP_scatterPlot_EOverTruthVsEta_;
    TH2F* _TEMP_scatterPlot_EOverTruthVsPhi_;
    
    TH2F* _TEMP_scatterPlot_deltaEtaVsEt_;
    TH2F* _TEMP_scatterPlot_deltaEtaVsE_;    
    TH2F* _TEMP_scatterPlot_deltaEtaVsEta_;
    TH2F* _TEMP_scatterPlot_deltaEtaVsPhi_;
    
    TH2F* _TEMP_scatterPlot_deltaPhiVsEt_;
    TH2F* _TEMP_scatterPlot_deltaPhiVsE_;
    TH2F* _TEMP_scatterPlot_deltaPhiVsEta_;
    TH2F* _TEMP_scatterPlot_deltaPhiVsPhi_;

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
    void saveHistosToRoot();
    void covertRootFileToDQMFile();

    double findRecoMass(reco::Photon pOne, reco::Photon pTwo);
    double findRecoMass(reco::PixelMatchGsfElectron eOne, reco::PixelMatchGsfElectron eTwo);
};
#endif

