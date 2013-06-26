#ifndef SimG4CMS_Calo_HFPMTHitAnalyzer_H
#define SimG4CMS_Calo_HFPMTHitAnalyzer_H

#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>
#include <TProfile.h>

class HFPMTHitAnalyzer : public edm::EDAnalyzer {

public:

  explicit HFPMTHitAnalyzer(const edm::ParameterSet&);
  ~HFPMTHitAnalyzer();

private:

  virtual void beginJob() ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob();
  void analyzeHits  (std::vector<PCaloHit> &,const std::vector<SimTrack> &);

  //user parameters
  std::string sourceLabel, g4Label, hcalHits;

  int event_no;

  //root file name and its objects
  TH1F *h_HFDepHit, *hHF_MC_e, *h_HFEta[3], *h_HFPhi[3];

  TH1F *hHF_e_1[3],  *hHF_em_1[3],  *hHF_had_1[3];
  TH1F *hHF_e_2[3],  *hHF_em_2[3],  *hHF_had_2[3];
  TH1F *hHF_e_12[3], *hHF_em_12[3], *hHF_had_12[3];

  TH1F *hHF1_time[3],  *hHF1_time_Ewt[3];
  TH1F *hHF2_time[3],  *hHF2_time_Ewt[3];
  TH1F *hHF12_time[3], *hHF12_time_Ewt[3];
      
};

#endif
