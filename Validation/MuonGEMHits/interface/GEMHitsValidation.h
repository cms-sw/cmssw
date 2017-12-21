#ifndef GEMHitsValidation_H
#define GEMHitsValidation_H

#include "Validation/MuonGEMHits/interface/GEMBaseValidation.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"



class GEMHitsValidation : public GEMBaseValidation
{
public:
  explicit GEMHitsValidation( const edm::ParameterSet& );
  ~GEMHitsValidation() override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void analyze(const edm::Event& e, const edm::EventSetup&) override;
 private:

  // Detail plots
  MonitorElement* gem_sh_xy[2][3][2];
  MonitorElement* gem_sh_zr[2][3][2];
  MonitorElement* gem_sh_tof[2][3][2];
  MonitorElement* gem_sh_tofMu[2][3][2];
  MonitorElement* gem_sh_eloss[2][3][2];
  MonitorElement* gem_sh_elossMu[2][3][2];

  std::unordered_map< UInt_t , MonitorElement* > gem_sh_xy_st_ch;

  // Simple plots
  std::unordered_map< UInt_t , MonitorElement* > Hit_dcEta;
  std::unordered_map< UInt_t , MonitorElement* > Hit_simple_zr;
  std::unordered_map< UInt_t , MonitorElement* > gem_sh_simple_tofMu;
  std::unordered_map< UInt_t , MonitorElement* > gem_sh_simple_elossMu;


  edm::EDGetToken InputTagToken_;
  int nBinXY_;
  bool detailPlot_;
};

#endif
