#ifndef GEMHitsValidation_H
#define GEMHitsValidation_H

#include "Validation/MuonGEMHits/interface/GEMBaseValidation.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"



class GEMHitsValidation : public GEMBaseValidation
{
public:
  explicit GEMHitsValidation( const edm::ParameterSet& );
  ~GEMHitsValidation();
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void analyze(const edm::Event& e, const edm::EventSetup&) override;
 private:

  // Detail plots
  MonitorElement* gem_sh_xy[2][3][2];
  std::map< std::string, MonitorElement* > gem_sh_xy_st_ch;
  MonitorElement* gem_sh_zr[2][3][2];
  MonitorElement* gem_sh_tof[2][3][2];
  MonitorElement* gem_sh_tofMu[2][3][2];
  MonitorElement* gem_sh_eloss[2][3][2];
  MonitorElement* gem_sh_elossMu[2][3][2];

  // Simple plots
  std::map< UInt_t , MonitorElement* > Hit_dcEta;
  std::map< UInt_t , MonitorElement* > Hit_simple_zr;

  Int_t npart;
  

};

#endif
