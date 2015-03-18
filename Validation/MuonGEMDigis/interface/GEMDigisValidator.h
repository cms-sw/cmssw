#ifndef GEMDigisValidation_H
#define GEMDigisValidation_H

#include "Validation/MuonGEMDigis/interface/GEMBaseValidation.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"



class GEMDigisValidation : public GEMBaseValidation
{
public:
  explicit GEMDigisValidation( const edm::ParameterSet& );
  ~GEMDigisValidation();
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void analyze(const edm::Event& e, const edm::EventSetup&) override;
 private:

  MonitorElement* gem_sh_xy[2][3][2];
  std::map< std::string, MonitorElement* > gem_sh_xy_st_ch;
  MonitorElement* gem_sh_zr[2][3][2];

  MonitorElement* gem_sh_tof[2][3][2];
  MonitorElement* gem_sh_tofMu[2][3][2];
  MonitorElement* gem_sh_eloss[2][3][2];
  MonitorElement* gem_sh_elossMu[2][3][2];

  Int_t npart;
  edm::EDGetToken InputTagToken_;

};

#endif
