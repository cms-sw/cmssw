#ifndef GEMHitsValidation_H
#define GEMHitsValidation_H

#include "Validation/MuonGEMHits/interface/GEMBaseValidation.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"



class GEMHitsValidation : public GEMBaseValidation
{
public:
  GEMHitsValidation(DQMStore* dbe,
                         edm::EDGetToken& InputTagToken, const edm::ParameterSet& pbInfo);
  ~GEMHitsValidation();
  void analyze(const edm::Event& e, const edm::EventSetup&);
  void bookHisto(const GEMGeometry*);
 private:

  MonitorElement* gem_sh_xy[2][3][2];
  std::map< std::string, MonitorElement* > gem_sh_xy_st_ch;
  MonitorElement* gem_sh_zr[2][3][2];

  MonitorElement* gem_sh_tof[2][3][2];
  MonitorElement* gem_sh_tofMu[2][3][2];
  MonitorElement* gem_sh_eloss[2][3][2];
  MonitorElement* gem_sh_elossMu[2][3][2];

  Int_t npart;

};

#endif
