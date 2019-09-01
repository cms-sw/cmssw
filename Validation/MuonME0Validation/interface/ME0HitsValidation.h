#ifndef ME0HitsValidation_H
#define ME0HitsValidation_H

#include "Validation/MuonME0Validation/interface/ME0BaseValidation.h"

class ME0HitsValidation : public ME0BaseValidation {
public:
  explicit ME0HitsValidation(const edm::ParameterSet &);
  ~ME0HitsValidation() override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void analyze(const edm::Event &e, const edm::EventSetup &) override;

private:
  MonitorElement *me0_sh_xy[2][6];
  MonitorElement *me0_sh_zr[2][6];
  MonitorElement *me0_sh_tot_zr[2];

  MonitorElement *me0_sh_tof[2][6];
  MonitorElement *me0_sh_tofMu[2][6];
  MonitorElement *me0_sh_eloss[2][6];
  MonitorElement *me0_sh_elossMu[2][6];

  Int_t npart;
};

#endif
