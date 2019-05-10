#ifndef ME0RecHitsValidation_H
#define ME0RecHitsValidation_H

#include "Validation/MuonME0Validation/interface/ME0BaseValidation.h"

// Data Format
#include "DataFormats/GEMRecHit/interface/ME0RecHitCollection.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include <DataFormats/GEMRecHit/interface/ME0RecHit.h>

class ME0RecHitsValidation : public ME0BaseValidation {
public:
  explicit ME0RecHitsValidation(const edm::ParameterSet &);
  ~ME0RecHitsValidation() override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void analyze(const edm::Event &e, const edm::EventSetup &) override;

private:
  MonitorElement *me0_rh_xy[2][6];
  MonitorElement *me0_rh_zr[2];

  MonitorElement *me0_rh_DeltaX[2][6];
  MonitorElement *me0_rh_DeltaY[2][6];
  MonitorElement *me0_rh_PullX[2][6];
  MonitorElement *me0_rh_PullY[2][6];

  edm::EDGetToken InputTagToken_RecHit;

  Int_t npart;
};

#endif
