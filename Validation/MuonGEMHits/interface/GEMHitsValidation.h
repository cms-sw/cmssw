#ifndef GEMHitsValidation_H
#define GEMHitsValidation_H

#include "Validation/MuonGEMHits/interface/GEMBaseValidation.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"

class GEMHitsValidation : public GEMBaseValidation
{
public:
  GEMHitsValidation(DQMStore* dbe,
                         const edm::InputTag & inputTag);
  ~GEMHitsValidation();
  void analyze(const edm::Event& e, const edm::EventSetup&);
  void bookHisto(const GEMGeometry*);
 private:

  MonitorElement* gem_sh_xy[2][3][2][2];
  MonitorElement* gem_sh_zr[2][3][2];

  MonitorElement* gem_sh_tof[2][3][2][2];
  MonitorElement* gem_sh_eloss[2][3][2][2];

  Int_t npart;

};

#endif
