#ifndef GEMHitsValidation_H
#define GEMHitsValidation_H

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "Validation/MuonGEMDigis/interface/GEMBaseValidation.h"



#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"

class GEMHitsValidation : public GEMBaseValidation
{
public:
  GEMHitsValidation(DQMStore* dbe,
                         const edm::InputTag & inputTag);
  ~GEMHitsValidation();
  void analyze(const edm::Event& e, const edm::EventSetup&);

 private:

  MonitorElement* gem_sh_xy_rm1_l1;
  MonitorElement* gem_sh_xy_rm1_l2;
  MonitorElement* gem_sh_xy_rp1_l1;
  MonitorElement* gem_sh_xy_rp1_l2;

  MonitorElement* gem_sh_zr_rm1;
  MonitorElement* gem_sh_zr_rp1;

};

#endif
