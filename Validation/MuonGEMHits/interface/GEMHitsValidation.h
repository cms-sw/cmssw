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
#include "FWCore/ParameterSet/interface/ParameterSet.h"


#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"

class GEMHitsValidation : public GEMBaseValidation
{
public:
  GEMHitsValidation(DQMStore* dbe,
                         const edm::InputTag & inputTag, const edm::ParameterSet& cfg);
  ~GEMHitsValidation();
  void analyze(const edm::Event& e, const edm::EventSetup&);

 private:

  MonitorElement* gem_sh_xy_rm1_l1[3];
  MonitorElement* gem_sh_xy_rm1_l2[3];
  MonitorElement* gem_sh_xy_rp1_l1[3];
  MonitorElement* gem_sh_xy_rp1_l2[3];

  MonitorElement* gem_sh_zr_rm1[3];
  MonitorElement* gem_sh_zr_rp1[3];

  MonitorElement* gem_sh_tof_rm1_l1[3];
  MonitorElement* gem_sh_tof_rm1_l2[3];
  MonitorElement* gem_sh_tof_rp1_l1[3];
  MonitorElement* gem_sh_tof_rp1_l2[3];


  MonitorElement* gem_sh_pabs[3];
  MonitorElement* gem_sh_pdgid[3];
  MonitorElement* gem_sh_global_eta[3];
  MonitorElement* gem_sh_energyloss[3];

  Int_t npart;
  Int_t nlayer;
  Int_t nregion;


};

#endif
