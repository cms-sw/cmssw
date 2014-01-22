#ifndef GEMStripDigiValidation_H
#define GEMStripDigiValidation_H

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "Validation/MuonGEMDigis/interface/GEMBaseValidation.h"


class GEMStripDigiValidation : public GEMBaseValidation
{
public:
  GEMStripDigiValidation(DQMStore* dbe,
                         const edm::InputTag & inputTag);
  ~GEMStripDigiValidation();
  void analyze(const edm::Event& e, const edm::EventSetup&);



 private:

  MonitorElement* theStrip_xy_rm1_l1;
  MonitorElement* theStrip_xy_rm1_l2;
  MonitorElement* theStrip_xy_rp1_l1;
  MonitorElement* theStrip_xy_rp1_l2;

  MonitorElement* theStrip_phistrip_rm1_l1;
  MonitorElement* theStrip_phistrip_rm1_l2;
  MonitorElement* theStrip_phistrip_rp1_l1;
  MonitorElement* theStrip_phistrip_rp1_l2;


  MonitorElement* theStrip_rm1_l1;
  MonitorElement* theStrip_rm1_l2;
  MonitorElement* theStrip_rp1_l1;
  MonitorElement* theStrip_rp1_l2;


  MonitorElement* theStrip_bx_rm1_l1;
  MonitorElement* theStrip_bx_rm1_l2;
  MonitorElement* theStrip_bx_rp1_l1;
  MonitorElement* theStrip_bx_rp1_l2;


  MonitorElement* theStrip_zr_rm1;
  MonitorElement* theStrip_zr_rp1;

};

#endif
