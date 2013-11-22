#ifndef GEMCSCPadDigiValidation_H
#define GEMCSCPadDigiValidation_H

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "Validation/MuonGEMDigis/interface/GEMBaseValidation.h"


class GEMCSCPadDigiValidation : public GEMBaseValidation
{
public:
  GEMCSCPadDigiValidation(DQMStore* dbe,
                         const edm::InputTag & inputTag);
  ~GEMCSCPadDigiValidation();
  void analyze(const edm::Event& e, const edm::EventSetup&);



 private:

  MonitorElement* theCSCPad_xy_rm1_l1;
  MonitorElement* theCSCPad_xy_rm1_l2;
  MonitorElement* theCSCPad_xy_rp1_l1;
  MonitorElement* theCSCPad_xy_rp1_l2;

  MonitorElement* theCSCPad_phipad_rm1_l1;
  MonitorElement* theCSCPad_phipad_rm1_l2;
  MonitorElement* theCSCPad_phipad_rp1_l1;
  MonitorElement* theCSCPad_phipad_rp1_l2;


  MonitorElement* theCSCPad_rm1_l1;
  MonitorElement* theCSCPad_rm1_l2;
  MonitorElement* theCSCPad_rp1_l1;
  MonitorElement* theCSCPad_rp1_l2;


  MonitorElement* theCSCPad_bx_rm1_l1;
  MonitorElement* theCSCPad_bx_rm1_l2;
  MonitorElement* theCSCPad_bx_rp1_l1;
  MonitorElement* theCSCPad_bx_rp1_l2;


  MonitorElement* theCSCPad_zr_rm1;
  MonitorElement* theCSCPad_zr_rp1;

};

#endif
