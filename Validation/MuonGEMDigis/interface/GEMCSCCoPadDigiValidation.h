#ifndef GEMCSCCoPadDigiValidation_H
#define GEMCSCCoPadDigiValidation_H

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Validation/MuonGEMHits/interface/GEMBaseValidation.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/GEMDigi/interface/GEMCSCPadDigiCollection.h"
#include <TMath.h>


class GEMCSCCoPadDigiValidation : public GEMBaseValidation
{
public:
  GEMCSCCoPadDigiValidation(DQMStore* dbe,
                         const edm::InputTag & inputTag);
  ~GEMCSCCoPadDigiValidation();
  void analyze(const edm::Event& e, const edm::EventSetup&);
  void bookHisto();
 private:

  MonitorElement* theCSCCoPad_xy_rm1;
  MonitorElement* theCSCCoPad_xy_rp1;

  MonitorElement* theCSCCoPad_phipad_rm1;
  MonitorElement* theCSCCoPad_phipad_rp1;


  MonitorElement* theCSCCoPad_rm1;
  MonitorElement* theCSCCoPad_rp1;


  MonitorElement* theCSCCoPad_bx_rm1;
  MonitorElement* theCSCCoPad_bx_rp1;


  MonitorElement* theCSCCoPad_zr_rm1;
  MonitorElement* theCSCCoPad_zr_rp1;

};

#endif
