#ifndef GEMCSCPadDigiValidation_H
#define GEMCSCPadDigiValidation_H

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Validation/MuonGEMHits/interface/GEMBaseValidation.h"
#include "DataFormats/GEMDigi/interface/GEMCSCPadDigiCollection.h"

#include "DataFormats/Common/interface/Handle.h"
#include <TMath.h>
class GEMCSCPadDigiValidation : public GEMBaseValidation
{
public:
  GEMCSCPadDigiValidation(DQMStore* dbe,
                         edm::EDGetToken& inputToken, const edm::ParameterSet& pbInfo);
  ~GEMCSCPadDigiValidation();
  void analyze(const edm::Event& e, const edm::EventSetup&);
  void bookHisto(const GEMGeometry* geom);
 private:

  MonitorElement* theCSCPad_xy[2][3][2];
  MonitorElement* theCSCPad_phipad[2][3][2];
  MonitorElement* theCSCPad[2][3][2];
  MonitorElement* theCSCPad_bx[2][3][2];
  MonitorElement* theCSCPad_zr[2][3][2];
};

#endif
