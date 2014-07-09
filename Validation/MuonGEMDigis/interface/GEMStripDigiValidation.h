#ifndef GEMStripDigiValidation_H
#define GEMStripDigiValidation_H

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
#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include <TMath.h>

class GEMStripDigiValidation : public GEMBaseValidation
{
public:
  GEMStripDigiValidation(DQMStore* dbe,
                         edm::EDGetToken& stripToken, const edm::ParameterSet& pbInfo);
  ~GEMStripDigiValidation();
  void analyze(const edm::Event& e, const edm::EventSetup&);
  void bookHisto(const GEMGeometry* geom) ; 
  void savePhiPlot() ; 

 private:

  MonitorElement* theStrip_xy[2][3][2];
  MonitorElement* theStrip_phistrip[2][3][2];
  MonitorElement* theStrip[2][3][2];
  MonitorElement* theStrip_bx[2][3][2];
  MonitorElement* theStrip_zr[2][3][2];
  std::map< std::string, MonitorElement* > theStrip_ro_phi;
	std::map< std::string, MonitorElement* > theStrip_st_dphi;
  std::map< std::string, MonitorElement* > theStrip_phiz_st_ch;
};

#endif
