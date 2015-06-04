#ifndef GEMStripDigiValidation_H
#define GEMStripDigiValidation_H


#include "Validation/MuonGEMHits/interface/GEMBaseValidation.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include <TMath.h>

class GEMStripDigiValidation : public GEMBaseValidation
{
public:
  explicit GEMStripDigiValidation(const edm::ParameterSet&);
  ~GEMStripDigiValidation();
  void analyze(const edm::Event& e, const edm::EventSetup&) override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;

 private:

  MonitorElement* theStrip_xy[2][3][2];
  MonitorElement* theStrip_phistrip[2][3][2];
  MonitorElement* theStrip[2][3][2];
  MonitorElement* theStrip_bx[2][3][2];
  MonitorElement* theStrip_zr[2][3][2];
  std::map< UInt_t , MonitorElement* > theStrip_xy_ch;
  //std::map< UInt_t , MonitorElement* > theStrip_st_dphi;
  //std::map< UInt_t , MonitorElement* > theStrip_phiz_st_ch;
  
  //MonitorElement* theSpecific_phiz[4];
};

#endif
