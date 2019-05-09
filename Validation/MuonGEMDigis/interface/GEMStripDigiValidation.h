#ifndef GEMStripDigiValidation_H
#define GEMStripDigiValidation_H

#include "Validation/MuonGEMHits/interface/GEMBaseValidation.h"

//#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"

class GEMStripDigiValidation : public GEMBaseValidation {
public:
  explicit GEMStripDigiValidation(const edm::ParameterSet &);
  ~GEMStripDigiValidation() override;
  void analyze(const edm::Event &e, const edm::EventSetup &) override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;

private:
  // Detail plots
  MonitorElement *theStrip_xy[2][3][2];
  MonitorElement *theStrip_phistrip[2][3][2];
  MonitorElement *theStrip[2][3][2];
  MonitorElement *theStrip_bx[2][3][2];
  MonitorElement *theStrip_zr[2][3][2];
  std::unordered_map<UInt_t, MonitorElement *> theStrip_xy_ch;

  // Simple plots
  std::unordered_map<UInt_t, MonitorElement *> theStrip_dcEta;
  std::unordered_map<UInt_t, MonitorElement *> theStrip_simple_zr;

  edm::EDGetToken InputTagToken_;
  int nBinXY_;
  bool detailPlot_;
};

#endif
