#ifndef GEMPadDigiValidation_H
#define GEMPadDigiValidation_H

#include "DataFormats/GEMDigi/interface/GEMPadDigiCollection.h"
#include "Validation/MuonGEMHits/interface/GEMBaseValidation.h"

//#include "DataFormats/Common/interface/Handle.h"
class GEMPadDigiValidation : public GEMBaseValidation {
public:
  explicit GEMPadDigiValidation(const edm::ParameterSet &);
  ~GEMPadDigiValidation() override;
  void analyze(const edm::Event &e, const edm::EventSetup &) override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;

private:
  // Detail plots
  MonitorElement *theCSCPad_xy[2][3][2];
  MonitorElement *theCSCPad_phipad[2][3][2];
  MonitorElement *theCSCPad[2][3][2];
  MonitorElement *theCSCPad_bx[2][3][2];
  MonitorElement *theCSCPad_zr[2][3][2];
  std::unordered_map<UInt_t, MonitorElement *> theCSCPad_xy_ch;

  // Simple plots
  std::unordered_map<UInt_t, MonitorElement *> thePad_dcEta;
  std::unordered_map<UInt_t, MonitorElement *> thePad_simple_zr;

  edm::EDGetToken InputTagToken_;
  int nBinXY_;
  bool detailPlot_;
};

#endif
