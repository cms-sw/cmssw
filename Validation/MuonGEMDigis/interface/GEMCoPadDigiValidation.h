#ifndef GEMCoPadDigiValidation_H
#define GEMCoPadDigiValidation_H

#include "DataFormats/GEMDigi/interface/GEMCoPadDigiCollection.h"
#include "Validation/MuonGEMHits/interface/GEMBaseValidation.h"

class GEMCoPadDigiValidation : public GEMBaseValidation {
public:
  explicit GEMCoPadDigiValidation(const edm::ParameterSet &);
  ~GEMCoPadDigiValidation() override;
  void analyze(const edm::Event &e, const edm::EventSetup &) override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;

private:
  MonitorElement *theCSCCoPad_xy[2][3];
  MonitorElement *theCSCCoPad_phipad[2][3];
  MonitorElement *theCSCCoPad[2][3];
  MonitorElement *theCSCCoPad_bx[2][3];
  MonitorElement *theCSCCoPad_zr[2][3];
  std::unordered_map<UInt_t, MonitorElement *> theCSCCoPad_xy_ch;

  // Simple plots
  std::unordered_map<UInt_t, MonitorElement *> theCoPad_dcEta;
  std::unordered_map<UInt_t, MonitorElement *> theCoPad_simple_zr;
  int minBXGEM_, maxBXGEM_;
  edm::EDGetToken InputTagToken_;
  int nBinXY_;
  bool detailPlot_;
};

#endif
