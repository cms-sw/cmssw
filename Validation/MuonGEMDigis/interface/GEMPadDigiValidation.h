#ifndef Validation_MuonGEMDigis_GEMPadDigiValidation_H
#define Validation_MuonGEMDigis_GEMPadDigiValidation_H

#include "Validation/MuonGEMHits/interface/GEMBaseValidation.h"
#include "DataFormats/GEMDigi/interface/GEMPadDigiCollection.h"

class GEMPadDigiValidation : public GEMBaseValidation
{
public:
  explicit GEMPadDigiValidation( const edm::ParameterSet& );
  ~GEMPadDigiValidation() override;
  void analyze(const edm::Event& e, const edm::EventSetup&) override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
 private:
  // Detail plots
  MonitorElement* theGEMPad_xy[2][3][2];
  MonitorElement* theGEMPad_phipad[2][3][2];
  MonitorElement* theGEMPad[2][3][2];
  MonitorElement* theGEMPad_bx[2][3][2];
  MonitorElement* theGEMPad_zr[2][3][2];
	std::unordered_map< UInt_t , MonitorElement* > theGEMPad_xy_ch;

  // Simple plots
  std::unordered_map< UInt_t , MonitorElement* > thePad_dcEta;
  std::unordered_map< UInt_t , MonitorElement* > thePad_simple_zr;

  edm::EDGetToken InputTagToken_;
  int nBinXY_;
  bool detailPlot_;
};

#endif
