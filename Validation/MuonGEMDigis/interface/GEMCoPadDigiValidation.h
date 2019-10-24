#ifndef Validation_MuonGEMDigis_GEMCoPadDigiValidation_H
#define Validation_MuonGEMDigis_GEMCoPadDigiValidation_H

#include "Validation/MuonGEMHits/interface/GEMBaseValidation.h"
#include "DataFormats/GEMDigi/interface/GEMCoPadDigiCollection.h"

class GEMCoPadDigiValidation : public GEMBaseValidation
{
public:
  explicit GEMCoPadDigiValidation( const edm::ParameterSet& );
  ~GEMCoPadDigiValidation() override;
  void analyze(const edm::Event& e, const edm::EventSetup&) override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
 private:

  MonitorElement* theGEMCoPad_xy[2][3];
  MonitorElement* theGEMCoPad_phipad[2][3];
  MonitorElement* theGEMCoPad[2][3];
  MonitorElement* theGEMCoPad_bx[2][3];
  MonitorElement* theGEMCoPad_zr[2][3];
	std::unordered_map< UInt_t , MonitorElement* > theGEMCoPad_xy_ch;


  // Simple plots
  std::unordered_map< UInt_t , MonitorElement* > theCoPad_dcEta;
  std::unordered_map< UInt_t , MonitorElement* > theCoPad_simple_zr;
  int minBXGEM_, maxBXGEM_;
  edm::EDGetToken InputTagToken_;
  int nBinXY_;
  bool detailPlot_;
};

#endif
