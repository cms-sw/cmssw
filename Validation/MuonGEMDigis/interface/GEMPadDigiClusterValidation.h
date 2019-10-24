#ifndef Validation_MuonGEMDigis_GEMPadDigiClusterValidation_H
#define Validation_MuonGEMDigis_GEMPadDigiClusterValidation_H

#include "Validation/MuonGEMHits/interface/GEMBaseValidation.h"
#include "DataFormats/GEMDigi/interface/GEMPadDigiClusterCollection.h"

class GEMPadDigiClusterValidation : public GEMBaseValidation
{
public:
  explicit GEMPadDigiClusterValidation( const edm::ParameterSet& );
  ~GEMPadDigiClusterValidation() override;
  void analyze(const edm::Event& e, const edm::EventSetup&) override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
 private:
  // Detail plots
  MonitorElement* theGEMCluster_xy[2][3][2];
  MonitorElement* theGEMCluster_phipad[2][3][2];
  MonitorElement* theGEMCluster[2][3][2];
  MonitorElement* theGEMCluster_bx[2][3][2];
  MonitorElement* theGEMCluster_zr[2][3][2];
	std::unordered_map< UInt_t , MonitorElement* > theGEMCluster_xy_ch;

  // Simple plots
  std::unordered_map< UInt_t , MonitorElement* > theCluster_dcEta;
  std::unordered_map< UInt_t , MonitorElement* > theCluster_simple_zr;

  edm::EDGetToken InputTagToken_;
  int nBinXY_;
  bool detailPlot_;
};

#endif
