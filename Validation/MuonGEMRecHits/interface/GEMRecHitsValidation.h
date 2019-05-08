#ifndef GEMRecHitsValidation_H
#define GEMRecHitsValidation_H

#include "Validation/MuonGEMHits/interface/GEMBaseValidation.h"

#include <DataFormats/GEMRecHit/interface/GEMRecHit.h>
#include <DataFormats/GEMRecHit/interface/GEMRecHitCollection.h>

class GEMRecHitsValidation : public GEMBaseValidation {
public:
  explicit GEMRecHitsValidation(const edm::ParameterSet &);
  ~GEMRecHitsValidation() override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void analyze(const edm::Event &e, const edm::EventSetup &) override;
  MonitorElement *BookHist1D(DQMStore::IBooker &,
                             const char *name,
                             const char *label,
                             unsigned int region_num,
                             unsigned int station_num,
                             unsigned int layer_num,
                             const unsigned int Nbin,
                             const Float_t xMin,
                             const Float_t xMax);
  MonitorElement *BookHist1D(DQMStore::IBooker &,
                             const char *name,
                             const char *label,
                             unsigned int region_num,
                             const unsigned int Nbin,
                             const Float_t xMin,
                             const Float_t xMax);

private:
  // Detaile Plots
  MonitorElement *gem_rh_xy[2][3][2];
  MonitorElement *gem_rh_zr[2][3][2];
  MonitorElement *gem_cls[2][3][2];
  MonitorElement *gem_pullX[2][3][2];
  MonitorElement *gem_pullY[2][3][2];

  // Simple Plots
  MonitorElement *gem_cls_tot;
  std::unordered_map<UInt_t, MonitorElement *> recHits_dcEta;
  std::unordered_map<UInt_t, MonitorElement *> recHits_simple_zr;
  MonitorElement *gem_region_pullX[2];
  MonitorElement *gem_region_pullY[2];

  edm::EDGetToken InputTagToken_, InputTagToken_RH;
  int nBinXY_;
  bool detailPlot_;
};

#endif
