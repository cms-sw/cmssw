#ifndef GEMBaseValidation_H
#define GEMBaseValidation_H

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <unordered_map>

class GEMBaseValidation : public DQMEDAnalyzer {
public:
  explicit GEMBaseValidation(const edm::ParameterSet& ps);
  ~GEMBaseValidation() override;
  void analyze(const edm::Event& e, const edm::EventSetup&) override = 0;
  MonitorElement* BookHistZR(DQMStore::IBooker&,
                             const char* name,
                             const char* label,
                             unsigned int region_num,
                             unsigned int station_num,
                             unsigned int layer_num = 99);
  MonitorElement* BookHistXY(DQMStore::IBooker&,
                             const char* name,
                             const char* label,
                             unsigned int region_num,
                             unsigned int station_num,
                             unsigned int layer_num = 99);
  std::string getSuffixName(int region, int station, int layer);
  std::string getSuffixName(int region, int station);
  std::string getSuffixName(int region);

  std::string getSuffixTitle(int region, int station, int layer);
  std::string getSuffixTitle(int region, int station);
  std::string getSuffixTitle(int region);

  std::string getStationLabel(int i);
  const GEMGeometry* initGeometry(const edm::EventSetup&);

  MonitorElement* getSimpleZR(DQMStore::IBooker&, TString, TString);
  MonitorElement* getDCEta(DQMStore::IBooker&, const GEMStation*, TString, TString);

  unsigned int nRegion() { return nregion; }
  unsigned int nStation() { return nstation; }
  unsigned int nStationForLabel() { return nstationForLabel; }
  unsigned int nPart() { return npart; }

  void setNStationForLabel(unsigned int number) { nstationForLabel = number; }

protected:
  int nBinXY_;
  std::vector<double> nBinZR_;
  std::vector<double> RangeZR_;

private:
  std::vector<std::string> regionLabel;
  std::vector<std::string> layerLabel;
  edm::EDGetToken InputTagToken_;
  unsigned int nregion, nstation, nstationForLabel, npart;
  bool detailPlot_;
};

#endif
