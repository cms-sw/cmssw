#ifndef HCALDIGISVALIDATION_H
#define HCALDIGISVALIDATION_H

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "Geometry/HcalCommonData/interface/HcalDDDRecConstants.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"
#include "Geometry/HcalTowerAlgo/interface/HcalTrigTowerGeometry.h"

#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/Records/interface/HcalGeometryRecord.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"

#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"

#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
#include "CalibFormats/HcalObjects/interface/HcalCoderDb.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"
#include "CondFormats/HcalObjects/interface/HcalQIEShape.h"

#include "DataFormats/HcalDigi/interface/HBHEDataFrame.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"

/*TP Code*/
#include "CalibFormats/CaloTPG/interface/CaloTPGTranscoder.h"
#include "CalibFormats/CaloTPG/interface/CaloTPGRecord.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
/*~TP Code*/

#include <map>
#include <vector>
#include <utility>
#include <ostream>
#include <string>
#include <algorithm>
#include <cmath>
#include <iostream>

class HcalDigisValidation : public DQMEDAnalyzer {
public:
  explicit HcalDigisValidation(const edm::ParameterSet &);

  ~HcalDigisValidation() override;

  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void dqmBeginRun(const edm::Run &run, const edm::EventSetup &c) override;

private:
  struct HistLim {
    HistLim(int nbin, double mini, double maxi) : n(nbin), min(mini), max(maxi) {}
    int n;
    double min;
    double max;
  };

  void analyze(const edm::Event &, const edm::EventSetup &) override;

  std::map<std::string, MonitorElement *> *msm_;

  void book1D(DQMStore::IBooker &ib, std::string name, int n, double min, double max);

  void book1D(DQMStore::IBooker &ib, std::string name, const HistLim &limX);

  void fill1D(std::string name, double X, double weight = 1);

  void book2D(DQMStore::IBooker &ib, std::string name, const HistLim &limX, const HistLim &limY);

  void fill2D(std::string name, double X, double Y, double weight = 1);

  void bookPf(DQMStore::IBooker &ib, std::string name, const HistLim &limX, const HistLim &limY);

  void bookPf(DQMStore::IBooker &ib, std::string name, const HistLim &limX, const HistLim &limY, const char *option);

  void fillPf(std::string name, double X, double Y);

  MonitorElement *monitor(std::string name);

  void booking(DQMStore::IBooker &ib, std::string subdetopt, int bnoise, int bmc);

  std::string str(int x);

  template <class Digi>
  void reco(const edm::Event &iEvent,
            const edm::EventSetup &iSetup,
            const edm::EDGetTokenT<edm::SortedCollection<Digi> > &tok);
  template <class dataFrameType>
  void reco(const edm::Event &iEvent,
            const edm::EventSetup &iSetup,
            const edm::EDGetTokenT<HcalDataFrameContainer<dataFrameType> > &tok);

  std::string outputFile_;
  std::string subdet_;
  std::string zside_;
  std::string dirName_;
  //    std::string inputLabel_;
  edm::InputTag inputTag_;
  edm::InputTag QIE10inputTag_;
  edm::InputTag QIE11inputTag_;
  edm::InputTag emulTPsTag_;
  edm::InputTag dataTPsTag_;
  std::string mode_;
  std::string mc_;
  int noise_;
  bool testNumber_;
  bool hep17_;
  bool HEPhase1_;
  bool HBPhase1_;
  bool Plot_TP_ver_;

  edm::EDGetTokenT<edm::PCaloHitContainer> tok_mc_;
  edm::EDGetTokenT<HBHEDigiCollection> tok_hbhe_;
  edm::EDGetTokenT<HODigiCollection> tok_ho_;
  edm::EDGetTokenT<HFDigiCollection> tok_hf_;
  edm::EDGetTokenT<HcalTrigPrimDigiCollection> tok_emulTPs_;
  edm::EDGetTokenT<HcalTrigPrimDigiCollection> tok_dataTPs_;

  edm::EDGetTokenT<QIE10DigiCollection> tok_qie10_hf_;
  edm::EDGetTokenT<QIE11DigiCollection> tok_qie11_hbhe_;

  edm::ESGetToken<HcalDDDRecConstants, HcalRecNumberingRecord> tok_HRNDC_;
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> tok_Geom_;
  edm::ESGetToken<CaloTPGTranscoder, CaloTPGRecord> tok_Decoder_;
  edm::ESGetToken<HcalTrigTowerGeometry, CaloGeometryRecord> tok_TPGeom_;
  edm::ESGetToken<HcalTopology, HcalRecNumberingRecord> tok_Topo_;
  edm::ESGetToken<HcalDbService, HcalDbRecord> tok_Cond_;

  const HcalDbService *conditions_;
  const HcalDDDRecConstants *hcons_;
  const HcalTopology *htopo_;

  int nevent1;
  int nevent2;
  int nevent3;
  int nevent4;
  int nevtot;

  int maxDepth_[5];   // 0:any, 1:HB, 2:HE, 3:HF
  int nChannels_[5];  // 0:any, 1:HB, 2:HE,

  bool skipDataTPs;
};

#endif
