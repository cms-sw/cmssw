#ifndef SimG4CMS_HcalTestHistoManager_H
#define SimG4CMS_HcalTestHistoManager_H
///////////////////////////////////////////////////////////////////////////////
// File: HcalTestHistoManager.h
// Histogram managing class for analysis in HcalTest
///////////////////////////////////////////////////////////////////////////////

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "SimDataFormats/CaloTest/interface/HcalTestHistoClass.h"

// root objects
#include "TROOT.h"
#include "TSystem.h"
#include "TFile.h"
#include "TDirectory.h"
#include "TTree.h"

#include <memory>
#include <string>

class HcalTestHistoManager {
public:
  HcalTestHistoManager(const std::string&);
  virtual ~HcalTestHistoManager();

  void fillTree(HcalTestHistoClass* histos);

private:
  edm::Service<TFileService> fs_;
  TTree* tree_;
  std::unique_ptr<HcalTestHistoClass> h_;
  int kount_;
};

#endif
