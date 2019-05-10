#ifndef HepPDTAAnalyzer_H
#define HepPDTAAnalyzer_H

// -*- C++ -*-
//
// Package:    HepPDTAnalyzer
// Class:      HepPDTAnalyzer
//
/**\class HepPDTAnalyzer HepPDTAnalyzer.cc
 test/HepPDTAnalyzer/src/HepPDTAnalyzer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Filip Moortgat
//         Created:  Wed Jul 19 14:41:13 CEST 2006
//
//

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include <memory>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"

namespace edm {
  class ParameterSet;
}

class HepPDTAnalyzer : public edm::EDAnalyzer {
public:
  explicit HepPDTAnalyzer(const edm::ParameterSet &);
  ~HepPDTAnalyzer() override;

  void analyze(const edm::Event &, const edm::EventSetup &) override;

  void printInfo(const ParticleData *&part);

  void printBanner();

private:
  std::string particleName_;
};

#endif
