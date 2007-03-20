#ifndef HepPDTAAnalyzer_H
#define HepPDTAAnalyzer_H

// -*- C++ -*-
//
// Package:    HepPDTAnalyzer
// Class:      HepPDTAnalyzer
// 
/**\class HepPDTAnalyzer HepPDTAnalyzer.cc test/HepPDTAnalyzer/src/HepPDTAnalyzer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Filip Moortgat
//         Created:  Wed Jul 19 14:41:13 CEST 2006
// $Id: HepPDTAnalyzer.h,v 1.1 2006/07/21 12:25:04 fmoortga Exp $
//
//

#include <memory>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace edm {
  class ParameterSet;
}

class HepPDTAnalyzer : public edm::EDAnalyzer {
public:
  explicit HepPDTAnalyzer( const edm::ParameterSet & );
  ~HepPDTAnalyzer();
  
  
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
private:
  std::string particleName_;
};

#endif
