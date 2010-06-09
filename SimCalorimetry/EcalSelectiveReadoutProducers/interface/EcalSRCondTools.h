#ifndef SRCONDACCESS_H
#define SRCONDACCESS_H

/*
 * $Id$
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

/**
 */
class EcalSRCondTools : public edm::EDAnalyzer {
  //ctors
public:
  explicit EcalSRCondTools(const edm::ParameterSet&);
  ~EcalSRCondTools();


  virtual void analyze(const edm::Event&, const edm::EventSetup&);

  void analyzeEB(const edm::Event&, const edm::EventSetup&) const;
  void analyzeEE(const edm::Event&, const edm::EventSetup&) const;
  
  //methods
public:
private:

  //fields
private:

  edm::ParameterSet ps_;
};

#endif //SRCONDACCESS_H not defined
