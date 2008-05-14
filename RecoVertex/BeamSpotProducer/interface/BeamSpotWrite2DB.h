#ifndef BeamSpotProducer_BeamSpotWrite2DB_h
#define BeamSpotProducer_BeamSpotWrite2DB_h

/**_________________________________________________________________
   class:   BeamSpotWrite2DB.h
   package: RecoVertex/BeamSpotProducer
   


 author: Francisco Yumiceva, Fermilab (yumiceva@fnal.gov)

 version $Id: BeamSpotWrite2DB.h,v 1.4 2007/08/15 21:51:13 yumiceva Exp $

________________________________________________________________**/


// C++ standard
#include <string>
// CMS
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoVertex/BeamSpotProducer/interface/BSTrkParameters.h"

// ROOT
#include "TFile.h"
#include "TTree.h"

#include<fstream>

class BeamSpotWrite2DB : public edm::EDAnalyzer {
 public:
  explicit BeamSpotWrite2DB(const edm::ParameterSet&);
  ~BeamSpotWrite2DB();

 private:
  virtual void beginJob(const edm::EventSetup&) ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

  
  std::ifstream fasciiFile;
  std::string fasciiFileName;

  
};

#endif
