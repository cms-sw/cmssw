// -*- C++ -*-
//
// Class:      TestMix
// 
/**\class TestMix

 Description: test of Mixing Module

*/
//
// Original Author:  Ursula Berthon
//         Created:  Fri Sep 23 11:38:38 CEST 2005
// $Id: TestMix.h,v 1.2 2013/03/01 00:13:36 wmtan Exp $
//
//


// system include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <vector>
#include <string>

namespace edm
{

  //
  // class declaration
  //

  class TestMix : public edm::EDAnalyzer {
  public:
    explicit TestMix(const edm::ParameterSet&);
    ~TestMix();


    virtual void analyze(const edm::Event&, const edm::EventSetup&) override;

  private:
    int level_;
    std::vector<std::string> track_containers_;
    std::vector<std::string> track_containers2_;
  };
}//edm
