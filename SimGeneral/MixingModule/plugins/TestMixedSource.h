// -*- C++ -*-
//
// Package:    TestMixedSource
// Class:      TestMixedSource
// 
/**\class TestMixedSource TestMixedSource.cc TestMixed/TestMixedSource/src/TestMixedSource.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Emilia Lubenova Becheva
//         Created:  Wed May 20 16:46:58 CEST 2009
// $Id$
//
//

#ifndef TestMixedSource_h
#define TestMixedSource_h

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <iostream>
#include <fstream>

//
// class decleration
//
namespace edm
{
class TestMixedSource : public edm::EDAnalyzer {
   public:
      explicit TestMixedSource(const edm::ParameterSet&);
      ~TestMixedSource();


   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      // ----------member data ---------------------------
      ofstream outputFile;
};
}//edm
#endif
