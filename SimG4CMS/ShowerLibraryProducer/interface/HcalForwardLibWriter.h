#ifndef SimG4CMS_ShowerLibraryProducer_HcalForwardLibWriter_h
#define SimG4CMS_ShowerLibraryProducer_HcalForwardLibWriter_h

// -*- C++ -*-
//
// Package:    HcalForwardLibWriter
// Class:      HcalForwardLibWriter
// 
/**\class HcalForwardLibWriter HcalForwardLibWriter.h SimG4CMS/ShowerLibraryProducer/interface/HcalForwardLibWriter.h

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Taylan Yetkin,510 1-004,+41227672815,
//         Created:  Thu Feb  9 13:02:38 CET 2012
// $Id: HcalForwardLibWriter.h,v 1.5 2013/05/25 17:03:41 chrjones Exp $
//
//


// system include files
#include <memory>
#include <string>
#include <fstream>
#include <utility>
#include <vector>


// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "SimDataFormats/CaloHit/interface/HFShowerPhoton.h"
#include "SimDataFormats/CaloHit/interface/HFShowerLibraryEventInfo.h"

#include "TFile.h"
#include "TTree.h"

//
// class declaration
//

class HcalForwardLibWriter : public edm::EDProducer {
   public:
      
      struct FileHandle{
          std::string name;
          std::string id;
          int momentum;
      };
      
      explicit HcalForwardLibWriter(const edm::ParameterSet&);
      ~HcalForwardLibWriter();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:
      virtual void produce(edm::Event&, const edm::EventSetup&) override;
      
      void readUserData();

      // ----------member data ---------------------------
      std::string fDataFile;
      std::vector<FileHandle> fFileHandle;
      TFile* fFile;
      TTree* fTree;
};
#endif
