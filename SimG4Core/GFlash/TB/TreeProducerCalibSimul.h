#ifndef TREEPRODUCERCALIBSIMUL_H
#define TREEPRODUCERCALIBSIMUL_H

// system include files
#include <memory>

// framework
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"

// for reconstruction
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "TBDataFormats/EcalTBObjects/interface/EcalTBHodoscopeRecInfo.h"
#include "TBDataFormats/EcalTBObjects/interface/EcalTBTDCRecInfo.h"
#include "TBDataFormats/EcalTBObjects/interface/EcalTBEventHeader.h"

// geometry
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

// my include files
#include "SimG4Core/GFlash/TB/TreeMatrixCalib.h"


// root includes
#include "TROOT.h"
#include "TSystem.h"
#include "TFile.h"
#include "TTree.h"
#include "TF1.h"
#include "TH1.h"
#include "TH2.h"
#include "TGraph.h"
#include "TStyle.h"
#include "TCanvas.h"
#include "TSelector.h"
#include "TApplication.h"

// c++ includes
#include <string>
#include <stdio.h>
#include <sstream>
#include <iostream>
#include <unistd.h>
#include <fstream>
#include <math.h>
#include <stdexcept>


class TreeProducerCalibSimul : public edm::EDAnalyzer {
   public:
      explicit TreeProducerCalibSimul(const edm::ParameterSet&);
      ~TreeProducerCalibSimul();

      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void beginJob();
      virtual void endJob();
 private:

      
      std::string rootfile_;
      std::string txtfile_;
      std::string EBRecHitCollection_;
      std::string RecHitProducer_;
      std::string hodoRecInfoCollection_;
      std::string hodoRecInfoProducer_;
      std::string tdcRecInfoCollection_;
      std::string tdcRecInfoProducer_;
      std::string eventHeaderCollection_;
      std::string eventHeaderProducer_;
      double posCluster_;

      TreeMatrixCalib* myTree;

      int xtalInBeam;
      int tot_events;
      int tot_events_ok;
      int noHits;
      int noHodo;
      int noTdc;
      int noHeader;
};



#endif
