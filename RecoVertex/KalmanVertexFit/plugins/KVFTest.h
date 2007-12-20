// -*- C++ -*-
//
// Package:    KVFTest
// Class:      KVFTest
// 
/**\class KVFTest KVFTest.cc RecoVertex/KVFTest/src/KVFTest.cc

 Description: steers tracker primary vertex reconstruction and storage

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Pascal Vanlaer
//         Created:  Tue Feb 28 11:06:34 CET 2006
// $Id: KVFTest.h,v 1.1 2007/04/24 07:43:15 tboccali Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertexContainer.h"
#include "RecoVertex/KalmanVertexFit/interface/SimpleVertexTree.h"
#include "SimTracker/TrackAssociation/interface/TrackAssociatorByChi2.h"
#include <TFile.h>

  /**
   * This is a very simple test analyzer mean to test the KalmanVertexFitter
   */

class KVFTest : public edm::EDAnalyzer {
public:
  explicit KVFTest(const edm::ParameterSet&);
  ~KVFTest();
  
  virtual void analyze(const edm::Event&, const edm::EventSetup&);

  virtual void beginJob(edm::EventSetup const&);
  virtual void endJob();

private:

  TrackingVertex getSimVertex(const edm::Event& iEvent) const;

  edm::ParameterSet theConfig;
  edm::ParameterSet kvfPSet;
  TrackAssociatorByChi2 * associatorForParamAtPca;
  SimpleVertexTree *tree;
  TFile*  rootFile_;

  std::string outputFile_; // output file
  std::string trackLabel_; // label of track producer
};
