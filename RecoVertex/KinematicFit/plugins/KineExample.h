// -*- C++ -*-
//
// Package:    KineExample
// Class:      KineExample
// 
/**\class KineExample KineExample.cc RecoVertex/KineExample/src/KineExample.cc

 Description: steers tracker primary vertex reconstruction and storage

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Pascal Vanlaer
//         Created:  Tue Feb 28 11:06:34 CET 2006
// $Id: KineExample.h,v 1.2 2009/12/14 22:24:18 wmtan Exp $
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
// #include "RecoVertex/KalmanVertexFit/test/SimpleVertexTree.h"
#include "SimTracker/TrackAssociation/interface/TrackAssociatorByChi2.h"
#include "RecoVertex/KinematicFitPrimitives/interface/RefCountedKinematicVertex.h"
#include "RecoVertex/KinematicFitPrimitives/interface/RefCountedKinematicParticle.h"
#include "RecoVertex/KinematicFitPrimitives/interface/RefCountedKinematicTree.h"
// #include "RecoVertex/KinematicFitPrimitives/interface/"
#include <TFile.h>

  /**
   * This is a very simple test analyzer mean to test the KalmanVertexFitter
   */

class KineExample : public edm::EDAnalyzer {
public:
  explicit KineExample(const edm::ParameterSet&);
  ~KineExample();
  
  virtual void analyze(const edm::Event&, const edm::EventSetup&);

  virtual void beginRun(const edm::EventSetup&);
  virtual void endJob();

private:

  void printout(const RefCountedKinematicVertex& myVertex) const;
  void printout(const RefCountedKinematicParticle& myParticle) const;
  void printout(const RefCountedKinematicTree& myTree) const;

  TrackingVertex getSimVertex(const edm::Event& iEvent) const;

  edm::ParameterSet theConfig;
  edm::ParameterSet kvfPSet;
  TrackAssociatorByChi2 * associatorForParamAtPca;
//   SimpleVertexTree *tree;
//   TFile*  rootFile_;

  std::string outputFile_; // output file
  std::string trackLabel_; // label of track producer
};
