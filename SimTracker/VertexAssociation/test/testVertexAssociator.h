#ifndef testVertexAssociator_h
#define testVertexAssociator_h

#include <memory>

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "TH1F.h"
#include "TFile.h"

#include <iostream>
#include <string>
#include <map>
#include <set>

class TrackAssociatorBase;
class VertexAssociatorBase;

class testVertexAssociator : public edm::EDAnalyzer {

 public:
  testVertexAssociator(const edm::ParameterSet& conf);
  virtual ~testVertexAssociator();
  virtual void beginJob( const edm::EventSetup& );
  virtual void endJob();
  virtual void analyze(const edm::Event&, const edm::EventSetup&);

 private:
  TrackAssociatorBase*  associatorByChi2;
  TrackAssociatorBase*  associatorByHits;
  VertexAssociatorBase* associatorByTracks;
  TFile* rootFile;
  TH1F*  xMiss;
  TH1F*  yMiss;
  TH1F*  zMiss;
  TH1F*  rMiss;

  TH1F*  zVert;
  TH1F*  zTrue;
  TH1F*  nReco;
  TH1F*  nTrue;
};

#endif
