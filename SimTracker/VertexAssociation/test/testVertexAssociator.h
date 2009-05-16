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

class testVertexAssociator : public edm::EDAnalyzer
{

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

    TH1F*  sr_xMiss;
    TH1F*  sr_yMiss;
    TH1F*  sr_zMiss;
    TH1F*  sr_rMiss;

    TH1F*  sr_zVert;
    TH1F*  sr_zTrue;
    TH1F*  sr_nReco;
    TH1F*  sr_nTrue;
    TH1F*  sr_qual;
    TH1F*  rs_qual;
};

#endif
