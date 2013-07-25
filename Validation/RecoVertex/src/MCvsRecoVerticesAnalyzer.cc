// -*- C++ -*-
//
// Package:    MCvsRecoVerticesAnalyzer
// Class:      MCvsRecoVerticesAnalyzer
// 
/**\class MCvsRecoVerticesAnalyzer MCvsRecoVerticesAnalyzer.cc TrackingPFG/PileUp/src/MCvsRecoVerticesAnalyzer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Andrea Venturi
//         Created:  Thu Dec 16 16:32:56 CEST 2010
// $Id: MCvsRecoVerticesAnalyzer.cc,v 1.4 2011/11/26 00:51:42 venturia Exp $
//
//


// system include files
#include <memory>
#include <numeric>

#include <vector>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "FWCore/Utilities/interface/InputTag.h"

#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "TH1F.h"
#include "TH2F.h"
#include "TProfile.h"

//
// class decleration
//


class MCvsRecoVerticesAnalyzer : public edm::EDAnalyzer {
public:
  explicit MCvsRecoVerticesAnalyzer(const edm::ParameterSet&);
  ~MCvsRecoVerticesAnalyzer();
  
private:
  virtual void beginJob() ;
  virtual void beginRun(const edm::Run&, const edm::EventSetup&);
  virtual void endRun(const edm::Run&, const edm::EventSetup&);
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  
      // ----------member data ---------------------------

  

  edm::InputTag m_pileupcollection;
  edm::InputTag m_mctruthcollection;
  edm::InputTag m_pvcollection;
  const bool m_useweight;
  edm::InputTag m_weight;
  const bool m_useVisibleVertices;
  const edm::ParameterSet m_histoParameters;

  TH2F* m_hrecovsmcnvtx2d;
  TProfile* m_hrecovsmcnvtxprof;
  TProfile* m_hrecovsmcnvtxweightedprof;
  TH2F* m_hrecovsmclumi2d;
  TProfile* m_hrecovsmclumiprof;
  TProfile* m_hrecovsmclumiweightedprof;
  TH1F* m_hdeltazfirst;
  TH1F* m_hdeltazclose;
  TH1F* m_hclosestvtx;
 
  TH2F* m_hdeltazfirstvsnpu;
  TH2F* m_hdeltazclosevsnpu;
  TH2F* m_hclosestvtxvsnpu;
 

};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
MCvsRecoVerticesAnalyzer::MCvsRecoVerticesAnalyzer(const edm::ParameterSet& iConfig):
  m_pileupcollection(iConfig.getParameter<edm::InputTag>("pileupSummaryCollection")),
  m_mctruthcollection(iConfig.getParameter<edm::InputTag>("mcTruthCollection")),
  m_pvcollection(iConfig.getParameter<edm::InputTag>("pvCollection")),
  m_useweight(iConfig.getParameter<bool>("useWeight")),
  m_weight(iConfig.getParameter<edm::InputTag>("weightProduct")),
  m_useVisibleVertices(iConfig.getParameter<bool>("useVisibleVertices")),
  m_histoParameters(iConfig.getUntrackedParameter<edm::ParameterSet>("histoParameters",edm::ParameterSet()))
{
   //now do what ever initialization is needed

  if(m_useVisibleVertices) edm::LogInfo("UseVisibleVertices") << "Only visible vertices will be used to compute Npileup";

  edm::Service<TFileService> tfserv;

  m_hrecovsmcnvtx2d = tfserv->make<TH2F>("recovsmcnvtx2d","Number of reco vertices vs pileup interactions",60,-0.5,59.5,60,-0.5,59.5);
  m_hrecovsmcnvtx2d->GetXaxis()->SetTitle("Pileup Interactions");  m_hrecovsmcnvtx2d->GetYaxis()->SetTitle("Reco Vertices");
  m_hrecovsmcnvtxprof = tfserv->make<TProfile>("recovsmcnvtxprof","Mean number of reco vs pileup vertices",60,-0.5,59.5);
  m_hrecovsmcnvtxprof->GetXaxis()->SetTitle("Pileup Interactions");  m_hrecovsmcnvtxprof->GetYaxis()->SetTitle("Reco Vertices");

  m_hrecovsmclumi2d = tfserv->make<TH2F>("recovsmclumi2d","Number of reco vertices vs ave pileup interactions",200,0.,50.,60,-0.5,59.5);
  m_hrecovsmclumi2d->GetXaxis()->SetTitle("Average Pileup Interactions");  m_hrecovsmclumi2d->GetYaxis()->SetTitle("Reco Vertices");
  m_hrecovsmclumiprof = tfserv->make<TProfile>("recovsmclumiprof","Mean number of reco vs ave pileup vertices",200,0.,50.);
  m_hrecovsmclumiprof->GetXaxis()->SetTitle("Average Pileup Interactions");  m_hrecovsmclumiprof->GetYaxis()->SetTitle("Reco Vertices");

  if(m_useweight) {
    m_hrecovsmcnvtxweightedprof = tfserv->make<TProfile>("recovsmcnvtxweightedprof","Mean number of reco vs pileup vertices (1-w) weight",60,-0.5,59.5);
    m_hrecovsmcnvtxweightedprof->GetXaxis()->SetTitle("Pileup Interactions");  m_hrecovsmcnvtxweightedprof->GetYaxis()->SetTitle("Reco Vertices (1-w)");

    m_hrecovsmclumiweightedprof = tfserv->make<TProfile>("recovsmclumiweightedprof","Mean number of reco vs ave pileup vertices (1-w) weight",
							 200,0.,50.);
    m_hrecovsmclumiweightedprof->GetXaxis()->SetTitle("Average Pileup Interactions");  
    m_hrecovsmclumiweightedprof->GetYaxis()->SetTitle("Reco Vertices (1-w)");
  }

  m_hdeltazfirst = tfserv->make<TH1F>("deltazfirst","Reco-MC vertex z position (first vertex)",
				      m_histoParameters.getUntrackedParameter<unsigned int>("zBins",1000),
				      m_histoParameters.getUntrackedParameter<double>("zMin",-1.),
				      m_histoParameters.getUntrackedParameter<double>("zMax",1.));
  m_hdeltazfirst->GetXaxis()->SetTitle("#Delta z (cm)");  m_hdeltazfirst->GetYaxis()->SetTitle("Events");

  m_hdeltazclose = tfserv->make<TH1F>("deltazclose","Reco-MC vertex z position (closest vertex)",
				      m_histoParameters.getUntrackedParameter<unsigned int>("zBins",1000),
				      m_histoParameters.getUntrackedParameter<double>("zMin",-1.),
				      m_histoParameters.getUntrackedParameter<double>("zMax",1.));
  m_hdeltazclose->GetXaxis()->SetTitle("#Delta z (cm)");  m_hdeltazclose->GetYaxis()->SetTitle("Events");

  m_hclosestvtx = tfserv->make<TH1F>("closestvtx","Closest reco vtx ID",30,-0.5,29.5);
  m_hclosestvtx->GetXaxis()->SetTitle("Vtx ID");  m_hclosestvtx->GetYaxis()->SetTitle("Events");

  m_hdeltazfirstvsnpu = tfserv->make<TH2F>("deltazfirstvsnpu","Reco-MC vertex z position (first vertex) vs Npileup",30,-0.5,29.5,
				      m_histoParameters.getUntrackedParameter<unsigned int>("zBins",1000),
				      m_histoParameters.getUntrackedParameter<double>("zMin",-1.),
				      m_histoParameters.getUntrackedParameter<double>("zMax",1.));
  m_hdeltazfirstvsnpu->GetXaxis()->SetTitle("pileup Interactions");  m_hdeltazfirstvsnpu->GetYaxis()->SetTitle("#Delta z (cm)");  

  m_hdeltazclosevsnpu = tfserv->make<TH2F>("deltazclosevsnpu","Reco-MC vertex z position (closest vertex) v Npileup",30,-0.5,29.5,
				      m_histoParameters.getUntrackedParameter<unsigned int>("zBins",1000),
				      m_histoParameters.getUntrackedParameter<double>("zMin",-1.),
				      m_histoParameters.getUntrackedParameter<double>("zMax",1.));
  m_hdeltazclosevsnpu->GetXaxis()->SetTitle("Pileup Interactions");  m_hdeltazclosevsnpu->GetYaxis()->SetTitle("#Delta z (cm)");

  m_hclosestvtxvsnpu = tfserv->make<TH2F>("closestvtxvsnpu","Closest reco vtx ID vs Npileup",30,-0.5,29.5,30,-0.5,29.5);
  m_hclosestvtxvsnpu->GetXaxis()->SetTitle("Pileup Interactions");  m_hclosestvtxvsnpu->GetYaxis()->SetTitle("Vtx ID");

}


MCvsRecoVerticesAnalyzer::~MCvsRecoVerticesAnalyzer()
{

   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
MCvsRecoVerticesAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  
  double weight = 1.;
  
  if(m_useweight) {
    Handle<double> weightprod;
    iEvent.getByLabel(m_weight,weightprod);
    
    weight = *weightprod;
    
  }
  
  Handle<std::vector<PileupSummaryInfo> > pileupinfos;
  iEvent.getByLabel(m_pileupcollection,pileupinfos);

  // look for the intime PileupSummaryInfo

  std::vector<PileupSummaryInfo>::const_iterator pileupinfo;

  for(pileupinfo = pileupinfos->begin(); pileupinfo != pileupinfos->end() ; ++pileupinfo) {

    if(pileupinfo->getBunchCrossing()==0) break;

  } 
  
  //
  
  Handle<reco::VertexCollection> pvcoll;
  iEvent.getByLabel(m_pvcollection,pvcoll);
  

   //

  if(pileupinfo->getBunchCrossing()!=0) {

    edm::LogError("NoInTimePileUpInfo") << "Cannot find the in-time pileup info " << pileupinfo->getBunchCrossing();

  }
  else {

    int npileup = pileupinfo->getPU_NumInteractions();

    if(m_useVisibleVertices) npileup = pileupinfo->getPU_zpositions().size();

    m_hrecovsmcnvtx2d->Fill(npileup,pvcoll->size(),weight);
    m_hrecovsmcnvtxprof->Fill(npileup,pvcoll->size(),weight);

    m_hrecovsmclumi2d->Fill(pileupinfo->getTrueNumInteractions(),pvcoll->size(),weight);
    m_hrecovsmclumiprof->Fill(pileupinfo->getTrueNumInteractions(),pvcoll->size(),weight);

    if(m_useweight) {
      m_hrecovsmcnvtxweightedprof->Fill(npileup,pvcoll->size(),1.-weight);
      m_hrecovsmclumiweightedprof->Fill(pileupinfo->getTrueNumInteractions(),pvcoll->size(),1.-weight);
    }
    //
    
    Handle< HepMCProduct > EvtHandle ;
    iEvent.getByLabel(m_mctruthcollection, EvtHandle ) ;
    
    const HepMC::GenEvent* Evt = EvtHandle->GetEvent();
    
    // compute the difference between the main interaction vertex z position and the first vertex of the collection
    
    if(pvcoll->size() !=0) {
      if(!(*pvcoll)[0].isFake()) {
	// get the first vertex
	if(Evt->vertices_begin() != Evt->vertices_end()) {
	  m_hdeltazfirst->Fill((*pvcoll)[0].z()-(*Evt->vertices_begin())->point3d().z()/10.,weight);
	  m_hdeltazfirstvsnpu->Fill(npileup,(*pvcoll)[0].z()-(*Evt->vertices_begin())->point3d().z()/10.,weight);
	}
      }
    }
    
    // compute the difference between the main interaction vertex z position and the closest reco vertex  
    
    double minabsdist = -1.;
    double mindist = -999.;
    int closestvtx = -1;
    
    for(unsigned int ivtx = 0 ; ivtx < pvcoll->size() ; ++ivtx) {
      
      if(closestvtx < 0 || minabsdist > std::abs((*pvcoll)[ivtx].z()-(*Evt->vertices_begin())->point3d().z()/10.)) {
	mindist = (*pvcoll)[ivtx].z()-(*Evt->vertices_begin())->point3d().z()/10.;
	closestvtx = ivtx;
	minabsdist = std::abs(mindist);
      }
      
    }
    if(closestvtx >= 0) {
      m_hdeltazclose->Fill(mindist,weight);
      m_hdeltazclosevsnpu->Fill(npileup,mindist,weight);
      m_hclosestvtx->Fill(closestvtx,weight);
      m_hclosestvtxvsnpu->Fill(npileup,closestvtx,weight);
    }
    
  }
}
  
  void 
MCvsRecoVerticesAnalyzer::beginRun(const edm::Run& iRun, const edm::EventSetup&)
{
}

void 
MCvsRecoVerticesAnalyzer::endRun(const edm::Run& iRun, const edm::EventSetup&)
{
}



// ------------ method called once each job just before starting event loop  ------------
void 
MCvsRecoVerticesAnalyzer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
MCvsRecoVerticesAnalyzer::endJob() 
{
}

//define this as a plug-in
DEFINE_FWK_MODULE(MCvsRecoVerticesAnalyzer);
