// File: MixingModule.cc
// Description:  see MixingModule.h
// Author:  Ursula Berthon, LLR Palaiseau
//
//--------------------------------------------

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Framework/interface/ConstProductRegistry.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "SimGeneral/MixingModule/interface/MixingModule.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/Selector.h"
#include "FWCore/Framework/interface/Provenance.h"

using namespace std;

namespace edm
{

  // Constructor 
  MixingModule::MixingModule(const edm::ParameterSet& ps) : BMixingModule(ps)
  {

    // get subdetector names
    edm::Service<edm::ConstProductRegistry> reg;
    // Loop over provenance of products in registry.
    for (edm::ProductRegistry::ProductList::const_iterator it = reg->productList().begin();
	 it != reg->productList().end(); ++it) {
      // See FWCore/Framework/interface/BranchDescription.h
      // BranchDescription contains all the information for the product.
      edm::BranchDescription desc = it->second;
      if (!desc.productInstanceName_.compare(0,8,"EcalHits") || !desc.productInstanceName_.compare(0,8,"HcalHits" )) {
	caloSubdetectors_.push_back(desc.productInstanceName_);
	LogInfo("Constructor") <<"Adding detector "<<desc.productInstanceName_ <<" for pileup treatment";
      }
      else if (!desc.productInstanceName_.compare(0,4,"Muon")) {
	muonSubdetectors_.push_back(desc.productInstanceName_);
	LogInfo("Constructor") <<"Adding detector "<<desc.productInstanceName_ <<" for pileup treatment";
     }
      else if (!desc.productInstanceName_.compare(0,11,"TrackerHits")) {
	 trackerSubdetectors_.push_back(desc.productInstanceName_);
	 LogInfo("Constructor") <<"Adding detector "<<desc.productInstanceName_ <<" for pileup treatment";
      }
    }

    produces<CrossingFrame> ();
}

  void MixingModule::createnewEDProduct() {
    simcf_=new CrossingFrame(minBunch(),maxBunch(),bunchSpace_,muonSubdetectors_,trackerSubdetectors_,caloSubdetectors_);
  }

  // Virtual destructor needed.
  MixingModule::~MixingModule() { }  

  void MixingModule::addSignals(const edm::Event &e) { 
    // fill in signal part of CrossingFrame

    // first add eventID
    simcf_->setEventID(e.id());
    LogDebug("add")<<"===============> adding signals for "<<e.id();

    // SimHits
    std::vector<edm::Handle<std::vector<PSimHit> > > resultsim;
    e.getManyByType(resultsim);
    int ss=resultsim.size();
    for (int ii=0;ii<ss;ii++) {
      edm::BranchDescription desc = resultsim[ii].provenance()->product;
      //      cout <<"=============================> Provenance "<<*(resultsim[ii].provenance())<<endl;
      LogDebug("addSignals") <<"For "<<desc.productInstanceName_<<resultsim[ii].product()->size()<<" Simhits added";
      simcf_->addSignalSimHits(desc.productInstanceName_,resultsim[ii].product());
    }


    // calo hits for all subdetectors
    std::vector<edm::Handle<std::vector<PCaloHit> > > resultcalo;
    e.getManyByType(resultcalo);
    int sc=resultcalo.size();
    for (int ii=0;ii<sc;ii++) {
      edm::BranchDescription desc = resultcalo[ii].provenance()->product;
      LogDebug("addSignals") <<"For "<<desc.productInstanceName_<<resultcalo[ii].product()->size()<<" Calohits added";
      simcf_->addSignalCaloHits(desc.productInstanceName_,resultcalo[ii].product());
    }
  

//     //tracks and vertices
    std::vector<edm::Handle<std::vector<EmbdSimTrack> > > result_t;
    e.getManyByType(result_t);
    int str=result_t.size();
    for (int ii=0;ii<str;ii++) {
      edm::BranchDescription desc =result_t[ii].provenance()->product;
      LogDebug("addSignals") <<result_t[ii].product()->size()<<" Simtracks added";
      if (result_t[ii].isValid()) simcf_->addSignalTracks(result_t[ii].product());
      else  LogWarning("InvalidData") <<"Invalid simtracks in signal";
    }

    std::vector<edm::Handle<std::vector<EmbdSimVertex> > > result_v;
    e.getManyByType(result_v);
    int sv=result_v.size();
    for (int ii=0;ii<sv;ii++) {
      edm::BranchDescription desc = result_v[ii].provenance()->product;
      LogDebug("addSignals") <<result_v[ii].product()->size()<<" Simvertices added";
      if (result_v[ii].isValid()) simcf_->addSignalVertices(result_v[ii].product());
      else  LogWarning("InvalidData") <<"Invalid simvertices in signal";
    }
  }

  void MixingModule::addPileups(const int bcr, Event *e) {

    LogDebug("addPileups") <<"===============> adding pileups from event  "<<e->id()<<" for bunchcrossing "<<bcr;

    // SimHits
    std::vector<edm::Handle<std::vector<PSimHit> > > resultsim;
    e->getManyByType(resultsim);
    int ss=resultsim.size();
    for (int ii=0;ii<ss;ii++) {
      edm::BranchDescription desc = resultsim[ii].provenance()->product;
      printf(" Adding Pileup productInstanceName_ %s, %d hits\n",desc.productInstanceName_.c_str(),resultsim[ii].product()->size());fflush(stdout);
      LogDebug("addPileups") <<"For "<<desc.productInstanceName_<<resultsim[ii].product()->size()<<" Simhits added";
      simcf_->addPileupSimHits(bcr,desc.productInstanceName_,resultsim[ii].product(),trackoffset,false);
    }

//     // Tracker
//     float tof = bcr*simcf_->getBunchSpace();
//     for(std::vector<std::string >::iterator itstr = trackerSubdetectors_.begin(); itstr != trackerSubdetectors_.end(); ++itstr) {
//       std::string subdethigh=(*itstr)+"HighTof";
//       std::string subdetlow=(*itstr)+"LowTof";

//       // do not read branches if clearly outside of tof bounds (and verification is asked for, default)
//       // add HighTof pileup to high and low signals
//       if ( !checktof_ || ((CrossingFrame::limHighLowTof +tof ) <= CrossingFrame::highTrackTof)) { 
// 	const edm::ProcessNameSelector sel(subdethigh+"_r");
// 	std::vector<edm::Handle<std::vector<PSimHit> > > result;
// 	e->getMany(sel, result);
// 	//	e->getByLabel("r",subdethigh,simHits);
// 	if (result.size()>0) {
// 	  printf("Processname %s, size %d\n",(*itstr).c_str(),result.size());fflush(stdout);
// 	  if (result.size()>1) {
// 	    LogWarning("addPileups") <<"Got more than one EDProduct corresponding to "<<subdethigh+"_r";
// 	  } else {
// 	    edm::Handle<std::vector<PSimHit> > simHits=result[0];
// 	    simcf_->addPileupSimHits(bcr,subdethigh,simHits.product(),trackoffset,checktof_);
// 	    simcf_->addPileupSimHits(bcr,subdetlow,simHits.product(),trackoffset,checktof_);
// 	  }
// 	}
//       }

//       // add LowTof pileup to high and low signals
//       if (  !checktof_ || ((tof+CrossingFrame::limHighLowTof) >= CrossingFrame::lowTrackTof && tof <= CrossingFrame::highTrackTof)) {     
// 	const edm::ProcessNameSelector sel(subdetlow+"_r");
// 	std::vector<edm::Handle<std::vector<PSimHit> > > result;
// 	e->getMany(sel, result);
// 	//	e->getByLabel("r",subdetlow,simHits);
// 	if (result.size()>0) {
// 	  printf("Processname %s, size %d\n",(*itstr).c_str(),result.size());fflush(stdout);
// 	  if (result.size()>1) {
// 	    LogWarning("addPileups") <<"Got more than one EDProduct corresponding to "<<subdethigh+"_r";
// 	  } else {
// 	    edm::Handle<std::vector<PSimHit> > simHits=result[0];
// 	    simcf_->addPileupSimHits(bcr,subdethigh,simHits.product(),trackoffset,checktof_);
// 	    simcf_->addPileupSimHits(bcr,subdetlow,simHits.product(),trackoffset,checktof_);
// 	  }
// 	}
//       }
//     }


    // calo hits for all subdetectors
    std::vector<edm::Handle<std::vector<PCaloHit> > > resultcalo;
    e->getManyByType(resultcalo);
    int sc=resultcalo.size();
    for (int ii=0;ii<sc;ii++) {
      edm::BranchDescription desc = resultcalo[ii].provenance()->product;
      LogDebug("addPileups") <<"For "<<desc.productInstanceName_<<resultcalo[ii].product()->size()<<" Calohits added";
      simcf_->addPileupCaloHits(bcr,desc.productInstanceName_,resultcalo[ii].product(),trackoffset);
    }
 
//     //tracks and vertices
    std::vector<edm::Handle<std::vector<EmbdSimTrack> > > result_t;
    e->getManyByType(result_t);
    int str=result_t.size();
    if (str>1) LogWarning("InvalidData") <<"Too many SimTrack containers, should be only one!";
    for (int ii=0;ii<str;ii++) {
      edm::BranchDescription desc =result_t[ii].provenance()->product;
      LogDebug("addPileups") <<result_t[ii].product()->size()<<" Simtracks added";
      if (result_t[ii].isValid()) simcf_->addPileupTracks(bcr,result_t[ii].product(),vertexoffset);
      else  LogWarning("InvalidData") <<"Invalid simtracks in signal";
    }
  

    std::vector<edm::Handle<std::vector<EmbdSimVertex> > > result_v;
    e->getManyByType(result_v);
    int sv=result_v.size();
    if (sv>1) LogWarning("InvalidData") <<"Too many SimVertex containers, should be only one!";
    for (int ii=0;ii<sv;ii++) {
      edm::BranchDescription desc = result_v[ii].provenance()->product;
      LogDebug("addPileups") <<result_v[ii].product()->size()<<" Simvertices added";
      if (result_v[ii].isValid()) simcf_->addPileupVertices(bcr,result_v[ii].product(),trackoffset);
      else  LogWarning("InvalidData") <<"Invalid simvertices in signal";
    }
 
    // increment offsets
    vertexoffset+=result_v[0].product()->size();
    trackoffset+=result_t[0].product()->size();
  }
 
  void MixingModule::put(edm::Event &e) {
    e.put(std::auto_ptr<CrossingFrame>(simcf_));
  }

} //edm
