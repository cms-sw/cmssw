// File: MixingModule.cc
// Description:  see MixingModule.h
// Author:  Ursula Berthon, LLR Palaiseau
//
//--------------------------------------------

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimGeneral/MixingModule/interface/MixingModule.h"
#include "FWCore/Framework/interface/ConstProductRegistry.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/Handle.h"

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
      }
      else if (!desc.productInstanceName_.compare(0,4,"Muon")) {
	muonSubdetectors_.push_back(desc.productInstanceName_);
     }
      // add only the subdetector name, one for high and low together
      else if (!desc.productInstanceName_.compare(0,11,"TrackerHits")) {
       int slow=(desc.productInstanceName_).find("LowTof");
       int iend=(desc.productInstanceName_).size();
       if (slow>0)  
	 trackerSubdetectors_.push_back(desc.productInstanceName_.substr(0,iend-6));
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
    LogDebug("addSignals")<<"adding signals for event "<<e.id();

    // muon hits for all subdetectors
    for(std::vector<std::string >::const_iterator it = muonSubdetectors_.begin(); it != muonSubdetectors_.end(); ++it) {  
      edm::Handle<std::vector<PSimHit> > simHits;
      e.getByLabel("SimG4Object",(*it),simHits);
      simcf_->addSignalSimHits((*it),simHits.product());
      LogDebug("addSignals") <<"Subdet "<<(*it)<<" got "<<(simHits.product())->size()<<" simhits";
    }
     // tracker hits for all subdetectors
    for(std::vector<std::string >::const_iterator it = trackerSubdetectors_.begin(); it != trackerSubdetectors_.end(); ++it) {  
      edm::Handle<std::vector<PSimHit> > simHits;
      std::string subdet=(*it)+"HighTof";
      e.getByLabel("SimG4Object",subdet,simHits);
      simcf_->addSignalSimHits(subdet,simHits.product());
      LogDebug("addSignals") <<"Subdet "<<subdet<<" got "<<(simHits.product())->size()<<" simhits";
      subdet=(*it)+"LowTof";
      e.getByLabel("SimG4Object",subdet,simHits);
      simcf_->addSignalSimHits(subdet,simHits.product());
      LogDebug("addSignals")  <<"Subdet "<<subdet<<" got "<<(simHits.product())->size()<<" simhits";
    }
    // calo hits for all subdetectors
    for(std::vector<std::string >::const_iterator it = caloSubdetectors_.begin(); it != caloSubdetectors_.end(); ++it) {  
      edm::Handle<std::vector<PCaloHit> > caloHits;
      e.getByLabel("SimG4Object",(*it),caloHits);
      simcf_->addSignalCaloHits((*it),caloHits.product());
      LogDebug("addSignals")  <<"Got "<<(caloHits.product())->size()<<" calohits for subdet "<<(*it);
    }
    edm::Handle<std::vector<EmbdSimTrack> > simtracks;
    e.getByLabel("SimG4Object",simtracks);
    if (simtracks.isValid()) simcf_->addSignalTracks(simtracks.product());
    else  LogWarning("InvalidData") <<"Invalid simtracks in signal";
    LogDebug("addSignals") <<"Got "<<(simtracks.product())->size()<<" simtracks";
    edm::Handle<std::vector<EmbdSimVertex> > simvertices;
    e.getByLabel("SimG4Object",simvertices);
    if (simvertices.isValid())     simcf_->addSignalVertices(simvertices.product());
    else LogWarning("InvalidData") <<"Invalid simvertices in signal";
    LogDebug("addSignals")  <<"Got "<<(simvertices.product())->size()<<" simvertices";
  }

  void MixingModule::addPileups(const int bcr, Event *e) {

    LogDebug("addPileups") <<"adding pileups from event  "<<e->id()<<" for bunchcrossing "<<bcr;

    // Muons
    for(std::vector<std::string >::iterator itstr = muonSubdetectors_.begin(); itstr != muonSubdetectors_.end(); ++itstr) {
      edm::Handle<std::vector<PSimHit> >  simHits;  //Event Pointer to minbias Hits

      e->getByLabel("SimG4Object",(*itstr),simHits);
      simcf_->addPileupSimHits(bcr,(*itstr),simHits.product(),trackoffset,false);
    }

    // Tracker
    float tof = bcr*simcf_->getBunchSpace();
    for(std::vector<std::string >::iterator itstr = trackerSubdetectors_.begin(); itstr != trackerSubdetectors_.end(); ++itstr) {
      edm::Handle<std::vector<PSimHit> >  simHits;  //Event Pointer to minbias Hits

      // do not read branches if clearly outside of tof bounds (and verification is asked for, default)
      std::string subdethigh=(*itstr)+"HighTof";
      std::string subdetlow=(*itstr)+"LowTof";
      // add HighTof pileup to high and low signals
      if ( !checktof_ || ((CrossingFrame::limHighLowTof +tof ) <= CrossingFrame::highTrackTof)) { 
	e->getByLabel("SimG4Object",subdethigh,simHits);
	simcf_->addPileupSimHits(bcr,subdethigh,simHits.product(),trackoffset,checktof_);
	simcf_->addPileupSimHits(bcr,subdetlow,simHits.product(),trackoffset,checktof_);
      }

      // add LowTof pileup to high and low signals
      if (  !checktof_ || ((tof+CrossingFrame::limHighLowTof) >= CrossingFrame::lowTrackTof && tof <= CrossingFrame::highTrackTof)) {     
	e->getByLabel("SimG4Object",subdetlow,simHits);
	simcf_->addPileupSimHits(bcr,subdethigh,simHits.product(),trackoffset,checktof_);
	simcf_->addPileupSimHits(bcr,subdetlow,simHits.product(),trackoffset,checktof_);
      }
    }

    // Calohits
    for(std::vector<std::string >::const_iterator itstr = caloSubdetectors_.begin(); itstr != caloSubdetectors_.end(); ++itstr) {

      edm::Handle<std::vector<PCaloHit> >  caloHits;  //Event Pointer to minbias Hits
      e->getByLabel("SimG4Object",(*itstr),caloHits);
      simcf_->addPileupCaloHits(bcr,(*itstr),caloHits.product(),trackoffset);
    }

    //then simtracks
    edm::Handle<std::vector<EmbdSimTrack> > simtracks;
    e->getByLabel("SimG4Object",simtracks);
    if (simtracks.isValid()) simcf_->addPileupTracks(bcr, simtracks.product(),vertexoffset);
    else LogWarning("InvalidData") <<"Invalid simtracks in pileup";

    //then simvertices
    edm::Handle<std::vector<EmbdSimVertex> > simvertices;
    e->getByLabel("SimG4Object",simvertices);
    if (simvertices.isValid())  simcf_->addPileupVertices(bcr,simvertices.product(),trackoffset);
    else  LogWarning("InvalidData") <<"Invalid simvertices in pileup";

    // increment offsets
    vertexoffset+=simvertices.product()->size();
    trackoffset+=simtracks.product()->size();
  }
 
  void MixingModule::put(edm::Event &e) {
    e.put(std::auto_ptr<CrossingFrame>(simcf_));
  }

} //edm
