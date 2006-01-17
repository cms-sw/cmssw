// File: MixingModule.cc
// Description:  see MixingModule.h
// Author:  Ursula Berthon, LLR Palaiseau
//
//--------------------------------------------

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

    // get tracker + muon subdetector names
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
      else if (!desc.productInstanceName_.compare(0,11,"TrackerHits") || !desc.productInstanceName_.compare(0,4,"Muon")) {
	trackerSubdetectors_.push_back(desc.productInstanceName_);
      }
    }

    produces<CrossingFrame> ();
     
  }

  void MixingModule::createnewEDProduct() {
    simcf_=new CrossingFrame(minBunch(),maxBunch(),bunchSpace_,trackerSubdetectors_,caloSubdetectors_);
  }

  // Virtual destructor needed.
  MixingModule::~MixingModule() { }  

  void MixingModule::addSignals(const edm::Event &e) { 
    // fill in signal part of CrossingFrame
    // first add eventID
    simcf_->setEventID(e.id());
    //    std::cout<<"\naddsignals for  "<<e.id()<<endl;

    // tracker hits for all subdetectors
    for(std::vector<std::string >::const_iterator it = trackerSubdetectors_.begin(); it != trackerSubdetectors_.end(); ++it) {  
      edm::Handle<std::vector<PSimHit> > simHits;
      e.getByLabel("r",(*it),simHits);
      simcf_->addSignalSimHits((*it),simHits.product());
      cout <<" Subdet "<<(*it)<<" got "<<(simHits.product())->size()<<" simhits "<<endl;
    }
    // calo hits for all subdetectors
    for(std::vector<std::string >::const_iterator it = caloSubdetectors_.begin(); it != caloSubdetectors_.end(); ++it) {  
      edm::Handle<std::vector<PCaloHit> > caloHits;
      e.getByLabel("r",(*it),caloHits);
      simcf_->addSignalCaloHits((*it),caloHits.product());
      cout <<" Got "<<(caloHits.product())->size()<<" calohits for subdet "<<(*it)<<endl;
    }
    edm::Handle<std::vector<EmbdSimTrack> > simtracks;
    e.getByLabel("r",simtracks);
    if (simtracks.isValid()) simcf_->addSignalTracks(simtracks.product());
    else cout <<"Invalid simtracks"<<endl;
    //cout <<" Got "<<(simtracks.product())->size()<<" simtracks"<<endl;
    edm::Handle<std::vector<EmbdSimVertex> > simvertices;
    e.getByLabel("r",simvertices);
    if (simvertices.isValid())     simcf_->addSignalVertices(simvertices.product());
    else cout <<"Invalid simvertices"<<endl;
  }

  void MixingModule::addPileups(const int bcr, Event *e) {

   //      std::cout<<"\naddPileups from event  "<<e->id()<<endl;
   // first all simhits
    for(std::vector<std::string >::iterator itstr = trackerSubdetectors_.begin(); itstr != trackerSubdetectors_.end(); ++itstr) {
      std::vector<PSimHit> *sig;
      simcf_->getSignal((*itstr),sig);
      if (!sig->size()) continue;  // do not accumulate if there are no signals

      edm::Handle<std::vector<PSimHit> >  simHits;  //Event Pointer to minbias Hits

      // do not read this branch if clearly outside of tof bounds
      float tof = bcr*simcf_->getBunchSpace();
      int slow=(*itstr).find("LowTof");
      int shigh=(*itstr).find("HighTof");
      if (slow>0 )
	if ( (tof<CrossingFrame::lowTrackTof || tof>0)) {
          continue;
	}
	else if (shigh>0 )
	  if ( ((CrossingFrame::highTrackTof+tof)<CrossingFrame::lowTrackTof || tof>0)) {
	  continue;
	  }
      e->getByLabel("r",(*itstr),simHits);
      simcf_->addPileupSimHits(bcr,(*itstr),simHits.product(),trackoffset,slow>0||shigh>0);
    }

    //     //then all calohits
    for(std::vector<std::string >::const_iterator itstr = caloSubdetectors_.begin(); itstr != caloSubdetectors_.end(); ++itstr) {
      std::vector<PCaloHit> *sig;
      simcf_->getSignal((*itstr),sig);
      if (!sig->size()) continue;  // do not accumulate if there are no signals

      edm::Handle<std::vector<PCaloHit> >  caloHits;  //Event Pointer to minbias Hits
      e->getByLabel("r",(*itstr),caloHits);
      simcf_->addPileupCaloHits(bcr,(*itstr),caloHits.product(),trackoffset);
    }
    //then simtracks
    edm::Handle<std::vector<EmbdSimTrack> > simtracks;
    std::vector<EmbdSimTrack> *sigtrack;
    simcf_->getSignal("",sigtrack);
    if (sigtrack->size()) { // do not accumulate if there are no signals
      e->getByLabel("r",simtracks);
      if (simtracks.isValid()) simcf_->addPileupTracks(bcr, simtracks.product(),vertexoffset);
      else cout <<"Invalid simtracks"<<endl;
    }

    //then simvertices
    edm::Handle<std::vector<EmbdSimVertex> > simvertices;
    std::vector<EmbdSimVertex> *sigvtx;
    simcf_->getSignal("",sigvtx);
    if (sigvtx->size()) { // do not accumulate if there are no signals
      e->getByLabel("r",simvertices);
      if (simvertices.isValid())  simcf_->addPileupVertices(bcr,simvertices.product(),trackoffset);
      else cout <<"Invalid simvertices"<<endl;
    }

    // increment offsets
    vertexoffset+=simvertices.product()->size();
    trackoffset+=simtracks.product()->size();
  }
 
  void MixingModule::put(edm::Event &e) {
    e.put(std::auto_ptr<CrossingFrame>(simcf_));
  }

} //edm
