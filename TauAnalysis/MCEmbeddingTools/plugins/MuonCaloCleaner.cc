// -*- C++ -*-
//
// Package:    MuonCaloCleaner
// Class:      MuonCaloCleaner
// 
/**\class MuonCaloCleaner MuonCaloCleaner.cc MyAna/MuonCaloCleaner/src/MuonCaloCleaner.cc

Description: <one line class summary>

Implementation:
    <Notes on implementation>
*/
//
// Original Author:  Tomasz Maciej Frueboes
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/View.h"
#include <DataFormats/RecoCandidate/interface/RecoCandidate.h>
#include <DataFormats/Candidate/interface/CompositeRefCandidate.h>
#include <DataFormats/MuonReco/interface/Muon.h>

#include <DataFormats/ParticleFlowCandidate/interface/PFCandidate.h>
#include <DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h>

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"

#include "DataFormats/Candidate/interface/CompositeCandidate.h"
#include <DataFormats/Math/interface/deltaR.h>

#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"
#include "TrackingTools/TrackAssociator/interface/TrackAssociatorParameters.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"

//
// class decleration
//
#include <boost/foreach.hpp>

class MuonCaloCleaner : public edm::EDProducer {
  public:
      explicit MuonCaloCleaner(const edm::ParameterSet&);
      ~MuonCaloCleaner();

  private:
      virtual void beginJob() ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      void fillMap(const TrackDetMatchInfo & info, 
		  std::auto_ptr<std::map< uint32_t , float> > & ret);
      
      void storeDeps(const TrackDetMatchInfo & info, 
		     std::auto_ptr<std::map< uint32_t , float> > & retPtrDeposits);
      
      edm::InputTag _inputCol;
      bool _useCombinedCandidate;
      bool _storeDeposits;

      TrackDetectorAssociator trackAssociator_;
      TrackAssociatorParameters parameters_;
      

      // ----------member data ---------------------------
};

MuonCaloCleaner::MuonCaloCleaner(const edm::ParameterSet& iConfig)
  : _inputCol(iConfig.getParameter<edm::InputTag>("selectedMuons")),
    _useCombinedCandidate(iConfig.getUntrackedParameter<bool>("useCombinedCandidate", false)),
    _storeDeposits(iConfig.getUntrackedParameter<bool>("storeDeps", false))

  
{

  // uid <-> length  assigment 
  produces<  std::map< uint32_t , float> >("plus");
  produces<  std::map< uint32_t , float> >("minus");
  if (_storeDeposits){
    produces<  std::map< uint32_t , float> >("plusDeposits");
    produces<  std::map< uint32_t , float> >("minusDeposits");
  }
  
  edm::ParameterSet parameters = iConfig.getParameter<edm::ParameterSet>("TrackAssociatorParameters");
  parameters_.loadParameters( parameters );
  trackAssociator_.useDefaultPropagator();
  
}

MuonCaloCleaner::~MuonCaloCleaner()
{

  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
MuonCaloCleaner::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  using namespace reco;
  std::auto_ptr<  std::map< uint32_t , float> > retPtrPlus(new std::map< uint32_t , float>) ;
  std::auto_ptr<  std::map< uint32_t , float> > retPtrDepositsPlus(new std::map< uint32_t , float>) ;
  std::auto_ptr<  std::map< uint32_t , float> > retPtrMinus(new std::map< uint32_t , float>) ;
  std::auto_ptr<  std::map< uint32_t , float> > retPtrDepositsMinus(new std::map< uint32_t , float>) ;
  
  
  std::vector< const reco::Muon * > toBeAdded;
  
  if (_useCombinedCandidate){
    edm::Handle< std::vector< reco::CompositeCandidate > > combCandidatesHandle;
    if (  iEvent.getByLabel(_inputCol, combCandidatesHandle) 
	  && combCandidatesHandle->size()>0 ) 
    {
      for (size_t idx = 0; idx < combCandidatesHandle->at(0).numberOfDaughters(); ++idx)	{
	const Muon * mu = dynamic_cast <const Muon *> (combCandidatesHandle->at(0).daughter(idx));
	if (mu == 0) {
	  std::cout << "XXX cannot cast" << std::endl;
	  return;
	  
	}
	toBeAdded.push_back(mu);
      }
    }
  } else {
    edm::Handle< std::vector<reco::Muon> > muonsHandle;
    iEvent.getByLabel(_inputCol, muonsHandle); 
    
    BOOST_FOREACH(const Muon & mu, *muonsHandle){
      if (!mu.isGlobalMuon()) continue;
      toBeAdded.push_back(&mu);
      if (toBeAdded.size()==2) break;
    }
    
  }
  
  if (toBeAdded.size() != 2){
      //std::cout << "XXX size wrong" << toBeAdded.size() << std::endl;
      //return;

  }
  
  for (size_t i = 0; i < toBeAdded.size(); ++i) {
    if ( !toBeAdded.at(i)->isGlobalMuon()){
      std::cout << "Is not global muon!" <<std::endl;
      return;
    }
      
    TrackDetMatchInfo info =
	trackAssociator_.associate(iEvent, iSetup, *(toBeAdded.at(i)->globalTrack()), parameters_);

    if (toBeAdded.at(i)->charge() > 0){
      fillMap(info, retPtrPlus);
      if (_storeDeposits) storeDeps(info, retPtrDepositsPlus);
    } else {
      fillMap(info, retPtrMinus);
      if (_storeDeposits) storeDeps(info, retPtrDepositsMinus);
    }

  }


  iEvent.put(retPtrPlus,"plus");
  iEvent.put(retPtrMinus,"minus");
  if (_storeDeposits){
    iEvent.put(retPtrDepositsPlus,"plusDeposits");
    iEvent.put(retPtrDepositsMinus,"minusDeposits");
  }
  
  
}

void MuonCaloCleaner::fillMap(const TrackDetMatchInfo & info, 
		  std::auto_ptr<std::map< uint32_t , float> > & ret)
{
		
  
  typedef std::map<std::string, const std::vector<DetId> * > TMyStore;
  TMyStore todo;
  todo["ecal"] = &(info.crossedEcalIds);	
  todo["hcal"] = &(info.crossedHcalIds);
  todo["ho"] = &(info.crossedHOIds);
  //todo["towers"] = &(info.crossedTowerIds);
  todo["es"] = &(info.crossedPreshowerIds);
    
    
  BOOST_FOREACH( TMyStore::value_type &entry, todo ){
    // get trajectory depending on detector type;
    std::vector<GlobalPoint> trajectory;
    std::vector<SteppingHelixStateInfo>::const_iterator it, itE;
    
    if (entry.first=="ecal") {
      it = trackAssociator_.cachedTrajectory_.getEcalTrajectory().begin();
      itE = trackAssociator_.cachedTrajectory_.getEcalTrajectory().end();
    } else if (entry.first=="hcal") {
      it = trackAssociator_.cachedTrajectory_.getHcalTrajectory().begin();
      itE = trackAssociator_.cachedTrajectory_.getHcalTrajectory().end();
    } else if (entry.first=="ho") {
      it = trackAssociator_.cachedTrajectory_.getHOTrajectory().begin();
      itE =  trackAssociator_.cachedTrajectory_.getHOTrajectory().end();
    } else if (entry.first=="es") {
      it = trackAssociator_.cachedTrajectory_.getPreshowerTrajectory().begin();
      itE = trackAssociator_.cachedTrajectory_.getPreshowerTrajectory().end();
    } else {
      std::cout << "Unsupportted type !\n";
      throw "XXX";
    }
    
    // copy trajectory points
    for (;it!=itE;++it) trajectory.push_back(it->position());
    
    // iterate over crossed detIDS
    std::vector<DetId>::const_iterator itDet, itDetE;
    itDet =  entry.second->begin();
    itDetE = entry.second->end();
    for (;itDet!=itDetE;++itDet){
      if (itDet->rawId()==0) continue;
      const CaloSubdetectorGeometry* subDetGeom = 
	    trackAssociator_.theCaloGeometry_->getSubdetectorGeometry(*itDet);
      const CaloCellGeometry * myGeo = subDetGeom->getGeometry(*itDet);GlobalPoint prev;
      bool prevInitilized = false;
      float distInside = 0;
      BOOST_FOREACH( GlobalPoint & p, trajectory) {
	if (!prevInitilized) {
	  prevInitilized = true; // copied at end of the loop
	} else {
	      float dx = p.x()-prev.x();
	      float dy = p.y()-prev.y();
	      float dz = p.z()-prev.z();
	      float dist = sqrt(dx*dx + dy*dy + dz*dz);
	      int steps = 100;
	      int stepsInsideLocal = 0;
	      for(int i = 0; i <= steps; ++i){
		float steppingX = prev.x()+i*dx/steps;
		float steppingY = prev.y()+i*dy/steps;
		float steppingZ = prev.z()+i*dz/steps;
		GlobalPoint steppingPoint(steppingX, steppingY, steppingZ );
		
		bool inside = myGeo->inside(steppingPoint);
		if (inside) ++stepsInsideLocal;
		/*
		std::cout << " " << steppingX
		    << " " << steppingY
		    << " " << steppingZ
		    << " " << inside
		    << std::endl;
		// */
	      }
	      distInside += float(stepsInsideLocal)/float(steps + 1)*dist;
	}
	prev = p;
      } // end trajectory iteration
      //std::cout << entry.first << " " << itDet->rawId() << " " << distInside << std::endl;
      (*ret)[itDet->rawId()]=distInside;
    }// end detID iteration
  }// end hcal/ecal/ho iteration
    

  
  
}


void MuonCaloCleaner::storeDeps(const TrackDetMatchInfo & info, 
				std::auto_ptr<std::map< uint32_t , float> > & retPtrDeposits)
{
  
  BOOST_FOREACH(const EcalRecHit * rh, info.crossedEcalRecHits)
    (*retPtrDeposits)[rh->detid().rawId()]=rh->energy();
  
  BOOST_FOREACH(const HBHERecHit * rh, info.crossedHcalRecHits)
    (*retPtrDeposits)[rh->detid().rawId()]=rh->energy();

  BOOST_FOREACH(const HORecHit * rh, info.crossedHORecHits)
    (*retPtrDeposits)[rh->detid().rawId()]=rh->energy();

    
}


// ------------ method called once each job just before starting event loop  ------------
void 
MuonCaloCleaner::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
MuonCaloCleaner::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(MuonCaloCleaner);
