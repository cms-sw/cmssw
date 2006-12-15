/**_________________________________________________________________
   class:   BeamSpotTest.cc
   package: RecoVertex/BeamSpotProducer
   


 author: Francisco Yumiceva, Fermilab (yumiceva@fnal.gov)

 version $Id: BSTrkParameters.h,v 1.0 2006/09/19 17:13:31 yumiceva Exp $

________________________________________________________________**/


// C++ standard
#include <string>
// CMS
#include "RecoVertex/BeamSpotProducer/interface/BeamSpotTest.h"
#include "RecoVertex/BeamSpotProducer/interface/BSFitter.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/RoadSearchCloud/interface/RoadSearchCloud.h"
#include "DataFormats/RoadSearchCloud/interface/RoadSearchCloudCollection.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidate.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"


BeamSpotTest::BeamSpotTest(const edm::ParameterSet& iConfig)
{

  file_ = new TFile(iConfig.getUntrackedParameter<std::string>("OutputFileName").c_str(),"RECREATE");

  ftree_ = new TTree("mytree","mytree");
  
  ftree_->Branch("pt",&fpt,"fpt/D");
  ftree_->Branch("d0",&fd0,"fd0/D");
  ftree_->Branch("sigmad0",&fsigmad0,"fsigmad0/D");
  ftree_->Branch("phi0",&fphi0,"fphi0/D");
  ftree_->Branch("z0",&fz0,"fz0/D");
  ftree_->Branch("sigmaz0",&fsigmaz0,"fsigmaz0/D");
  ftree_->Branch("theta",&ftheta,"ftheta/D");
  ftree_->Branch("eta",&feta,"feta/D");
  ftree_->Branch("charge",&fcharge,"fcharge/I");
  ftree_->Branch("chi2",&fchi2,"fchi2/D");
  ftree_->Branch("ndof",&fndof,"fndof/D");
  ftree_->Branch("nHit",&fnHit,"fnHit/i");
  ftree_->Branch("nStripHit",&fnStripHit,"fnStripHit/i");
  ftree_->Branch("nPixelHit",&fnPixelHit,"fnPixelHit/i");
  ftree_->Branch("nTIBHit",&fnTIBHit,"fnTIBHit/i");
  ftree_->Branch("nTOBHit",&fnTOBHit,"fnTOBHit/i");
  ftree_->Branch("nTIDHit",&fnTIDHit,"fnTIDHit/i");
  ftree_->Branch("nTECHit",&fnTECHit,"fnTECHit/i");
  ftree_->Branch("nPXBHit",&fnPXBHit,"fnPXBHit/i");
  ftree_->Branch("nPXFHit",&fnPXFHit,"fnPXFHit/i");
    
  
  //ftree_->Branch("vx",&fvx,"fvx/D");
  //ftree_->Branch("vx",&fvx,"fvx/D");
  
  fBSvector.erase(fBSvector.begin(),fBSvector.end());

  
  // get parameter
  rsSeedProducerLabel_ = iConfig.getUntrackedParameter<std::string>("rsSeedProducerLabel");
  rsRawCloudProducerLabel_ = iConfig.getUntrackedParameter<std::string>("rsRawCloudProducerLabel");
  rsCleanCloudProducerLabel_ = iConfig.getUntrackedParameter<std::string>("rsCleanCloudProducerLabel");
  rsTrackCandidateProducerLabel_ = iConfig.getUntrackedParameter<std::string>("rsTrackCandidateProducerLabel");
  rsTrackProducerLabel_ = iConfig.getUntrackedParameter<std::string>("rsTrackProducerLabel");

  ckfSeedProducerLabel_ = iConfig.getUntrackedParameter<std::string>("ckfSeedProducerLabel");
  ckfTrackCandidateProducerLabel_ = iConfig.getUntrackedParameter<std::string>("ckfTrackCandidateProducerLabel");
  ckfTrackProducerLabel_ = iConfig.getUntrackedParameter<std::string>("ckfTrackProducerLabel");

  sameNumberOfTracks = iConfig.getUntrackedParameter<unsigned int>("sameNumberOfTracks");

}


BeamSpotTest::~BeamSpotTest()
{
 
  if ( file_ != 0 ) {
    file_->cd();
    file_->Write();
    delete file_;
  }

}


void
BeamSpotTest::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

	ftree_->SetBranchAddress("theta",&ftheta);
	ftree_->SetBranchAddress("pt",&fpt);
	ftree_->SetBranchAddress("eta",&feta);
	ftree_->SetBranchAddress("charge",&fcharge);
	ftree_->SetBranchAddress("chi2",&fchi2);
	ftree_->SetBranchAddress("ndof",&fndof);
	ftree_->SetBranchAddress("d0",&fd0);
	ftree_->SetBranchAddress("sigmad0",&fsigmad0);
	ftree_->SetBranchAddress("phi0",&fphi0);
	ftree_->SetBranchAddress("z0",&fz0);
	ftree_->SetBranchAddress("sigmaz0",&fsigmaz0);
	ftree_->SetBranchAddress("nHit",&fnHit);
	ftree_->SetBranchAddress("nStripHit",&fnStripHit);
	ftree_->SetBranchAddress("nPixelHit",&fnPixelHit);
	ftree_->SetBranchAddress("nTIBHit",&fnTIBHit);
	ftree_->SetBranchAddress("nTOBHit",&fnTOBHit);
	ftree_->SetBranchAddress("nTIDHit",&fnTIDHit);
	ftree_->SetBranchAddress("nTECHit",&fnTECHit);
	ftree_->SetBranchAddress("nPXBHit",&fnPXBHit);
	ftree_->SetBranchAddress("nPXFHit",&fnPXFHit);
	
  
	// get collections
	edm::Handle<TrajectorySeedCollection> rsSeedCollectionHandle;
	iEvent.getByLabel(rsSeedProducerLabel_,rsSeedCollectionHandle);
	const TrajectorySeedCollection *rsSeedCollection = rsSeedCollectionHandle.product();

	edm::Handle<RoadSearchCloudCollection> rsRawCloudCollectionHandle;
	iEvent.getByLabel(rsRawCloudProducerLabel_,rsRawCloudCollectionHandle);
	const RoadSearchCloudCollection *rsRawCloudCollection = rsRawCloudCollectionHandle.product();

	edm::Handle<RoadSearchCloudCollection> rsCleanCloudCollectionHandle;
	iEvent.getByLabel(rsCleanCloudProducerLabel_,rsCleanCloudCollectionHandle);
	const RoadSearchCloudCollection *rsCleanCloudCollection = rsCleanCloudCollectionHandle.product();

	edm::Handle<TrackCandidateCollection> rsTrackCandidateCollectionHandle;
	iEvent.getByLabel(rsTrackCandidateProducerLabel_,rsTrackCandidateCollectionHandle);
	const TrackCandidateCollection *rsTrackCandidateCollection = rsTrackCandidateCollectionHandle.product();
	
	edm::Handle<reco::TrackCollection> rsTrackCollectionHandle;
	iEvent.getByLabel(rsTrackProducerLabel_,rsTrackCollectionHandle);
	const reco::TrackCollection *rsTrackCollection = rsTrackCollectionHandle.product();

	edm::Handle<TrajectorySeedCollection> ckfSeedCollectionHandle;
	iEvent.getByLabel(ckfSeedProducerLabel_,ckfSeedCollectionHandle);
	const TrajectorySeedCollection *ckfSeedCollection = ckfSeedCollectionHandle.product();

	edm::Handle<TrackCandidateCollection> ckfTrackCandidateCollectionHandle;
	iEvent.getByLabel(ckfTrackCandidateProducerLabel_,ckfTrackCandidateCollectionHandle);
	const TrackCandidateCollection *ckfTrackCandidateCollection = ckfTrackCandidateCollectionHandle.product();

	edm::Handle<reco::TrackCollection> ckfTrackCollectionHandle;
	iEvent.getByLabel(ckfTrackProducerLabel_,ckfTrackCollectionHandle);
	const reco::TrackCollection *ckfTrackCollection = ckfTrackCollectionHandle.product();



  // Ckf tracks
  //ckf_numTracks->Fill(ckfTrackCollection->size());
  for ( reco::TrackCollection::const_iterator track = ckfTrackCollection->begin();
	track != ckfTrackCollection->end();
	++track ) {
	  fpt = track->pt();
	  feta = track->eta();
	  fphi0 = track->phi0();
	  fcharge = track->charge();
	  fchi2 = track->chi2();
	  fndof = track->ndof();
	  
	  //fsigmaphi0 = track->phi0Error();
	  fd0 = track->d0();
	  fsigmad0 = track->d0Error();
	  fz0 = track->dz();
	  fsigmaz0 = track->dzError();
	  ftheta = track->theta();
	  
	  // loop over hits in tracks, count
	  fnHit      = 0;
	  fnStripHit = 0;
	  fnPixelHit = 0;
	  fnTIBHit   = 0;
	  fnTOBHit   = 0;
	  fnTIDHit   = 0;
	  fnTECHit   = 0;
	  fnPXBHit   = 0;
	  fnPXFHit   = 0;

	  for ( trackingRecHit_iterator recHit = track->recHitsBegin();
			recHit != track->recHitsEnd();
			++ recHit ) {

		  ++fnHit;
		  DetId id((*recHit)->geographicalId());

		  if ( (unsigned int)id.subdetId() == StripSubdetector::TIB ) {
			  ++fnStripHit;
			  ++fnTIBHit;
		  } else if ( (unsigned int)id.subdetId() == StripSubdetector::TOB ) {
			  ++fnStripHit;
			  ++fnTOBHit;
		  } else if ( (unsigned int)id.subdetId() == StripSubdetector::TID ) {
			  ++fnStripHit;
			  ++fnTIDHit;
		  } else if ( (unsigned int)id.subdetId() == StripSubdetector::TEC ) {
			  ++fnStripHit;
			  ++fnTECHit;
		  } else if ( (unsigned int)id.subdetId() == PixelSubdetector::PixelBarrel ) {
			  ++fnPixelHit;
			  ++fnPXBHit;
		  } else if ( (unsigned int)id.subdetId() == PixelSubdetector::PixelEndcap ) {
			  ++fnPixelHit;
			  ++fnPXFHit;
		  }
	  }

	  ftree_->Fill();

	  if (fnStripHit>=8 && fnPixelHit >= 2 &&
		  fchi2/fndof<5) {
		  fBSvector.push_back(BSTrkParameters(fz0,fsigmaz0,fd0,fsigmad0,fphi0,fpt));
	  }
	  
    
  }
  //std::cout << fBSvector.size() << " tracks read in.\n";

}



void 
BeamSpotTest::beginJob(const edm::EventSetup&)
{
}

void 
BeamSpotTest::endJob() {

	std::cout << " calculating beam spot..." << std::endl;
	std::cout << " we will use " << fBSvector.size() << " tracks." << std::endl;
		
	BSFitter *myalgo = new BSFitter( fBSvector );
	std::string fit_type = "chi2";
	myalgo->SetFitVariable(std::string("z"));
	myalgo->SetFitType(std::string("chi2"));
	BeamSpot beam_fit_z_chi2 = myalgo->Fit();
	beam_fit_z_chi2.Print(std::string("z chi2"));
	
	//std::cout << " z chi2 fit done." << std::endl;

	fit_type = "combined";
	myalgo->SetFitVariable("z");
	myalgo->SetFitType("combined");
	BeamSpot beam_fit_z_lh = myalgo->Fit();
	beam_fit_z_lh.Print(std::string("z likelihood"));

	//std::cout << " z combined chi2-likelihood fit done." << std::endl;

	myalgo->SetFitVariable("d");
	myalgo->SetFitType("d0phi");
	BeamSpot beam_fit_dphi = myalgo->Fit();
	beam_fit_dphi.Print(std::string("d0-phi"));

	//std::cout << " d0-phi fit done." << std::endl;

	
	myalgo->SetFitVariable(std::string("d*z"));
	myalgo->SetFitType(std::string("likelihood"));
	BeamSpot beam_fit_dz_lh = myalgo->Fit();
	beam_fit_dphi.Print(std::string("likelihood"));

	myalgo->SetFitVariable(std::string("d*z"));
	myalgo->SetFitType(std::string("resolution"));
	BeamSpot beam_fit_dresz_lh = myalgo->Fit();
	beam_fit_dphi.Print(std::string("resolution"));

	std::cout << "c0 = " << myalgo->GetResPar0() << " +- " << std::endl;
	std::cout << "c1 = " << myalgo->GetResPar1() << " +- " << std::endl;
	
	
}
