/**_________________________________________________________________
   class:   BeamFitter.cc
   package: RecoVertex/BeamSpotProducer
   


 author: Francisco Yumiceva, Fermilab (yumiceva@fnal.gov)
         Geng-Yuan Jeng, UC Riverside (Geng-Yuan.Jeng@cern.ch)
 
 version $Id: BeamFitter.cc,v 1.12 2009/10/26 16:31:44 yumiceva Exp $

 ________________________________________________________________**/

#include "RecoVertex/BeamSpotProducer/interface/BeamFitter.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/BeamSpotObjects/interface/BeamSpotObjects.h"

#include "FWCore/ParameterSet/interface/InputTag.h"

#include "DataFormats/TrackCandidate/interface/TrackCandidate.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"

BeamFitter::BeamFitter(const edm::ParameterSet& iConfig)
{

  debug_             = iConfig.getParameter<edm::ParameterSet>("BeamFitter").getUntrackedParameter<bool>("Debug");
  tracksLabel_       = iConfig.getParameter<edm::ParameterSet>("BeamFitter").getUntrackedParameter<edm::InputTag>("TrackCollection");
  writeTxt_          = iConfig.getParameter<edm::ParameterSet>("BeamFitter").getUntrackedParameter<bool>("WriteAscii");
  outputTxt_         = iConfig.getParameter<edm::ParameterSet>("BeamFitter").getUntrackedParameter<std::string>("AsciiFileName");
  saveNtuple_        = iConfig.getParameter<edm::ParameterSet>("BeamFitter").getUntrackedParameter<bool>("SaveNtuple");
  isMuon_        = iConfig.getParameter<edm::ParameterSet>("BeamFitter").getUntrackedParameter<bool>("IsMuonCollection");

  trk_MinpT_         = iConfig.getParameter<edm::ParameterSet>("BeamFitter").getUntrackedParameter<double>("MinimumPt");
  trk_MaxEta_        = iConfig.getParameter<edm::ParameterSet>("BeamFitter").getUntrackedParameter<double>("MaximumEta");
  trk_MaxIP_         = iConfig.getParameter<edm::ParameterSet>("BeamFitter").getUntrackedParameter<double>("MaximumImpactParameter");
  trk_MaxZ_          = iConfig.getParameter<edm::ParameterSet>("BeamFitter").getUntrackedParameter<double>("MaximumZ");
  trk_MinNTotLayers_ = iConfig.getParameter<edm::ParameterSet>("BeamFitter").getUntrackedParameter<int>("MinimumTotalLayers");
  trk_MinNPixLayers_ = iConfig.getParameter<edm::ParameterSet>("BeamFitter").getUntrackedParameter<int>("MinimumPixelLayers");
  trk_MaxNormChi2_   = iConfig.getParameter<edm::ParameterSet>("BeamFitter").getUntrackedParameter<double>("MaximumNormChi2");
  trk_Algorithm_     = iConfig.getParameter<edm::ParameterSet>("BeamFitter").getUntrackedParameter<std::vector<std::string> >("TrackAlgorithm");
  trk_Quality_       = iConfig.getParameter<edm::ParameterSet>("BeamFitter").getUntrackedParameter<std::vector<std::string> >("TrackQuality");
  min_Ntrks_         = iConfig.getParameter<edm::ParameterSet>("BeamFitter").getUntrackedParameter<int>("MinimumInputTracks");
  convergence_       = iConfig.getParameter<edm::ParameterSet>("BeamFitter").getUntrackedParameter<double>("FractionOfFittedTrks");
  inputBeamWidth_    = iConfig.getParameter<edm::ParameterSet>("BeamFitter").getUntrackedParameter<double>("InputBeamWidth",-1.);
  
  for (unsigned int j=0;j<trk_Algorithm_.size();j++)
    algorithm_.push_back(reco::TrackBase::algoByName(trk_Algorithm_[j]));
  for (unsigned int j=0;j<trk_Quality_.size();j++)
    quality_.push_back(reco::TrackBase::qualityByName(trk_Quality_[j]));

  //dump to file
  if (writeTxt_)
    fasciiFile.open(outputTxt_.c_str());

  if (saveNtuple_) {
    outputfilename_ = iConfig.getParameter<edm::ParameterSet>("BeamFitter").getUntrackedParameter<std::string>("OutputFileName");
    file_ = TFile::Open(outputfilename_.c_str(),"RECREATE");
    ftree_ = new TTree("mytree","mytree");
    ftree_->AutoSave();

    ftree_->Branch("pt",&fpt,"fpt/D");
    ftree_->Branch("d0",&fd0,"fd0/D");
    ftree_->Branch("sigmad0",&fsigmad0,"fsigmad0/D");
    ftree_->Branch("phi0",&fphi0,"fphi0/D");
    ftree_->Branch("z0",&fz0,"fz0/D");
    ftree_->Branch("sigmaz0",&fsigmaz0,"fsigmaz0/D");
    ftree_->Branch("theta",&ftheta,"ftheta/D");
    ftree_->Branch("eta",&feta,"feta/D");
    ftree_->Branch("charge",&fcharge,"fcharge/I");
    ftree_->Branch("normchi2",&fnormchi2,"fnormchi2/D");
    ftree_->Branch("nTotLayerMeas",&fnTotLayerMeas,"fnTotLayerMeas/i");
    ftree_->Branch("nStripLayerMeas",&fnStripLayerMeas,"fnStripLayerMeas/i");
    ftree_->Branch("nPixelLayerMeas",&fnPixelLayerMeas,"fnPixelLayerMeas/i");
    ftree_->Branch("nTIBLayerMeas",&fnTIBLayerMeas,"fnTIBLayerMeas/i");
    ftree_->Branch("nTOBLayerMeas",&fnTOBLayerMeas,"fnTOBLayerMeas/i");
    ftree_->Branch("nTIDLayerMeas",&fnTIDLayerMeas,"fnTIDLayerMeas/i");
    ftree_->Branch("nTECLayerMeas",&fnTECLayerMeas,"fnTECLayerMeas/i");
    ftree_->Branch("nPXBLayerMeas",&fnPXBLayerMeas,"fnPXBLayerMeas/i");
    ftree_->Branch("nPXFLayerMeas",&fnPXFLayerMeas,"fnPXFLayerMeas/i");
    ftree_->Branch("cov",&fcov,"fcov[7][7]/D");
	ftree_->Branch("vx",&fvx,"fvx/D");
 	ftree_->Branch("vy",&fvy,"fvy/D");
  }
  
  fBSvector.clear();
  ftotal_tracks = 0;
  fnTotLayerMeas = fnPixelLayerMeas = fnStripLayerMeas = fnTIBLayerMeas = 0;
  fnTIDLayerMeas = fnTOBLayerMeas = fnTECLayerMeas = fnPXBLayerMeas = fnPXFLayerMeas = 0;
  
}

BeamFitter::~BeamFitter() {
  if (saveNtuple_) {
    file_->cd();
	if (h1z) h1z->Write();
    ftree_->Write();
    file_->Close();
  }
}


void BeamFitter::readEvent(const edm::Event& iEvent)
{

	edm::Handle<reco::TrackCollection> TrackCollection;
	iEvent.getByLabel(tracksLabel_, TrackCollection);
	
	const reco::TrackCollection *tracks = TrackCollection.product();

	
	for ( reco::TrackCollection::const_iterator track = tracks->begin();
		  track != tracks->end();
		  ++track ) {

		
		if ( ! isMuon_) {
		
			const reco::HitPattern& trkHP = track->hitPattern();

			fnPixelLayerMeas = trkHP.pixelLayersWithMeasurement();
			fnStripLayerMeas = trkHP.stripLayersWithMeasurement();
			fnTotLayerMeas = trkHP.trackerLayersWithMeasurement();
			fnPXBLayerMeas = trkHP.pixelBarrelLayersWithMeasurement();
			fnPXFLayerMeas = trkHP.pixelEndcapLayersWithMeasurement();
			fnTIBLayerMeas = trkHP.stripTIBLayersWithMeasurement();
			fnTIDLayerMeas = trkHP.stripTIDLayersWithMeasurement();
			fnTOBLayerMeas = trkHP.stripTOBLayersWithMeasurement();
			fnTECLayerMeas = trkHP.stripTECLayersWithMeasurement();
		} else {

			fnTotLayerMeas = track->numberOfValidHits();

		}
	
		fpt = track->pt();
		feta = track->eta();
		fphi0 = track->phi();
		fcharge = track->charge();
		fnormchi2 = track->normalizedChi2();
    
		fd0 = track->d0();
		fsigmad0 = track->d0Error();
		fz0 = track->dz();
		fsigmaz0 = track->dzError();
		ftheta = track->theta();
		fvx = track->vx();
		fvy = track->vy();

		for (int i=0; i<5; ++i) {
			for (int j=0; j<5; ++j) {
				fcov[i][j] = track->covariance(i,j);
			}
		}

//     if (debug_) {
//       std::cout << "pt= " << fpt << " eta= " << feta << " fd0= " << fd0 << " sigmad0= " << fsigmad0;
//       std::cout << " track quality = "  << track->qualityMask();
//       std::cout << " track algorithm = " << track->algoName() << std::endl;
//     }

    // Track quality
    bool quality_ok=true;
	bool algo_ok = true;
	
	if (! isMuon_ ) {
		if (quality_.size()!=0) {
			quality_ok = false;
			for (unsigned int i = 0; i<quality_.size();++i) {
				if(debug_) std::cout << "quality_[" << i << "] = " << track->qualityName(quality_[i]) << std::endl;
				if (track->quality(quality_[i])) {
					quality_ok = true;
					break;
				}
			}
		}
		
    
		// Track algorithm
		
		if (algorithm_.size()!=0) {
			if (std::find(algorithm_.begin(),algorithm_.end(),track->algo())==algorithm_.end())
				algo_ok = false;
		}

	}
    
    if (saveNtuple_) ftree_->Fill();
    ftotal_tracks++;
    
    // Final track selection
    if (fnTotLayerMeas >= trk_MinNTotLayers_
        && fnPixelLayerMeas >= trk_MinNPixLayers_
		&& fnormchi2 < trk_MaxNormChi2_
		&& fpt > trk_MinpT_
		&& algo_ok
		&& quality_ok
		&& std::abs( fd0 ) < trk_MaxIP_
		&& std::abs( fz0 ) < trk_MaxZ_
		&& std::abs( feta ) < trk_MaxEta_
        ) {
		if (debug_){
			std::cout << "Selected track quality = " << track->qualityMask();
			std::cout << "; track algorithm = " << track->algoName() << "= TrackAlgorithm: " << track->algo() << std::endl;
		}
		BSTrkParameters BSTrk(fz0,fsigmaz0,fd0,fsigmad0,fphi0,fpt,0.,0.);
		BSTrk.setVx(fvx);
		BSTrk.setVy(fvy);
		fBSvector.push_back(BSTrk);
    }

  }

}

bool BeamFitter::runFitter() {
  bool fit_ok = false;
  // default fit to extract beam spot info
  if(fBSvector.size() > 1 ){
    if(debug_){
      std::cout << "Calculating beam spot..." << std::endl;
      std::cout << "We will use " << fBSvector.size() << " good tracks out of " << ftotal_tracks << std::endl;
    }
    BSFitter *myalgo = new BSFitter( fBSvector );
	myalgo->SetMaximumZ( trk_MaxZ_ );
	myalgo->SetConvergence( convergence_ );
	myalgo->SetMinimumNTrks(min_Ntrks_);

    fbeamspot = myalgo->Fit();

    if(writeTxt_) dumpTxtFile();

	// retrieve histogram for Vz
 	h1z = (TH1F*) myalgo->GetVzHisto();
		
    delete myalgo;
    if ( fbeamspot.type() != 0 ) // not Fake
      fit_ok = true;
  }
  else{
    fbeamspot.setType(reco::BeamSpot::Fake);
    if(debug_) std::cout << "Not enough good tracks selected! No beam fit!" << std::endl;
  }
  return fit_ok;
}

void BeamFitter::dumpTxtFile(){
  fasciiFile << "X " << fbeamspot.x0() << std::endl;
  fasciiFile << "Y " << fbeamspot.y0() << std::endl;
  fasciiFile << "Z " << fbeamspot.z0() << std::endl;
  fasciiFile << "sigmaZ " << fbeamspot.sigmaZ() << std::endl;
  fasciiFile << "dxdz " << fbeamspot.dxdz() << std::endl;
  fasciiFile << "dydz " << fbeamspot.dydz() << std::endl;
  if (inputBeamWidth_ > 0 ) {
    fasciiFile << "BeamWidthX " << inputBeamWidth_ << std::endl;
    fasciiFile << "BeamWidthY " << inputBeamWidth_ << std::endl;
  } else {
    fasciiFile << "BeamWidthX " << fbeamspot.BeamWidthX() << std::endl;
    fasciiFile << "BeamWidthY " << fbeamspot.BeamWidthY() << std::endl;
  }
	
  for (int i = 0; i<6; ++i) {
    fasciiFile << "Cov("<<i<<",j) ";
    for (int j=0; j<7; ++j) {
      fasciiFile << fbeamspot.covariance(i,j) << " ";
    }
    fasciiFile << std::endl;
  }
  // beam width error
  if (inputBeamWidth_ > 0 ) {
    fasciiFile << "Cov(6,j) 0 0 0 0 0 0 " << pow(2.e-4,2) << std::endl;
  } else {
    fasciiFile << "Cov(6,j) 0 0 0 0 0 0 " << fbeamspot.covariance(6,6) << std::endl;
  }
  fasciiFile << "EmittanceX " << fbeamspot.emittanceX() << std::endl;
  fasciiFile << "EmittanceY " << fbeamspot.emittanceY() << std::endl;
  fasciiFile << "BetaStar " << fbeamspot.betaStar() << std::endl;
}

void BeamFitter::write2DB(){
  BeamSpotObjects *pBSObjects = new BeamSpotObjects();
  
  pBSObjects->SetPosition(fbeamspot.position().X(),fbeamspot.position().Y(),fbeamspot.position().Z());
  //std::cout << " wrote: x= " << fbeamspot.position().X() << " y= "<< fbeamspot.position().Y() << " z= " << fbeamspot.position().Z() << std::endl;
  pBSObjects->SetSigmaZ(fbeamspot.sigmaZ());
  pBSObjects->Setdxdz(fbeamspot.dxdz());
  pBSObjects->Setdydz(fbeamspot.dydz());
  if (inputBeamWidth_ > 0 ) {
    std::cout << " beam width value forced to be " << inputBeamWidth_ << std::endl;
    pBSObjects->SetBeamWidthX(inputBeamWidth_);
    pBSObjects->SetBeamWidthY(inputBeamWidth_);
  } else {
    // need to fix this
    std::cout << " using default value, 15e-4, for beam width!!!"<<std::endl;
    pBSObjects->SetBeamWidthX(15.0e-4);
    pBSObjects->SetBeamWidthY(15.0e-4);
    
  }
		
  for (int i = 0; i<7; ++i) {
    for (int j=0; j<7; ++j) {
      pBSObjects->SetCovariance(i,j,fbeamspot.covariance(i,j));
    }
  }
  edm::Service<cond::service::PoolDBOutputService> poolDbService;
  if( poolDbService.isAvailable() ) {
    std::cout << "poolDBService available"<<std::endl;
    if ( poolDbService->isNewTagRequest( "BeamSpotObjectsRcd" ) ) {
      std::cout << "new tag requested" << std::endl;
      poolDbService->createNewIOV<BeamSpotObjects>( pBSObjects, poolDbService->beginOfTime(),poolDbService->endOfTime(),
						    "BeamSpotObjectsRcd"  );
    }
    else {
      std::cout << "no new tag requested" << std::endl;
      poolDbService->appendSinceTime<BeamSpotObjects>( pBSObjects, poolDbService->currentTime(),
						       "BeamSpotObjectsRcd" );
    }
  }
}

void BeamFitter::runAllFitter() {
  if(fBSvector.size()!=0){
    BSFitter *myalgo = new BSFitter( fBSvector );    
    fbeamspot = myalgo->Fit_d0phi();
        
    // iterative
    if(debug_)
      std::cout << " d0-phi Iterative:" << std::endl;
    BSFitter *myitealgo = new BSFitter( fBSvector );
    myitealgo->Setd0Cut_d0phi(4.0);
    reco::BeamSpot beam_ite = myitealgo->Fit_ited0phi();
    if (debug_){
      std::cout << beam_ite << std::endl;
      std::cout << "\n Now run tests of the different fits\n";
    }
    // from here are just tests
    std::string fit_type = "chi2";
    myalgo->SetFitVariable(std::string("z"));
    myalgo->SetFitType(std::string("chi2"));
    reco::BeamSpot beam_fit_z_chi2 = myalgo->Fit();
    if (debug_){
      std::cout << " z Chi2 Fit ONLY:" << std::endl;
      std::cout << beam_fit_z_chi2 << std::endl;
    }
    
    fit_type = "combined";
    myalgo->SetFitVariable("z");
    myalgo->SetFitType("combined");
    reco::BeamSpot beam_fit_z_lh = myalgo->Fit();
    if (debug_){
      std::cout << " z Log-Likelihood Fit ONLY:" << std::endl;
      std::cout << beam_fit_z_lh << std::endl;
    }
    
    myalgo->SetFitVariable("d");
    myalgo->SetFitType("d0phi");
    reco::BeamSpot beam_fit_dphi = myalgo->Fit();
    if (debug_){
      std::cout << " d0-phi0 Fit ONLY:" << std::endl;
      std::cout << beam_fit_dphi << std::endl;
    }
    
    myalgo->SetFitVariable(std::string("d*z"));
    myalgo->SetFitType(std::string("likelihood"));
    reco::BeamSpot beam_fit_dz_lh = myalgo->Fit();
    if (debug_){
      std::cout << " Log-Likelihood Fit:" << std::endl;
      std::cout << beam_fit_dz_lh << std::endl;
    }
    
    myalgo->SetFitVariable(std::string("d*z"));
    myalgo->SetFitType(std::string("resolution"));
    reco::BeamSpot beam_fit_dresz_lh = myalgo->Fit();
    if(debug_){
      std::cout << " IP Resolution Fit" << std::endl;
      std::cout << beam_fit_dresz_lh << std::endl;
      
      std::cout << "c0 = " << myalgo->GetResPar0() << " +- " << myalgo->GetResPar0Err() << std::endl;
      std::cout << "c1 = " << myalgo->GetResPar1() << " +- " << myalgo->GetResPar1Err() << std::endl;
    }
  }
  else
    if (debug_) std::cout << "No good track selected! No beam fit!" << std::endl;
}
