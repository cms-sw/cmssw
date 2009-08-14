/**_________________________________________________________________
   class:   BeamFitter.cc
   package: RecoVertex/BeamSpotProducer
   


 author: Francisco Yumiceva, Fermilab (yumiceva@fnal.gov)
         Geng-Yuan Jeng, UC Riverside (Geng-Yuan.Jeng@cern.ch)
 
 version $Id: BeamFitter.cc,v 1.1 2009/08/13 21:05:07 yumiceva Exp $

 ________________________________________________________________**/

#include "RecoVertex/BeamSpotProducer/interface/BeamFitter.h"


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

  trk_MinpT_         = iConfig.getParameter<edm::ParameterSet>("BeamFitter").getUntrackedParameter<double>("MinimumPt");
  trk_MaxEta_        = iConfig.getParameter<edm::ParameterSet>("BeamFitter").getUntrackedParameter<double>("MaximumEta");
  trk_MinNTotLayers_ = iConfig.getParameter<edm::ParameterSet>("BeamFitter").getUntrackedParameter<int>("MinimumTotalLayers");
  trk_MinNPixLayers_ = iConfig.getParameter<edm::ParameterSet>("BeamFitter").getUntrackedParameter<int>("MinimumPixelLayers");
  trk_MaxNormChi2_   = iConfig.getParameter<edm::ParameterSet>("BeamFitter").getUntrackedParameter<double>("MaximumNormChi2");
  trk_Algorithm_     = iConfig.getParameter<edm::ParameterSet>("BeamFitter").getUntrackedParameter<std::vector<std::string> >("TrackAlgorithm");
  trk_Quality_       = iConfig.getParameter<edm::ParameterSet>("BeamFitter").getUntrackedParameter<std::vector<std::string> >("TrackQuality");

  for (unsigned int j=0;j<trk_Algorithm_.size();j++) 
    algorithm_.push_back(reco::TrackBase::algoByName(trk_Algorithm_[j]));
  for (unsigned int j=0;j<trk_Quality_.size();j++) 
    quality_.push_back(reco::TrackBase::qualityByName(trk_Quality_[j]));

  ftotal_tracks = 0;
}

BeamFitter::~BeamFitter() {}


void BeamFitter::readEvent(const edm::Event& iEvent)
{

  edm::Handle<reco::TrackCollection> TrackCollection;
  iEvent.getByLabel(tracksLabel_, TrackCollection);

  const reco::TrackCollection *tracks = TrackCollection.product();

  for ( reco::TrackCollection::const_iterator track = tracks->begin();
	track != tracks->end();
	++track ) {
    const reco::HitPattern& trkHP = track->hitPattern();

    int nPxLayerMeas = trkHP.pixelLayersWithMeasurement();
    int nTotLayerMeas = trkHP.trackerLayersWithMeasurement();
    
    double pt = track->pt();
    double eta = track->eta();
    double phi0 = track->momentum().phi();
    double charge = track->charge();
    double normchi2 = track->normalizedChi2();
    
    double d0 = track->d0();
    double sigmad0 = track->d0Error();
    double z0 = track->dz();
    double sigmaz0 = track->dzError();
    double theta = track->theta();

    double cov[7][7];

    for (int i=0; i<5; ++i) {
      for (int j=0; j<5; ++j) {
	cov[i][j] = track->covariance(i,j);
      }
    }

    if (debug_) {
      std::cout << "pt= " << pt << " eta= " << eta << " fd0= " << d0 << " sigmad0= " << sigmad0;
      std::cout << " track quality = "  << track->qualityMask();
      std::cout << " track algorithm = " << track->algoName() << std::endl;
    }

    // Track quality
    bool quality_ok=true;
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
    
    // Tracl algorithm
    bool algo_ok = true;
    if (algorithm_.size()!=0) {
      if (std::find(algorithm_.begin(),algorithm_.end(),track->algo())==algorithm_.end())
	algo_ok = false;
    }
    
    ftotal_tracks++;
    
    // Final track selection
    if (nTotLayerMeas >= trk_MinNTotLayers_
        && nPxLayerMeas >= trk_MinNPixLayers_
	&& normchi2 < trk_MaxNormChi2_
	&& pt > trk_MinpT_
 	&& algo_ok
 	&& quality_ok
        ) {
      if (debug_){
	std::cout << "Selected track quality = " << track->qualityMask();
	std::cout << "; track algorithm = " << track->algoName() << "= TrackAlgorithm: " << track->algo() << std::endl;
      }
      fBSvector.push_back(BSTrkParameters(z0,sigmaz0,d0,sigmad0,phi0,pt,0.,0.));
    }

  }

}

bool BeamFitter::runFitter() {
  bool fit_ok = false;
  // default fit to extract beam spot info
  if(fBSvector.size()!=0){
    std::cout << "Calculating beam spot..." << std::endl;
    std::cout << "We will use " << fBSvector.size() << " good tracks out of " << ftotal_tracks << std::endl;
    
    BSFitter *myalgo = new BSFitter( fBSvector );
    fbeamspot = myalgo->Fit();
    
    delete myalgo;
    fit_ok = true;
  }
  else
    std::cout << "No good track selected! No beam fit!" << std::endl;
  return fit_ok;
}

void BeamFitter::runAllFitter() {
  if(fBSvector.size()!=0){
    BSFitter *myalgo = new BSFitter( fBSvector );    
    fbeamspot = myalgo->Fit_d0phi();
    
    
    // iterative
    std::cout << " d0-phi Iterative:" << std::endl;
    BSFitter *myitealgo = new BSFitter( fBSvector );
    myitealgo->Setd0Cut_d0phi(4.0);
    reco::BeamSpot beam_ite = myitealgo->Fit_ited0phi();
    std::cout << beam_ite << std::endl;
    
    
    std::cout << "\n Now run tests of the different fits\n";
    // from here are just tests
    std::string fit_type = "chi2";
    myalgo->SetFitVariable(std::string("z"));
    myalgo->SetFitType(std::string("chi2"));
    reco::BeamSpot beam_fit_z_chi2 = myalgo->Fit();
    std::cout << " z Chi2 Fit ONLY:" << std::endl;
    std::cout << beam_fit_z_chi2 << std::endl;
    
    
    fit_type = "combined";
    myalgo->SetFitVariable("z");
    myalgo->SetFitType("combined");
    reco::BeamSpot beam_fit_z_lh = myalgo->Fit();
    std::cout << " z Log-Likelihood Fit ONLY:" << std::endl;
    std::cout << beam_fit_z_lh << std::endl;
    
    
    myalgo->SetFitVariable("d");
    myalgo->SetFitType("d0phi");
    reco::BeamSpot beam_fit_dphi = myalgo->Fit();
    std::cout << " d0-phi0 Fit ONLY:" << std::endl;
    std::cout << beam_fit_dphi << std::endl;
    
    
//     myalgo->SetFitVariable(std::string("d*z"));
//     myalgo->SetFitType(std::string("likelihood"));
//     reco::BeamSpot beam_fit_dz_lh = myalgo->Fit();
//     std::cout << " Log-Likelihood Fit:" << std::endl;
//     std::cout << beam_fit_dz_lh << std::endl;
    
    
//     myalgo->SetFitVariable(std::string("d*z"));
//     myalgo->SetFitType(std::string("resolution"));
//     reco::BeamSpot beam_fit_dresz_lh = myalgo->Fit();
//     std::cout << " IP Resolution Fit" << std::endl;
//     std::cout << beam_fit_dresz_lh << std::endl;
    
//     std::cout << "c0 = " << myalgo->GetResPar0() << " +- " << myalgo->GetResPar0Err() << std::endl;
//     std::cout << "c1 = " << myalgo->GetResPar1() << " +- " << myalgo->GetResPar1Err() << std::endl;
    
  }
  else
    std::cout << "No good track selected! No beam fit!" << std::endl;
}
