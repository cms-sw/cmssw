/**_________________________________________________________________
   class:   BeamFitter.cc
   package: RecoVertex/BeamSpotProducer
   


 author: Francisco Yumiceva, Fermilab (yumiceva@fnal.gov)
         Geng-Yuan Jeng, UCRiverside

 version $Id: BeamFitter.cc,v 1.0 2009/03/26 20:04:12 yumiceva Exp $

 ________________________________________________________________**/

#include "RecoVertex/BeamSpotProducer/interface/BeamFitter.h"


#include "FWCore/ParameterSet/interface/InputTag.h"

#include "DataFormats/TrackCandidate/interface/TrackCandidate.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"


BeamFitter::BeamFitter(const edm::ParameterSet& iConfig)
{

  tracksLabel_       = iConfig.getParameter<edm::ParameterSet>("BeamFitter").getUntrackedParameter<edm::InputTag>("TrackCollection");
  writeTxt_          = iConfig.getParameter<edm::ParameterSet>("BeamFitter").getUntrackedParameter<bool>("WriteAscii");
  outputTxt_         = iConfig.getParameter<edm::ParameterSet>("BeamFitter").getUntrackedParameter<std::string>("AsciiFileName");

  trk_MinpT_         = iConfig.getParameter<edm::ParameterSet>("BeamFitter").getUntrackedParameter<double>("MinimumPt");
  trk_MaxEta_        = iConfig.getParameter<edm::ParameterSet>("BeamFitter").getUntrackedParameter<double>("MaximumEta");
  trk_MinNTotLayers_ = iConfig.getParameter<edm::ParameterSet>("BeamFitter").getUntrackedParameter<int>("MinimumTotalLayers");
  trk_MinNPixLayers_ = iConfig.getParameter<edm::ParameterSet>("BeamFitter").getUntrackedParameter<int>("MinimumPixelLayers");
  trk_MinNormChi2_   = iConfig.getParameter<edm::ParameterSet>("BeamFitter").getUntrackedParameter<double>("MinimumNormChi2");
  trk_Algorithm_     = iConfig.getParameter<edm::ParameterSet>("BeamFitter").getUntrackedParameter<int>("TrackAlgorithm");

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

    
    double pt = track->pt();
    double eta = track->eta();
    double phi0 = track->momentum().phi();
    double charge = track->charge();
    double chi2 = track->chi2();
    double ndof = track->ndof();
      
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
      std::cout << "pt= "<< pt << " eta= " << eta << " fd0= " << d0 << " sigmad0= " << sigmad0 <<std::endl;

    }

    // track quality

    fBSvector.push_back(BSTrkParameters(z0,sigmaz0,d0,sigmad0,phi0,pt,0.,0.));
    

  }

}

void BeamFitter::runFitter() {

  // default fit to extract beam spot info
  BSFitter *myalgo = new BSFitter( fBSvector );
  fbeamspot = myalgo->Fit();

  delete myalgo;

}





