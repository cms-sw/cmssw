#include <iostream>
//
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
//
#include "Validation/RecoEgamma/plugins/ElectronConversionRejectionValidator.h"

//
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
//

//
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "RecoEgamma/EgammaTools/interface/ConversionTools.h"
#include "CommonTools/Statistics/interface/ChiSquaredProbability.h"

//
//
#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TTree.h"
#include "TVector3.h"
#include "TProfile.h"
#include "TMath.h"
//
/** \class ElectronConversionRejectionValidator
 **
 **
 **  $Id: ElectronConversionRejectionValidator
 **  \author J.Bendavid
 **
 ***/

using namespace std;


ElectronConversionRejectionValidator::ElectronConversionRejectionValidator( const edm::ParameterSet& pset )
{

  fName_     = pset.getUntrackedParameter<std::string>("Name");
  verbosity_ = pset.getUntrackedParameter<int>("Verbosity");
  parameters_ = pset;

  gsfElectronCollectionProducer_ = pset.getParameter<std::string>("gsfElectronProducer");
  gsfElectronCollection_ = pset.getParameter<std::string>("gsfElectronCollection");

  conversionCollectionProducer_ = pset.getParameter<std::string>("convProducer");
  conversionCollection_ = pset.getParameter<std::string>("conversionCollection");
  // conversionTrackProducer_ = pset.getParameter<std::string>("trackProducer");
  gsfElecToken_ = consumes<reco::GsfElectronCollection>(
      edm::InputTag(gsfElectronCollectionProducer_,
                    gsfElectronCollection_));
  convToken_ = consumes<reco::ConversionCollection>(
      edm::InputTag(conversionCollectionProducer_,
                    conversionCollection_));

  isRunCentrally_=   pset.getParameter<bool>("isRunCentrally");

  elePtMin_ = pset.getParameter<double>("elePtMin");
  eleExpectedHitsInnerMax_ = pset.getParameter<int>("eleExpectedHitsInnerMax");
  eleD0Max_ = pset.getParameter<double>("eleD0Max");
  offline_pvToken_ = consumes<reco::VertexCollection>(
      pset.getUntrackedParameter<edm::InputTag> ("offlinePV",
                                                 edm::InputTag("offlinePrimaryVertices")));
  beamspotToken_ = consumes<reco::BeamSpot>(
      pset.getUntrackedParameter<edm::InputTag> ("beamspot",
                                                 edm::InputTag("offlineBeamSpot")));

  nEvt_=0;
  nEntry_=0;
}

ElectronConversionRejectionValidator::~ElectronConversionRejectionValidator() {}


void ElectronConversionRejectionValidator::bookHistograms(DQMStore::IBooker & ibooker,
  edm::Run const &, edm::EventSetup const & ){

  double ptMin = parameters_.getParameter<double>("ptMin");
  double ptMax = parameters_.getParameter<double>("ptMax");
  int ptBin = parameters_.getParameter<int>("ptBin");

  double trackptMin = parameters_.getParameter<double>("trackptMin");
  double trackptMax = parameters_.getParameter<double>("trackptMax");
  int trackptBin = parameters_.getParameter<int>("trackptBin");

  double etaMin = parameters_.getParameter<double>("etaMin");
  double etaMax = parameters_.getParameter<double>("etaMax");
  int etaBin = parameters_.getParameter<int>("etaBin");

  double phiMin = -TMath::Pi();
  double phiMax =  TMath::Pi();
  int    phiBin = parameters_.getParameter<int>("phiBin");


  double rhoMin = parameters_.getParameter<double>("rhoMin");
  double rhoMax = parameters_.getParameter<double>("rhoMax");
  int    rhoBin = parameters_.getParameter<int>("rhoBin");

  double zMin = parameters_.getParameter<double>("zMin");
  double zMax = parameters_.getParameter<double>("zMax");
  int    zBin = parameters_.getParameter<int>("zBin");

  //// All MC photons
  // SC from reco photons

  //TString simfolder = TString(
  //std::string simpath = dqmpath_ + "SimulationInfo";
  ibooker.setCurrentFolder(dqmpath_);
  //
  // simulation information about conversions
  // Histograms for efficiencies
  h_elePtAll_ = ibooker.book1D("elePtAll", "# of Electrons", ptBin, ptMin, ptMax);
  h_eleEtaAll_ = ibooker.book1D("eleEtaAll", "# of Electrons", etaBin, etaMin, etaMax);
  h_elePhiAll_ = ibooker.book1D("elePhiAll", "# of Electrons", phiBin, phiMin, phiMax);

  h_elePtPass_ = ibooker.book1D("elePtPass", "# of Electrons", ptBin, ptMin, ptMax);
  h_eleEtaPass_ = ibooker.book1D("eleEtaPass", "# of Electrons", etaBin, etaMin, etaMax);
  h_elePhiPass_ = ibooker.book1D("elePhiPass", "# of Electrons", phiBin, phiMin, phiMax);

  h_elePtFail_ = ibooker.book1D("elePtFail", "# of Electrons", ptBin, ptMin, ptMax);
  h_eleEtaFail_ = ibooker.book1D("eleEtaFail", "# of Electrons",etaBin, etaMin, etaMax);
  h_elePhiFail_ = ibooker.book1D("elePhiFail", "# of Electrons", phiBin, phiMin, phiMax);

  h_convPt_ = ibooker.book1D("convPt", "# of Electrons", ptBin, ptMin, ptMax);
  h_convEta_ = ibooker.book1D("convEta", "# of Electrons", etaBin, etaMin, etaMax);
  h_convPhi_ = ibooker.book1D("convPhi", "# of Electrons", phiBin, phiMin, phiMax);
  h_convRho_ = ibooker.book1D("convRho", "# of Electrons", rhoBin, rhoMin, rhoMax);
  h_convZ_ = ibooker.book1D("convZ", "# of Electrons", zBin, zMin, zMax);
  h_convProb_ = ibooker.book1D("convProb", "# of Electrons", 100, 0.0, 1.0);

  h_convLeadTrackpt_ = ibooker.book1D("convLeadTrackpt", "# of Electrons", trackptBin,
      trackptMin, trackptMax);

  h_convTrailTrackpt_ = ibooker.book1D("convTrailTrackpt", "# of Electrons", trackptBin,
      trackptMin, trackptMax);

  h_convLog10TrailTrackpt_ = ibooker.book1D("convLog10TrailTrackpt", "# of Electrons",
      ptBin, -2.0, 3.0);

  h_convLeadTrackAlgo_ = ibooker.book1D("convLeadTrackAlgo", "# of Electrons", 31, -0.5, 30.5);
  h_convTrailTrackAlgo_ = ibooker.book1D("convLeadTrackAlgo", "# of Electrons", 31, -0.5, 30.5);
}

void ElectronConversionRejectionValidator::analyze( const edm::Event& e, const edm::EventSetup& esup ) {

  using namespace edm;


  nEvt_++;
  LogInfo("ElectronConversionRejectionValidator")
      << "ElectronConversionRejectionValidator Analyzing event number: "
      << e.id() << " Global Counter " << nEvt_ <<"\n";

  ///// Get the recontructed  conversions
  Handle<reco::ConversionCollection> convHandle;
  e.getByToken(convToken_, convHandle);
  if (!convHandle.isValid()) {
    edm::LogError("ElectronConversionRejectionValidator")
        << "Error! Can't get the Conversion collection "<< std::endl;
    return;
  }

  ///// Get the recontructed  photons
  Handle<reco::GsfElectronCollection> gsfElectronHandle;
  e.getByToken(gsfElecToken_, gsfElectronHandle);
  const reco::GsfElectronCollection &gsfElectronCollection = *(gsfElectronHandle.product());
  if (!gsfElectronHandle.isValid()) {
    edm::LogError("ElectronConversionRejectionValidator")
        << "Error! Can't get the Electron collection "<< std::endl;
    return;
  }

  // offline  Primary vertex
  edm::Handle<reco::VertexCollection> vertexHandle;
  e.getByToken(offline_pvToken_, vertexHandle);
  if (!vertexHandle.isValid()) {
    edm::LogError("ElectronConversionRejectionValidator")
        << "Error! Can't get the product primary Vertex Collection "<< "\n";
    return;
  }
  const reco::Vertex &thevtx = vertexHandle->at(0);

  edm::Handle<reco::BeamSpot> bsHandle;
  e.getByToken(beamspotToken_, bsHandle);
  if (!bsHandle.isValid()) {
    edm::LogError("ElectronConversionRejectionValidator")
        << "Error! Can't get the product beamspot Collection "<< "\n";
    return;
  }
  const reco::BeamSpot &thebs = *bsHandle.product();


  //loop over electrons
  for (reco::GsfElectronCollection::const_iterator iele = gsfElectronCollection.begin(); iele!=gsfElectronCollection.end(); ++iele) {
    //apply basic pre-selection cuts to remove the conversions with obviously displaced tracks which will anyways be
    //removed from the analysis by the hit pattern or impact parameter requirements
    if (iele->pt() < elePtMin_) continue;
    if (iele->gsfTrack()->hitPattern().numberOfHits(reco::HitPattern::MISSING_INNER_HITS) > eleExpectedHitsInnerMax_) continue;
    if ( std::abs(iele->gsfTrack()->dxy(thevtx.position())) > eleD0Max_ ) continue;

    //fill information for all electrons
    h_elePtAll_->Fill(iele->pt());
    h_eleEtaAll_->Fill(iele->eta());
    h_elePhiAll_->Fill(iele->phi());


    //find matching conversion if any
    reco::ConversionRef convref = ConversionTools::matchedConversion(*iele,convHandle,thebs.position());
    //fill information on passing electrons only if there is no matching conversion (electron passed the conversion rejection cut!)
    if (convref.isNull()) {
      h_elePtPass_->Fill(iele->pt());
      h_eleEtaPass_->Fill(iele->eta());
      h_elePhiPass_->Fill(iele->phi());
    }
    else {
      //matching conversion, electron failed conversion rejection cut
      //fill information on electron and matching conversion
      //(Note that in case of multiple matching conversions passing the requirements, the conversion tools returns the one closest to the IP,
      //which is most likely to be the conversion of the primary photon in case there was one.)

      //fill electron info
      h_elePtFail_->Fill(iele->pt());
      h_eleEtaFail_->Fill(iele->eta());
      h_elePhiFail_->Fill(iele->phi());

      //fill conversion info
      math::XYZVectorF convmom = convref->refittedPairMomentum();
      h_convPt_->Fill(convmom.rho());
      h_convEta_->Fill(convmom.eta());
      h_convPhi_->Fill(convmom.phi());
      h_convRho_->Fill(convref->conversionVertex().position().rho());
      h_convZ_->Fill(convref->conversionVertex().position().z());
      h_convProb_->Fill(ChiSquaredProbability(convref->conversionVertex().chi2(),convref->conversionVertex().ndof()));

      //fill information about conversion tracks
      if (convref->tracks().size()<2) continue;

      RefToBase<reco::Track> tk1 = convref->tracks().front();
      RefToBase<reco::Track> tk2 = convref->tracks().back();

      RefToBase<reco::Track> tklead;
      RefToBase<reco::Track> tktrail;
      if (tk1->pt() >= tk2->pt()) {
        tklead = tk1;
        tktrail = tk2;
      }
      else {
        tklead = tk2;
        tktrail = tk1;
      }
      h_convLeadTrackpt_->Fill(tklead->pt());
      h_convTrailTrackpt_->Fill(tktrail->pt());
      h_convLog10TrailTrackpt_->Fill(log10(tktrail->pt()));
      h_convLeadTrackAlgo_->Fill(tklead->algo());
      h_convTrailTrackAlgo_->Fill(tktrail->algo());

    }
  }

}
