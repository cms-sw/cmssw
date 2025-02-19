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
#include "DataFormats/EgammaCandidates/interface/ConversionFwd.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
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
 **  $Date: 2011/07/29 03:00:20 $
 **  $Revision: 1.1 $
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

    isRunCentrally_=   pset.getParameter<bool>("isRunCentrally");
    
    elePtMin_ = pset.getParameter<double>("elePtMin");
    eleExpectedHitsInnerMax_ = pset.getParameter<int>("eleExpectedHitsInnerMax");
    eleD0Max_ = pset.getParameter<double>("eleD0Max");
    
  }





ElectronConversionRejectionValidator::~ElectronConversionRejectionValidator() {}




void  ElectronConversionRejectionValidator::beginJob() {

  nEvt_=0;
  nEntry_=0;


  dbe_ = 0;
  dbe_ = edm::Service<DQMStore>().operator->();


  double ptMin = parameters_.getParameter<double>("ptMin");
  double ptMax = parameters_.getParameter<double>("ptMax");
  int ptBin = parameters_.getParameter<int>("ptBin");

  double trackptMin = parameters_.getParameter<double>("trackptMin");
  double trackptMax = parameters_.getParameter<double>("trackptMax");
  int trackptBin = parameters_.getParameter<int>("trackptBin");  

//   double resMin = parameters_.getParameter<double>("resMin");
//   double resMax = parameters_.getParameter<double>("resMax");
//   int resBin = parameters_.getParameter<int>("resBin");

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

//   double dPhiTracksMin = parameters_.getParameter<double>("dPhiTracksMin");
//   double dPhiTracksMax = parameters_.getParameter<double>("dPhiTracksMax");
//   int dPhiTracksBin = parameters_.getParameter<int>("dPhiTracksBin");

//   double eoverpMin = parameters_.getParameter<double>("eoverpMin");
//   double eoverpMax = parameters_.getParameter<double>("eoverpMax");
//   int    eoverpBin = parameters_.getParameter<int>("eoverpBin");


  //  double dEtaTracksMin = parameters_.getParameter<double>("dEtaTracksMin");  // unused
  //  double dEtaTracksMax = parameters_.getParameter<double>("dEtaTracksMax"); // unused
  //  int    dEtaTracksBin = parameters_.getParameter<int>("dEtaTracksBin");  // unused

//   double dCotTracksMin = parameters_.getParameter<double>("dCotTracksMin");
//   double dCotTracksMax = parameters_.getParameter<double>("dCotTracksMax");
//   int    dCotTracksBin = parameters_.getParameter<int>("dCotTracksBin");


  if (dbe_) {

    //// All MC photons
    // SC from reco photons

    //TString simfolder = TString(
    //std::string simpath = dqmpath_ + "SimulationInfo";
    dbe_->setCurrentFolder(dqmpath_);
    //
    // simulation information about conversions
    // Histograms for efficiencies
    h_elePtAll_ = dbe_->book1D("elePtAll","# of Electrons",ptBin,ptMin,ptMax);
    h_eleEtaAll_ = dbe_->book1D("eleEtaAll","# of Electrons",etaBin,etaMin,etaMax);
    h_elePhiAll_ = dbe_->book1D("elePhiAll","# of Electrons",phiBin,phiMin,phiMax);
    
    h_elePtPass_ = dbe_->book1D("elePtPass","# of Electrons",ptBin,ptMin,ptMax);
    h_eleEtaPass_ = dbe_->book1D("eleEtaPass","# of Electrons",etaBin,etaMin,etaMax);
    h_elePhiPass_ = dbe_->book1D("elePhiPass","# of Electrons",phiBin,phiMin,phiMax);

    h_elePtFail_ = dbe_->book1D("elePtFail","# of Electrons",ptBin,ptMin,ptMax);
    h_eleEtaFail_ = dbe_->book1D("eleEtaFail","# of Electrons",etaBin,etaMin,etaMax);
    h_elePhiFail_ = dbe_->book1D("elePhiFail","# of Electrons",phiBin,phiMin,phiMax);    
    
    h_convPt_ = dbe_->book1D("convPt","# of Electrons",ptBin,ptMin,ptMax);
    h_convEta_ = dbe_->book1D("convEta","# of Electrons",etaBin,etaMin,etaMax);
    h_convPhi_ = dbe_->book1D("convPhi","# of Electrons",phiBin,phiMin,phiMax); 
    h_convRho_ = dbe_->book1D("convRho","# of Electrons",rhoBin,rhoMin,rhoMax);        
    h_convZ_ = dbe_->book1D("convZ","# of Electrons",zBin,zMin,zMax);        
    h_convProb_ = dbe_->book1D("convProb","# of Electrons",100,0.0,1.0);    

    h_convLeadTrackpt_ = dbe_->book1D("convLeadTrackpt","# of Electrons",trackptBin,trackptMin,trackptMax);
    h_convTrailTrackpt_ = dbe_->book1D("convTrailTrackpt","# of Electrons",trackptBin,trackptMin,trackptMax);
    h_convLog10TrailTrackpt_ = dbe_->book1D("convLog10TrailTrackpt","# of Electrons",ptBin,-2.0,3.0);

    h_convLeadTrackAlgo_ = dbe_->book1D("convLeadTrackAlgo","# of Electrons",31,-0.5,30.5);
    h_convTrailTrackAlgo_ = dbe_->book1D("convLeadTrackAlgo","# of Electrons",31,-0.5,30.5);

    
  } // if DQM



}



 void  ElectronConversionRejectionValidator::beginRun (edm::Run const & r, edm::EventSetup const & theEventSetup) {


}

void  ElectronConversionRejectionValidator::endRun (edm::Run& r, edm::EventSetup const & theEventSetup) {


}



void ElectronConversionRejectionValidator::analyze( const edm::Event& e, const edm::EventSetup& esup ) {

  using namespace edm;


  nEvt_++;
  LogInfo("ElectronConversionRejectionValidator") << "ElectronConversionRejectionValidator Analyzing event number: " << e.id() << " Global Counter " << nEvt_ <<"\n";
  //  std::cout << "ElectronConversionRejectionValidator Analyzing event number: "  << e.id() << " Global Counter " << nEvt_ <<"\n";



  ///// Get the recontructed  conversions
  Handle<reco::ConversionCollection> convHandle;
  e.getByLabel(conversionCollectionProducer_, conversionCollection_ , convHandle);
  if (!convHandle.isValid()) {
    edm::LogError("ElectronConversionRejectionValidator") << "Error! Can't get the Conversion collection "<< std::endl;
    return;
  }

  ///// Get the recontructed  photons
  Handle<reco::GsfElectronCollection> gsfElectronHandle;
  e.getByLabel(gsfElectronCollectionProducer_, gsfElectronCollection_ , gsfElectronHandle);
  const reco::GsfElectronCollection &gsfElectronCollection = *(gsfElectronHandle.product());
  if (!gsfElectronHandle.isValid()) {
    edm::LogError("ElectronConversionRejectionValidator") << "Error! Can't get the Electron collection "<< std::endl;
    return;
  }

  // offline  Primary vertex
  edm::Handle<reco::VertexCollection> vertexHandle;
  e.getByLabel("offlinePrimaryVertices", vertexHandle);
  if (!vertexHandle.isValid()) {
      edm::LogError("ElectronConversionRejectionValidator") << "Error! Can't get the product primary Vertex Collection "<< "\n";
      return;
  }
  const reco::Vertex &thevtx = vertexHandle->at(0);
  
  edm::Handle<reco::BeamSpot> bsHandle;
  e.getByLabel("offlineBeamSpot", bsHandle);
  if (!bsHandle.isValid()) {
      edm::LogError("ElectronConversionRejectionValidator") << "Error! Can't get the product beamspot Collection "<< "\n";
      return;
  }
  const reco::BeamSpot &thebs = *bsHandle.product();

 
  //loop over electrons
  for (reco::GsfElectronCollection::const_iterator iele = gsfElectronCollection.begin(); iele!=gsfElectronCollection.end(); ++iele) {
    //apply basic pre-selection cuts to remove the conversions with obviously displaced tracks which will anyways be
    //removed from the analysis by the hit pattern or impact parameter requirements
    if (iele->pt() < elePtMin_) continue;
    if (iele->gsfTrack()->trackerExpectedHitsInner().numberOfHits() > eleExpectedHitsInnerMax_) continue;
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





void ElectronConversionRejectionValidator::endJob() {


  std::string outputFileName = parameters_.getParameter<std::string>("OutputFileName");
  if ( ! isRunCentrally_ ) {
    dbe_->save(outputFileName);
  }

  edm::LogInfo("ElectronConversionRejectionValidator") << "Analyzed " << nEvt_  << "\n";
  // std::cout  << "::endJob Analyzed " << nEvt_ << " events " << " with total " << nPho_ << " Photons " << "\n";
  //  std::cout  << "ElectronConversionRejectionValidator::endJob Analyzed " << nEvt_ << " events " << "\n";

  return ;
}


