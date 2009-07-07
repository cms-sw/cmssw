#include <iostream>
//
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"
//
#include "Validation/RecoEgamma/interface/PhotonValidator.h"

//
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
//
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
//
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertexContainer.h"
//
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"
#include "SimTracker/Records/interface/TrackAssociatorRecord.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
//
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/PatternTools/interface/TwoTrackMinimumDistance.h"
#include "TrackingTools/TransientTrack/interface/TrackTransientTrack.h"
//
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CommonTools/Statistics/interface/ChiSquaredProbability.h"

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
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
//
#include "RecoEgamma/EgammaMCTools/interface/PhotonMCTruthFinder.h"
#include "RecoEgamma/EgammaMCTools/interface/PhotonMCTruth.h"
#include "RecoEgamma/EgammaMCTools/interface/ElectronMCTruth.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"
#include "RecoCaloTools/MetaCollections/interface/CaloRecHitMetaCollections.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
//
//#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
//
#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TTree.h"
#include "TVector3.h"
#include "TProfile.h"
// 
/** \class PhotonValidator
 **  
 **
 **  $Id: PhotonValidator
 **  $Date: 2009/05/12 18:11:47 $ 
 **  $Revision: 1.28 $
 **  \author Nancy Marinelli, U. of Notre Dame, US
 **
 ***/

using namespace std;

 
PhotonValidator::PhotonValidator( const edm::ParameterSet& pset )
  {

    fName_     = pset.getUntrackedParameter<std::string>("Name");
    verbosity_ = pset.getUntrackedParameter<int>("Verbosity");
    parameters_ = pset;

    
    photonCollectionProducer_ = pset.getParameter<std::string>("phoProducer");
    photonCollection_ = pset.getParameter<std::string>("photonCollection");

   
    label_tp_   = pset.getParameter<edm::InputTag>("label_tp");

    barrelEcalHits_   = pset.getParameter<edm::InputTag>("barrelEcalHits");
    endcapEcalHits_   = pset.getParameter<edm::InputTag>("endcapEcalHits");

    conversionOITrackProducer_ = pset.getParameter<std::string>("conversionOITrackProducer");
    conversionIOTrackProducer_ = pset.getParameter<std::string>("conversionIOTrackProducer");



    minPhoEtCut_ = pset.getParameter<double>("minPhoEtCut");   
    convTrackMinPtCut_ = pset.getParameter<double>("convTrackMinPtCut");   


    trkIsolExtRadius_ = pset.getParameter<double>("trkIsolExtR");   
    trkIsolInnRadius_ = pset.getParameter<double>("trkIsolInnR");   
    trkPtLow_     = pset.getParameter<double>("minTrackPtCut");   
    lip_       = pset.getParameter<double>("lipCut");   
    ecalIsolRadius_ = pset.getParameter<double>("ecalIsolR");   
    bcEtLow_     = pset.getParameter<double>("minBcEtCut");   
    hcalIsolExtRadius_ = pset.getParameter<double>("hcalIsolExtR");   
    hcalIsolInnRadius_ = pset.getParameter<double>("hcalIsolInnR");   
    hcalHitEtLow_     = pset.getParameter<double>("minHcalHitEtCut");   

    numOfTracksInCone_ = pset.getParameter<int>("maxNumOfTracksInCone");   
    trkPtSumCut_  = pset.getParameter<double>("trkPtSumCut");   
    ecalEtSumCut_ = pset.getParameter<double>("ecalEtSumCut");   
    hcalEtSumCut_ = pset.getParameter<double>("hcalEtSumCut");   
    dCotCutOn_ = pset.getParameter<bool>("dCotCutOn");   
    dCotCutValue_ = pset.getParameter<double>("dCotCutValue");   
    dCotHardCutValue_ = pset.getParameter<double>("dCotHardCutValue");   
    signal_ = pset.getParameter<bool>("signal");

    thePhotonMCTruthFinder_ = new PhotonMCTruthFinder();
   

  }





PhotonValidator::~PhotonValidator() {
  
  delete thePhotonMCTruthFinder_;


}




void PhotonValidator::initVectors() { 



  double etaMin = parameters_.getParameter<double>("etaMin");
  double etaMax = parameters_.getParameter<double>("etaMax");
  int etaBin = parameters_.getParameter<int>("etaBin");
  int etaBin2 = parameters_.getParameter<int>("etaBin2");


 
  double phiMin = parameters_.getParameter<double>("phiMin");
  double phiMax = parameters_.getParameter<double>("phiMax");
  int    phiBin = parameters_.getParameter<int>("phiBin");



  double rMin = parameters_.getParameter<double>("rMin");
  double rMax = parameters_.getParameter<double>("rMax");
  int    rBin = parameters_.getParameter<int>("rBin");

  double zMin = parameters_.getParameter<double>("zMin");
  double zMax = parameters_.getParameter<double>("zMax");
  int    zBin = parameters_.getParameter<int>("zBin");
 

  double etMin = parameters_.getParameter<double>("etMin");
  double etMax = parameters_.getParameter<double>("etMax");
  int etBin = parameters_.getParameter<int>("etBin");

 
  double step=(etaMax-etaMin)/etaBin;
  etaintervals_.push_back(etaMin);
    for (int k=1;k<etaBin+1;k++) {
      double d=etaMin+k*step;
      etaintervals_.push_back(d);
      totSimPhoEta_.push_back(0);     
      totMatchedSimPhoEta_.push_back(0);     
    }   

    step=(etaMax-etaMin)/etaBin2;
    etaintervalslarge_.push_back(etaMin);
    for (int k=1;k<etaBin2+1;k++) {
      double d=etaMin+k*step;
      etaintervalslarge_.push_back(d);
      totSimConvEta_.push_back(0);     
      totMatchedSimPhoEtaSmallR9_.push_back(0);
      totMatchedSimConvEtaTwoTracks_.push_back(0); 
      totMatchedConvEtaTwoTracksWithVtx_.push_back(0);
      totMatchedConvEtaTwoTracksWithVtxProbGT0_.push_back(0);
      totMatchedConvEtaTwoTracksWithVtxProbGT005_.push_back(0);
      totMatchedConvEtaTwoTracksWithVtxProbGT01_.push_back(0);
      totMatchedSimConvEtaAllTwoTracks_.push_back(0);
      totMatchedRecConvEtaTwoTracks_.push_back(0);     
      totRecAssConvEtaTwoTracks_.push_back(0);     
      
      totMatchedSimConvEtaOneTrack_.push_back(0);      
    }   
    



    step=(phiMax-phiMin)/phiBin;
    phiintervals_.push_back(phiMin);
    for (int k=1;k<phiBin+1;k++) {
      double d=phiMin+k*step;
      phiintervals_.push_back(d);
      totSimPhoPhi_.push_back(0);     
      totMatchedSimPhoPhi_.push_back(0);     
      //
      totSimConvPhi_.push_back(0);     
      totMatchedSimConvPhiTwoTracks_.push_back(0);     
      totMatchedRecConvPhiTwoTracks_.push_back(0);     
      totRecAssConvPhiTwoTracks_.push_back(0);     
    }   
    step=(rMax-rMin)/rBin;
    rintervals_.push_back(rMin);
    for (int k=1;k<rBin+1;k++) {
      double d=rMin+k*step;
      rintervals_.push_back(d);
      totSimConvR_.push_back(0);     
      totMatchedSimConvRTwoTracks_.push_back(0);    
      totMatchedSimConvRAllTwoTracks_.push_back(0);    
      totMatchedRecConvRTwoTracks_.push_back(0);     
      totRecAssConvRTwoTracks_.push_back(0);     
      
      totMatchedSimConvROneTrack_.push_back(0);      
    }

   
    step=(zMax-zMin)/zBin;
    zintervals_.push_back(zMin);
    for (int k=1;k<zBin+1;k++) {
      double d=zMin+k*step;
      zintervals_.push_back(d);
      totSimConvZ_.push_back(0);     
      totMatchedSimConvZTwoTracks_.push_back(0);  
      totMatchedRecConvZTwoTracks_.push_back(0);     
      totRecAssConvZTwoTracks_.push_back(0);     
    
    }   

    step=(etMax-etMin)/etBin;
    etintervals_.push_back(etMin);
    for (int k=1;k<etBin+1;k++) {
      double d=etMin+k*step;
      etintervals_.push_back(d);
      totSimPhoEt_.push_back(0);     
      totMatchedSimPhoEt_.push_back(0);     
      totSimConvEt_.push_back(0);     
      totMatchedSimConvEtTwoTracks_.push_back(0);  
      totMatchedRecConvEtTwoTracks_.push_back(0);     
      totRecAssConvEtTwoTracks_.push_back(0); 
      totMatchedSimConvEtOneTrack_.push_back(0); 
    }   
    



}


void  PhotonValidator::beginJob() {

  nEvt_=0;
  nEntry_=0;
  nRecConv_=0;
  nRecConvAss_=0;
  nRecConvAssWithEcal_=0;
   
  nInvalidPCA_=0;
  
  dbe_ = 0;
  dbe_ = edm::Service<DQMStore>().operator->();
  



  double resMin = parameters_.getParameter<double>("resMin");
  double resMax = parameters_.getParameter<double>("resMax");
  int resBin = parameters_.getParameter<int>("resBin");

  double eMin = parameters_.getParameter<double>("eMin");
  double eMax = parameters_.getParameter<double>("eMax");
  int eBin = parameters_.getParameter<int>("eBin");

  double etMin = parameters_.getParameter<double>("etMin");
  double etMax = parameters_.getParameter<double>("etMax");
  int etBin = parameters_.getParameter<int>("etBin");

  double etScale = parameters_.getParameter<double>("etScale");

  double etaMin = parameters_.getParameter<double>("etaMin");
  double etaMax = parameters_.getParameter<double>("etaMax");
  int etaBin = parameters_.getParameter<int>("etaBin");
  int etaBin2 = parameters_.getParameter<int>("etaBin2");

  double dEtaMin = parameters_.getParameter<double>("dEtaMin");
  double dEtaMax = parameters_.getParameter<double>("dEtaMax");
  int dEtaBin = parameters_.getParameter<int>("dEtaBin");
 
  double phiMin = parameters_.getParameter<double>("phiMin");
  double phiMax = parameters_.getParameter<double>("phiMax");
  int    phiBin = parameters_.getParameter<int>("phiBin");

  double dPhiMin = parameters_.getParameter<double>("dPhiMin");
  double dPhiMax = parameters_.getParameter<double>("dPhiMax");
  int    dPhiBin = parameters_.getParameter<int>("dPhiBin");

  double rMin = parameters_.getParameter<double>("rMin");
  double rMax = parameters_.getParameter<double>("rMax");
  int    rBin = parameters_.getParameter<int>("rBin");

  double zMin = parameters_.getParameter<double>("zMin");
  double zMax = parameters_.getParameter<double>("zMax");
  int    zBin = parameters_.getParameter<int>("zBin");
 

 
  double r9Min = parameters_.getParameter<double>("r9Min"); 
  double r9Max = parameters_.getParameter<double>("r9Max"); 
  int r9Bin = parameters_.getParameter<int>("r9Bin");

  double dPhiTracksMin = parameters_.getParameter<double>("dPhiTracksMin"); 
  double dPhiTracksMax = parameters_.getParameter<double>("dPhiTracksMax"); 
  int dPhiTracksBin = parameters_.getParameter<int>("dPhiTracksBin"); 
  
  double dEtaTracksMin = parameters_.getParameter<double>("dEtaTracksMin"); 
  double dEtaTracksMax = parameters_.getParameter<double>("dEtaTracksMax"); 
  int    dEtaTracksBin = parameters_.getParameter<int>("dEtaTracksBin"); 

  double dCotTracksMin = parameters_.getParameter<double>("dCotTracksMin"); 
  double dCotTracksMax = parameters_.getParameter<double>("dCotTracksMax"); 
  int    dCotTracksBin = parameters_.getParameter<int>("dCotTracksBin"); 


  double povereMin = parameters_.getParameter<double>("povereMin");
  double povereMax = parameters_.getParameter<double>("povereMax");
  int povereBin = parameters_.getParameter<int>("povereBin");

  double eoverpMin = parameters_.getParameter<double>("eoverpMin"); 
  double eoverpMax = parameters_.getParameter<double>("eoverpMax"); 
  int    eoverpBin = parameters_.getParameter<int>("eoverpBin"); 

  double chi2Min = parameters_.getParameter<double>("chi2Min"); 
  double chi2Max = parameters_.getParameter<double>("chi2Max"); 

  int    ggMassBin = parameters_.getParameter<int>("ggMassBin");   
  double ggMassMin = parameters_.getParameter<double>("ggMassMin"); 
  double ggMassMax = parameters_.getParameter<double>("ggMassMax"); 


  double rMinForXray = parameters_.getParameter<double>("rMinForXray");
  double rMaxForXray = parameters_.getParameter<double>("rMaxForXray");
  int    rBinForXray = parameters_.getParameter<int>("rBinForXray");
  double zMinForXray = parameters_.getParameter<double>("zMinForXray");
  double zMaxForXray = parameters_.getParameter<double>("zMaxForXray");
  int    zBinForXray = parameters_.getParameter<int>("zBinForXray");
  int    zBin2ForXray = parameters_.getParameter<int>("zBin2ForXray");


  //////// set up vectors
  initVectors();



  if (dbe_) {  
    //// All MC photons
    // SC from reco photons

    dbe_->setCurrentFolder("Egamma/PhotonValidator/SimulationInfo");
    //
  // simulation information about all MC photons found 
    std::string histname = "nOfSimPhotons";    
    h_nSimPho_[0] = dbe_->book1D(histname,"# of Sim photons per event ",20,-0.5,19.5);
    histname = "SimPhoE";    
    h_SimPhoE_[0] = dbe_->book1D(histname,"Sim photon energy spectrum",eBin,eMin,eMax);
    histname = "SimPhoEt";    
    h_SimPhoEt_[0] = dbe_->book1D(histname,"Sim photon tranverse energy spectrum",etBin,etMin,etMax);
    h_SimPhoEta_[0] = dbe_->book1D("SimPhoEta"," Sim Photon Eta ",etaBin,etaMin, etaMax) ;
    
    h_SimPhoPhi_[0] = dbe_->book1D("SimPhoPhi"," Sim Photon  Phi ",phiBin,phiMin,phiMax) ;
    histname = "SimPhoMotherEt";    
    h_SimPhoMotherEt_[0] = dbe_->book1D(histname,"Sim photon Mother tranverse energy spectrum",etBin,etMin,etMax);
    h_SimPhoMotherEta_[0] = dbe_->book1D("SimPhoMotherEta"," Sim Photon Mother Eta ",etaBin,etaMin, etaMax) ;

    // simulation information about those MC photons matched by a reco Photon 
    histname = "nOfSimPhotonsMatched";    
    h_nSimPho_[1] = dbe_->book1D(histname,"# of Sim photons matched by a reco Photon per event ",20,-0.5,19.5);
    histname = "SimPhoEMatched";    
    h_SimPhoE_[1] = dbe_->book1D(histname,"Sim photon matched by a reco Photon: energy spectrum",eBin,eMin,eMax);
    histname = "SimPhoEtMatched";    
    h_SimPhoEt_[1] = dbe_->book1D(histname,"Sim photon matched by a reco Photon: tranverse energy spectrum",etBin,etMin,etMax);
    h_SimPhoEta_[1] = dbe_->book1D("SimPhoEtaMatched"," Sim Photon matched by a reco Photon: Eta ",etaBin,etaMin, etaMax) ;
    
    h_SimPhoPhi_[1] = dbe_->book1D("SimPhoPhiMatched"," Sim Photon  matched by a reco Photon: Phi ",phiBin,phiMin,phiMax) ;
    histname = "SimPhoMotherEtMatched";    
    h_SimPhoMotherEt_[1] = dbe_->book1D(histname,"Sim photon  matched by a reco Photon: Mother tranverse energy spectrum",etBin,etMin,etMax);
    h_SimPhoMotherEta_[1] = dbe_->book1D("SimPhoMotherEtaMatched"," Sim Photon matched by a reco Photon:  Mother Eta ",etaBin,etaMin, etaMax) ;


    //
    histname = "nOfSimConversions";    
    h_nSimConv_[0] = dbe_->book1D(histname,"# of Sim conversions per event ",20,-0.5,19.5);
    histname = "SimConvE";    
    h_SimConvE_[0]= dbe_->book1D(histname,"Sim conversions energy spectrum",eBin,eMin,eMax);
    histname = "SimConvEt";    
    h_SimConvEt_[0] = dbe_->book1D(histname,"Sim conversion tranverse energy spectrum",etBin,etMin,etMax);
    h_SimConvEta_[0] = dbe_->book1D("SimConvEta"," Sim Conversion Eta ",etaBin,etaMin, etaMax) ;



    h_SimConvPhi_[0] = dbe_->book1D("SimConvPhi"," Sim Conversion  Phi ",phiBin,phiMin,phiMax) ;
    h_SimConvR_[0] = dbe_->book1D("SimConvR"," Sim Conversion Radius ",rBin,rMin,rMax) ;
    h_SimConvZ_[0] = dbe_->book1D("SimConvZ"," Sim Conversion Z ",zBin,zMin,zMax) ;
    //   
    histname = "nOfVisSimConversions";    
    h_nSimConv_[1] = dbe_->book1D(histname,"# of Sim conversions per event ",20,-0.5,19.5);
    histname = "VisSimConvE";    
    h_SimConvE_[1]= dbe_->book1D(histname,"Sim conversions energy spectrum",eBin,eMin,eMax);
    histname = "VisSimConvEt";    
    h_SimConvEt_[1] = dbe_->book1D(histname,"Visible Sim conversion tranverse energy spectrum",etBin,etMin,etMax);
    h_SimConvEta_[1] = dbe_->book1D("VisSimConvEta"," Visible Sim Conversion Eta ",etaBin,etaMin, etaMax) ;

    h_SimConvEtaPix_[0] = dbe_->book1D("simConvEtaPix"," sim converted Photon Eta: Pix ",etaBin,etaMin, etaMax) ;


    h_SimConvPhi_[1] = dbe_->book1D("VisSimConvPhi"," Visible Sim Conversion  Phi ",phiBin,phiMin,phiMax) ;
    h_SimConvR_[1] = dbe_->book1D("VisSimConvR"," Visible Sim Conversion Radius ",rBin,rMin,rMax) ;
    h_SimConvZ_[1] = dbe_->book1D("VisSimConvZ"," Visible Sim Conversion Z ",zBin,zMin,zMax) ;
    h_simTkPt_ = dbe_->book1D("simTkPt","Sim conversion tracks pt ",etBin*3,0.,etMax);
    h_simTkEta_ = dbe_->book1D("simTkEta","Sim conversion tracks eta ",etaBin,etaMin,etaMax);
   

    dbe_->setCurrentFolder("Egamma/PhotonValidator/Photons");

    //// Reconstructed photons
    histname = "recoEffVsEta";
    phoEffEta_ =  dbe_->book1D(histname,"Photon reconstruction efficiency vs simulated #eta",etaBin,etaMin, etaMax);
    histname = "recoEffVsPhi";
    phoEffPhi_ =  dbe_->book1D(histname,"Photon reconstruction efficiency vs simulated #phi",phiBin,phiMin, phiMax);
    histname = "recoEffVsEt";
    phoEffEt_ =  dbe_->book1D(histname,"Photon reconstruction efficiency vs simulated Et",etBin,etMin, etMax) ;


    h_phoEta_[0] = dbe_->book1D("phoEta"," Photon Eta ",etaBin,etaMin, etaMax) ;
    h_phoPhi_[0] = dbe_->book1D("phoPhi"," Photon  Phi ",phiBin,phiMin,phiMax) ;

    h_phoDEta_[0] = dbe_->book1D("phoDEta"," Photon Eta(rec)-Eta(true) ",dEtaBin,dEtaMin, dEtaMax) ;
    h_phoDPhi_[0] = dbe_->book1D("phoDPhi"," Photon  Phi(rec)-Phi(true) ",dPhiBin,dPhiMin,dPhiMax) ;

    h_scEta_[0] =   dbe_->book1D("scEta"," SC Eta ",etaBin,etaMin, etaMax);
    h_scPhi_[0] =   dbe_->book1D("scPhi"," SC Phi ",phiBin,phiMin,phiMax);

    h_scEtaWidth_[0] =   dbe_->book1D("scEtaWidth"," SC Eta Width ",100,0., 0.1);
    h_scPhiWidth_[0] =   dbe_->book1D("scPhiWidth"," SC Phi Width ",100,0., 1.);


    histname = "scE";
    h_scE_[0][0] = dbe_->book1D(histname+"All"," SC Energy: All Ecal  ",eBin,eMin, eMax);
    h_scE_[0][1] = dbe_->book1D(histname+"Barrel"," SC Energy: Barrel ",eBin,eMin, eMax);
    h_scE_[0][2] = dbe_->book1D(histname+"Endcap"," SC Energy: Endcap ",eBin,eMin, eMax);

    histname = "psE";
    h_psE_ = dbe_->book1D(histname+"Endcap"," ES Energy  ",eBin,eMin, 50.);


    histname = "scEt";
    h_scEt_[0][0] = dbe_->book1D(histname+"All"," SC Et: All Ecal ",etBin,etMin, etMax) ;
    h_scEt_[0][1] = dbe_->book1D(histname+"Barrel"," SC Et: Barrel",etBin,etMin, etMax) ;
    h_scEt_[0][2] = dbe_->book1D(histname+"Endcap"," SC Et: Endcap",etBin,etMin, etMax) ;

    histname = "r9";
    h_r9_[0][0] = dbe_->book1D(histname+"All",   " r9: All Ecal",r9Bin,r9Min, r9Max) ;
    h_r9_[0][1] = dbe_->book1D(histname+"Barrel"," r9: Barrel ",r9Bin,r9Min, r9Max) ;
    h_r9_[0][2] = dbe_->book1D(histname+"Endcap"," r9: Endcap ",r9Bin,r9Min, r9Max) ;
    //
    histname = "r9ConvFromMC";
    h_r9_[1][0] = dbe_->book1D(histname+"All",   " r9: All Ecal",r9Bin,r9Min, r9Max) ;
    h_r9_[1][1] = dbe_->book1D(histname+"Barrel"," r9: Barrel ",r9Bin,r9Min, r9Max) ;
    h_r9_[1][2] = dbe_->book1D(histname+"Endcap"," r9: Endcap ",r9Bin,r9Min, r9Max) ;
    //
    histname = "r9ConvFromReco";
    h_r9_[2][0] = dbe_->book1D(histname+"All",   " r9: All Ecal",r9Bin,r9Min, r9Max) ;
    h_r9_[2][1] = dbe_->book1D(histname+"Barrel"," r9: Barrel ",r9Bin,r9Min, r9Max) ;
    h_r9_[2][2] = dbe_->book1D(histname+"Endcap"," r9: Endcap ",r9Bin,r9Min, r9Max) ;
    //
    histname="R9VsEta";
    h2_r9VsEta_[0] = dbe_->book2D(histname+"All"," All photons r9 vs #eta: all Ecal ",etaBin2,etaMin, etaMax,100, 0.,1.1);
    h2_r9VsEta_[1] = dbe_->book2D(histname+"Unconv"," All photons r9 vs #eta: all Ecal ",etaBin2,etaMin, etaMax,100, 0.,1.1);
    //
    histname="R9VsEt";
    h2_r9VsEt_[0] = dbe_->book2D(histname+"All"," All photons r9 vs Et: all Ecal ",etBin,etMin, etMax,100, 0.,1.1);
    h2_r9VsEt_[1] = dbe_->book2D(histname+"Unconv"," All photons r9 vs Et: all Ecal ",etBin,etMin, etMax,100, 0.,1.1);
    //
    histname = "r1";
    h_r1_[0][0] = dbe_->book1D(histname+"All",   " e1x5/e5x5: All Ecal",r9Bin,r9Min, r9Max) ;
    h_r1_[0][1] = dbe_->book1D(histname+"Barrel"," e1x5/e5x5: Barrel ",r9Bin,r9Min, r9Max) ;
    h_r1_[0][2] = dbe_->book1D(histname+"Endcap"," e1x5/e5x5: Endcap ",r9Bin,r9Min, r9Max) ;
    //
    histname="R1VsEta";
    h2_r1VsEta_[0] = dbe_->book2D(histname+"All"," All photons e1x5/e5x5 vs #eta: all Ecal ",etaBin2,etaMin, etaMax,100, 0.,1.1);
    h2_r1VsEta_[1] = dbe_->book2D(histname+"Unconv"," All photons e1x5/e5x5 vs #eta: all Ecal ",etaBin2,etaMin, etaMax,100, 0.,1.1);
    //
    histname="R1VsEt";
    h2_r1VsEt_[0] = dbe_->book2D(histname+"All"," All photons e1x5/e5x5 vs Et: all Ecal ",etBin,etMin, etMax,100, 0.,1.1);
    h2_r1VsEt_[1] = dbe_->book2D(histname+"Unconv"," All photons e1x5/e5x5 vs Et: all Ecal ",etBin,etMin, etMax,100, 0.,1.1);
    //
    histname = "r2";
    h_r2_[0][0] = dbe_->book1D(histname+"All",   " e2x5/e5x5: All Ecal",r9Bin,r9Min, r9Max) ;
    h_r2_[0][1] = dbe_->book1D(histname+"Barrel"," e2x5/e5x5: Barrel ",r9Bin,r9Min, r9Max) ;
    h_r2_[0][2] = dbe_->book1D(histname+"Endcap"," e2x5/e5x5: Endcap ",r9Bin,r9Min, r9Max) ;
    //
    histname="R2VsEta";
    h2_r2VsEta_[0] = dbe_->book2D(histname+"All"," All photons e2x5/e5x5 vs #eta: all Ecal ",etaBin2,etaMin, etaMax,100, 0.,1.1);
    h2_r2VsEta_[1] = dbe_->book2D(histname+"Unconv"," All photons e2x5/e5x5 vs #eta: all Ecal ",etaBin2,etaMin, etaMax,100, 0.,1.1);
    //
    histname="R2VsEt";
    h2_r2VsEt_[0] = dbe_->book2D(histname+"All"," All photons e2x5/e5x5 vs Et: all Ecal ",etBin,etMin, etMax,100, 0.,1.1);
    h2_r2VsEt_[1] = dbe_->book2D(histname+"Unconv"," All photons e2x5/e5x5 vs Et: all Ecal ",etBin,etMin, etMax,100, 0.,1.1);
    //
    histname = "sigmaIetaIeta";
    h_sigmaIetaIeta_[0][0] = dbe_->book1D(histname+"All",   "sigmaIetaIeta: All Ecal",100,0., 0.1) ;
    h_sigmaIetaIeta_[0][1] = dbe_->book1D(histname+"Barrel","sigmaIetaIeta: Barrel ", 100,0., 0.1) ;
    h_sigmaIetaIeta_[0][2] = dbe_->book1D(histname+"Endcap","sigmaIetaIeta: Endcap ", 100,0., 0.1) ;
    //
    histname="sigmaIetaIetaVsEta";
    h2_sigmaIetaIetaVsEta_[0] = dbe_->book2D(histname+"All"," All photons sigmaIetaIeta vs #eta: all Ecal ",  etaBin2,etaMin, etaMax,100, 0.,0.1);
    h2_sigmaIetaIetaVsEta_[1] = dbe_->book2D(histname+"Unconv"," All photons sigmaIetaIeta vs #eta: all Ecal ",etaBin2,etaMin, etaMax,100, 0.,0.1);
    //
    histname="sigmaIetaIetaVsEt";
    h2_sigmaIetaIetaVsEt_[0] = dbe_->book2D(histname+"All"," All photons sigmaIetaIeta vs Et: all Ecal ",etBin,etMin, etMax,100, 0.,0.1);
    h2_sigmaIetaIetaVsEt_[1] = dbe_->book2D(histname+"Unconv"," All photons sigmaIetaIeta vs Et: all Ecal ",etBin,etMin, etMax,100, 0.,0.1);
    //
    histname = "hOverE";
    h_hOverE_[0][0] = dbe_->book1D(histname+"All",   "H/E: All Ecal",100,0., 0.1) ;
    h_hOverE_[0][1] = dbe_->book1D(histname+"Barrel","H/E: Barrel ", 100,0., 0.1) ;
    h_hOverE_[0][2] = dbe_->book1D(histname+"Endcap","H/E: Endcap ", 100,0., 0.1) ;
    //
    histname="hOverEVsEta";
    h2_hOverEVsEta_[0] = dbe_->book2D(histname+"All"," All photons H/E vs #eta: all Ecal ",  etaBin2,etaMin, etaMax,100, 0.,0.1);
    h2_hOverEVsEta_[1] = dbe_->book2D(histname+"Unconv"," All photons H/E vs #eta: all Ecal ",etaBin2,etaMin, etaMax,100, 0.,0.1);
    //
    histname="hOverEVsEt";
    h2_hOverEVsEt_[0] = dbe_->book2D(histname+"All"," All photons H/E vs Et: all Ecal ",etBin,etMin, etMax,100, 0.,0.1);
    h2_hOverEVsEt_[1] = dbe_->book2D(histname+"Unconv"," All photons H/E vs Et: all Ecal ",etBin,etMin, etMax,100, 0.,0.1);
    //
    histname = "ecalRecHitSumEtConeDR04";
    h_ecalRecHitSumEtConeDR04_[0][0] = dbe_->book1D(histname+"All",   "ecalRecHitSumEtDR04: All Ecal",etBin,etMin,etMax*etScale);
    h_ecalRecHitSumEtConeDR04_[0][1] = dbe_->book1D(histname+"Barrel","ecalRecHitSumEtDR04: Barrel ", etBin,etMin,etMax*etScale);
    h_ecalRecHitSumEtConeDR04_[0][2] = dbe_->book1D(histname+"Endcap","ecalRecHitSumEtDR04: Endcap ", etBin,etMin,etMax*etScale);
    //
    histname="ecalRecHitSumEtConeDR04VsEta";
    h2_ecalRecHitSumEtConeDR04VsEta_[0] = dbe_->book2D(histname+"All"," All photons ecalRecHitSumEtDR04 vs #eta: all Ecal ",  etaBin2,etaMin, etaMax, etBin,etMin,etMax*etScale);
    h2_ecalRecHitSumEtConeDR04VsEta_[1] = dbe_->book2D(histname+"Unconv"," All photons ecalRecHitSumEtDR04 vs #eta: all Ecal ",etaBin2,etaMin, etaMax,etBin,etMin,etMax*etScale);
    //
    histname="ecalRecHitSumEtConeDR04VsEt";
    h2_ecalRecHitSumEtConeDR04VsEt_[0] = dbe_->book2D(histname+"All"," All photons ecalRecHitSumEtDR04 vs Et: all Ecal ",etBin,etMin, etMax, etBin,etMin,etMax*etScale);
    h2_ecalRecHitSumEtConeDR04VsEt_[1] = dbe_->book2D(histname+"Barrel"," All photons ecalRecHitSumEtDR04 vs Et: Barrel ",etBin,etMin, etMax, etBin,etMin,etMax*etScale);
    h2_ecalRecHitSumEtConeDR04VsEt_[2] = dbe_->book2D(histname+"Endcap"," All photons ecalRecHitSumEtDR04 vs Et: Endcap ",etBin,etMin, etMax, etBin,etMin,etMax*etScale);
    //
    histname = "hcalTowerSumEtConeDR04";
    h_hcalTowerSumEtConeDR04_[0][0] = dbe_->book1D(histname+"All",   "hcalTowerSumEtConeDR04: All Ecal",etBin,etMin,etMax*0.1);
    h_hcalTowerSumEtConeDR04_[0][1] = dbe_->book1D(histname+"Barrel","hcalTowerSumEtConeDR04: Barrel ", etBin,etMin,etMax*0.1);
    h_hcalTowerSumEtConeDR04_[0][2] = dbe_->book1D(histname+"Endcap","hcalTowerSumEtConeDR04: Endcap ", etBin,etMin,etMax*0.1);
    //
    histname="hcalTowerSumEtConeDR04VsEta";
    h2_hcalTowerSumEtConeDR04VsEta_[0] = dbe_->book2D(histname+"All"," All photons hcalTowerSumEtConeDR04 vs #eta: all Ecal ",  etaBin2,etaMin, etaMax, etBin,etMin,etMax*0.1);
    h2_hcalTowerSumEtConeDR04VsEta_[1] = dbe_->book2D(histname+"Unconv"," All photons hcalTowerSumEtConeDR04 vs #eta: all Ecal ",etaBin2,etaMin, etaMax,etBin,etMin,etMax*0.1);
    //
    histname="hcalTowerSumEtConeDR04VsEt";
    h2_hcalTowerSumEtConeDR04VsEt_[0] = dbe_->book2D(histname+"All"," All photons hcalTowerSumEtConeDR04 vs Et: all Ecal ",etBin,etMin, etMax, etBin,etMin,etMax*0.1);
    h2_hcalTowerSumEtConeDR04VsEt_[1] = dbe_->book2D(histname+"Barrel"," All photons hcalTowerSumEtConeDR04 vs Et: Barrel ",etBin,etMin, etMax,etBin,etMin,etMax*0.1);
    h2_hcalTowerSumEtConeDR04VsEt_[2] = dbe_->book2D(histname+"Endcap"," All photons hcalTowerSumEtConeDR04 vs Et: Endcap ",etBin,etMin, etMax,etBin,etMin,etMax*0.1);
    //
    histname = "isoTrkSolidConeDR04";
    h_isoTrkSolidConeDR04_[0][0] = dbe_->book1D(histname+"All",   "isoTrkSolidConeDR04: All Ecal",etBin,etMin,etMax*0.1);
    h_isoTrkSolidConeDR04_[0][1] = dbe_->book1D(histname+"Barrel","isoTrkSolidConeDR04: Barrel ", etBin,etMin,etMax*0.1);
    h_isoTrkSolidConeDR04_[0][2] = dbe_->book1D(histname+"Endcap","isoTrkSolidConeDR04: Endcap ", etBin,etMin,etMax*0.1);
    //
    histname="isoTrkSolidConeDR04VsEta";
    h2_isoTrkSolidConeDR04VsEta_[0] = dbe_->book2D(histname+"All"," All photons isoTrkSolidConeDR04 vs #eta: all Ecal ",  etaBin2,etaMin, etaMax, etBin,etMin,etMax*0.1);
    h2_isoTrkSolidConeDR04VsEta_[1] = dbe_->book2D(histname+"Unconv"," All photons isoTrkSolidConeDR04 vs #eta: all Ecal ",etaBin2,etaMin, etaMax,etBin,etMin,etMax*0.1);
    //
    histname="isoTrkSolidConeDR04VsEt";
    h2_isoTrkSolidConeDR04VsEt_[0] = dbe_->book2D(histname+"All"," All photons isoTrkSolidConeDR04 vs Et: all Ecal ",etBin,etMin, etMax, etBin,etMin,etMax*0.1);
    h2_isoTrkSolidConeDR04VsEt_[1] = dbe_->book2D(histname+"Unconv"," All photons isoTrkSolidConeDR04 vs Et: all Ecal ",etBin,etMin, etMax, etBin,etMin,etMax*0.1);
    //
    histname = "nTrkSolidConeDR04";
    h_nTrkSolidConeDR04_[0][0] = dbe_->book1D(histname+"All",   "nTrkSolidConeDR04: All Ecal",20,0., 20) ;
    h_nTrkSolidConeDR04_[0][1] = dbe_->book1D(histname+"Barrel","nTrkSolidConeDR04: Barrel ", 20,0., 20) ;
    h_nTrkSolidConeDR04_[0][2] = dbe_->book1D(histname+"Endcap","nTrkSolidConeDR04: Endcap ", 20,0., 20) ;
    //
    histname="nTrkSolidConeDR04VsEta";
    h2_nTrkSolidConeDR04VsEta_[0] = dbe_->book2D(histname+"All"," All photons nTrkSolidConeDR04 vs #eta: all Ecal ",  etaBin2,etaMin, etaMax, 20,0., 20) ;
    h2_nTrkSolidConeDR04VsEta_[1] = dbe_->book2D(histname+"Unconv"," All photons nTrkSolidConeDR04 vs #eta: all Ecal ",etaBin2,etaMin, etaMax,20,0., 20) ;
    //
    histname="nTrkSolidConeDR04VsEt";
    h2_nTrkSolidConeDR04VsEt_[0] = dbe_->book2D(histname+"All"," All photons nTrkSolidConeDR04 vs Et: all Ecal ",etBin,etMin, etMax, 20,0., 20) ;
    h2_nTrkSolidConeDR04VsEt_[1] = dbe_->book2D(histname+"Unconv"," All photons nTrkSolidConeDR04 vs Et: all Ecal ",etBin,etMin, etMax,20,0., 20) ;
    //
    histname = "phoE";
    h_phoE_[0][0]=dbe_->book1D(histname+"All"," Photon Energy: All ecal ", eBin,eMin, eMax);
    h_phoE_[0][1]=dbe_->book1D(histname+"Barrel"," Photon Energy: barrel ",eBin,eMin, eMax);
    h_phoE_[0][2]=dbe_->book1D(histname+"Endcap"," Photon Energy: Endcap ",eBin,eMin, eMax);

    histname = "phoEt";
    h_phoEt_[0][0] = dbe_->book1D(histname+"All"," Photon Transverse Energy: All ecal ", etBin,etMin, etMax);
    h_phoEt_[0][1] = dbe_->book1D(histname+"Barrel"," Photon Transverse Energy: Barrel ",etBin,etMin, etMax);
    h_phoEt_[0][2] = dbe_->book1D(histname+"Endcap"," Photon Transverse Energy: Endcap ",etBin,etMin, etMax);

    histname = "eRes";
    h_phoERes_[0][0] = dbe_->book1D(histname+"All"," Photon rec/true Energy: All ecal ", resBin,resMin, resMax);
    h_phoERes_[0][1] = dbe_->book1D(histname+"Barrel"," Photon rec/true Energy: Barrel ",resBin,resMin, resMax);
    h_phoERes_[0][2] = dbe_->book1D(histname+"Endcap"," Photon rec/true Energy: Endcap ",resBin,resMin, resMax);

    h_phoERes_[1][0] = dbe_->book1D(histname+"unconvAll"," Photon rec/true Energy if r9>0.93: All ecal ", resBin,resMin, resMax);
    h_phoERes_[1][1] = dbe_->book1D(histname+"unconvBarrel"," Photon rec/true Energy if r9>0.93: Barrel ",resBin,resMin, resMax);
    h_phoERes_[1][2] = dbe_->book1D(histname+"unconvEndcap"," Photon rec/true Energyif r9>0.93: Endcap ",resBin,resMin, resMax);

    h_phoERes_[2][0] = dbe_->book1D(histname+"convAll"," Photon rec/true Energy if r9<0.93: All ecal ", resBin,resMin, resMax);
    h_phoERes_[2][1] = dbe_->book1D(histname+"convBarrel"," Photon rec/true Energyif r9<0.93: Barrel ",resBin,resMin, resMax);
    h_phoERes_[2][2] = dbe_->book1D(histname+"convEndcap"," Photon rec/true Energyif r9<0.93: Endcap ",resBin,resMin, resMax);

    histname="eResVsEta";
    h2_eResVsEta_[0] = dbe_->book2D(histname+"All"," All photons E/Etrue vs #eta: all Ecal ",etaBin2,etaMin, etaMax,100, 0., 2.5);
    histname="pEResVsEta";
    p_eResVsEta_[0] = dbe_->book1D(histname+"All"," All photons  E/Etrue vs #eta: all Ecal ",etaBin2,etaMin, etaMax);
    histname="eResVsEta";
    h2_eResVsEta_[1] = dbe_->book2D(histname+"Unconv"," Unconv photons E/Etrue vs #eta: all Ecal ",etaBin2,etaMin, etaMax,100, 0., 2.5);
    histname="pEResVsEta";
    p_eResVsEta_[1] = dbe_->book1D(histname+"Unconv"," Unconv photons  E/Etrue vs #eta: all Ecal ",etaBin2,etaMin, etaMax);

    // Photon pair invariant mass
    histname = "gamgamMass"; 
    h_gamgamMass_[0][0] = dbe_->book1D(histname+"All","2 photons invariant mass: All ecal ", ggMassBin, ggMassMin, ggMassMax);
    h_gamgamMass_[0][1] = dbe_->book1D(histname+"Barrel","2 photons invariant mass:  Barrel ",ggMassBin, ggMassMin, ggMassMax);
    h_gamgamMass_[0][2] = dbe_->book1D(histname+"Endcap","2 photons invariant mass:  Endcap ",ggMassBin, ggMassMin, ggMassMax);
    //
    histname = "gamgamMassNoConv";
    h_gamgamMass_[1][0] = dbe_->book1D(histname+"All","2 photons with no conversion invariant mass: All ecal ",ggMassBin, ggMassMin, ggMassMax); 
    h_gamgamMass_[1][1] = dbe_->book1D(histname+"Barrel","2 photons with no conversion  invariant mass:  Barrel ",ggMassBin, ggMassMin, ggMassMax);
    h_gamgamMass_[1][2] = dbe_->book1D(histname+"Endcap","2 photons with no conversion  invariant mass:  Endcap ",ggMassBin, ggMassMin, ggMassMax);
    //
    histname = "gamgamMassConv";
    h_gamgamMass_[2][0] = dbe_->book1D(histname+"All","2 photons with conversion invariant mass: All ecal ", ggMassBin, ggMassMin, ggMassMax);
    h_gamgamMass_[2][1] = dbe_->book1D(histname+"Barrel","2 photons with  conversion  invariant mass:  Barrel ",ggMassBin, ggMassMin, ggMassMax);
    h_gamgamMass_[2][2] = dbe_->book1D(histname+"Endcap","2 photons with  conversion  invariant mass:  Endcap ",ggMassBin, ggMassMin, ggMassMax);


    dbe_->setCurrentFolder("Egamma/PhotonValidator/ConversionInfo");

    histname = "convEffVsEtaTwoTracks";
    convEffEtaTwoTracks_ =  dbe_->book1D(histname,histname,etaBin2,etaMin, etaMax);
    histname = "convEffVsEtaAllTwoTracks";
    convEffEtaAllTwoTracks_ =  dbe_->book1D(histname,histname,etaBin2,etaMin, etaMax);

    histname = "convEffVsEtaTwoTracksAndVtx";
    convEffEtaTwoTracksAndVtx_ =  dbe_->book1D(histname,histname,etaBin2,etaMin, etaMax);
    histname = "convEffVsEtaTwoTracksAndVtx2";
    convEffEtaTwoTracksAndVtx2_ =  dbe_->book1D(histname,"Conversion vertex fitting efficiency vs #eta",etaBin2,etaMin, etaMax);
    histname = "convEffVsEtaTwoTracksAndVtxProbGT0";
    convEffEtaTwoTracksAndVtxProbGT0_ =  dbe_->book1D(histname,histname,etaBin2,etaMin, etaMax);
    histname = "convEffVsEtaTwoTracksAndVtxProbGT005";
    convEffEtaTwoTracksAndVtxProbGT005_ =  dbe_->book1D(histname,histname,etaBin2,etaMin, etaMax);
    histname = "convEffVsEtaTwoTracksAndVtxProbGT01";
    convEffEtaTwoTracksAndVtxProbGT01_ =  dbe_->book1D(histname,histname,etaBin2,etaMin, etaMax);


    histname = "convEffVsEtTwoTracks";
    convEffEtTwoTracks_ =  dbe_->book1D(histname,histname,etBin,etMin, etMax);
    histname = "convEffVsPhiTwoTracks";
    convEffPhiTwoTracks_ =  dbe_->book1D(histname,histname,phiBin,phiMin,phiMax);
    histname = "convEffVsRTwoTracks";
    convEffRTwoTracks_ =  dbe_->book1D(histname,histname,rBin,rMin, rMax);
    histname = "convEffVsRAllTwoTracks";
    convEffRAllTwoTracks_ =  dbe_->book1D(histname,histname,rBin,rMin, rMax);
    histname = "convEffVsZTwoTracks";
    convEffZTwoTracks_ =  dbe_->book1D(histname,histname,zBin,zMin,zMax);

    histname = "convEffVsEtaOneTrack";
    convEffEtaOneTrack_ =  dbe_->book1D(histname,histname,etaBin2,etaMin, etaMax);
    histname = "convEffVsROneTrack";
    convEffROneTrack_ =  dbe_->book1D(histname,histname,rBin,rMin, rMax);
    histname = "convEffVsEtOneTrack";
    convEffEtOneTrack_ =  dbe_->book1D(histname,histname,etBin,etMin, etMax);


    histname = "convFakeRateVsEtaTwoTracks";
    convFakeRateEtaTwoTracks_ =  dbe_->book1D(histname,histname,etaBin2,etaMin, etaMax);
    histname = "convFakeRateVsPhiTwoTracks";
    convFakeRatePhiTwoTracks_ =  dbe_->book1D(histname,histname,phiBin,phiMin,phiMax);
    histname = "convFakeRateVsRTwoTracks";
    convFakeRateRTwoTracks_ =  dbe_->book1D(histname,histname,rBin,rMin, rMax);
    histname = "convFakeRateVsZTwoTracks";
    convFakeRateZTwoTracks_ =  dbe_->book1D(histname,histname,zBin,zMin,zMax);
    histname = "convFakeRateVsEtTwoTracks";
    convFakeRateEtTwoTracks_ =  dbe_->book1D(histname,histname,etBin,etMin, etMax);


    histname="nConv";
    h_nConv_[0][0] = dbe_->book1D(histname+"All","Number Of Conversions per isolated candidates per events: All Ecal  ",10,-0.5, 9.5);
    h_nConv_[0][1] = dbe_->book1D(histname+"Barrel","Number Of Conversions per isolated candidates per events: Ecal Barrel  ",10,-0.5, 9.5);
    h_nConv_[0][2] = dbe_->book1D(histname+"Endcap","Number Of Conversions per isolated candidates per events: Ecal Endcap ",10,-0.5, 9.5);

    h_convEta_[0] = dbe_->book1D("convEta1"," converted Photon Eta >1 track",etaBin,etaMin, etaMax) ;
    h_convEta_[1] = dbe_->book1D("convEta2"," converted Photon Eta =2 tracks ",etaBin,etaMin, etaMax) ;
    h_convEta_[2] = dbe_->book1D("convEta2ass"," converted Photon Eta =2 tracks, both ass ",etaBin,etaMin, etaMax) ;
    h_convPhi_[0] = dbe_->book1D("convPhi"," converted Photon  Phi ",phiBin,phiMin,phiMax) ;


    histname = "convERes";
    h_convERes_[0][0] = dbe_->book1D(histname+"All"," Conversion rec/true Energy: All ecal ", resBin,resMin, resMax);
    h_convERes_[0][1] = dbe_->book1D(histname+"Barrel"," Conversion rec/true Energy: Barrel ",resBin,resMin, resMax);
    h_convERes_[0][2] = dbe_->book1D(histname+"Endcap"," Conversion rec/true Energy: Endcap ",resBin,resMin, resMax);
    histname = "convPRes";
    h_convPRes_[1][0] = dbe_->book1D(histname+"All"," Conversion rec/true Energy: All ecal ", resBin,resMin, resMax);
    h_convPRes_[1][1] = dbe_->book1D(histname+"Barrel"," Conversion rec/true Energy: Barrel ",resBin,resMin, resMax);
    h_convPRes_[1][2] = dbe_->book1D(histname+"Endcap"," Conversion rec/true Energy: Endcap ",resBin,resMin, resMax);



    histname="r9VsTracks";
    h_r9VsNofTracks_[0][0] = dbe_->book2D(histname+"All"," photons r9 vs nTracks from conversions: All Ecal",r9Bin,r9Min, r9Max, 3, -0.5, 2.5) ;
    h_r9VsNofTracks_[0][1] = dbe_->book2D(histname+"Barrel"," photons r9 vs nTracks from conversions: Barrel Ecal",r9Bin,r9Min, r9Max, 3, -0.5, 2.5) ;
    h_r9VsNofTracks_[0][2] = dbe_->book2D(histname+"Endcap"," photons r9 vs nTracks from conversions: Endcap Ecal",r9Bin,r9Min, r9Max, 3, -0.5, 2.5) ;

    histname="EoverPtracks";
    h_EoverPTracks_[0][0] = dbe_->book1D(histname+"BarrelPix"," photons conversion E/p: barrel pix",eoverpBin, eoverpMin,eoverpMax);
    h_EoverPTracks_[0][1] = dbe_->book1D(histname+"BarrelTib"," photons conversion E/p: barrel tib",eoverpBin, eoverpMin,eoverpMax);
    h_EoverPTracks_[0][2] = dbe_->book1D(histname+"BarrelTob"," photons conversion E/p: barrel tob ",eoverpBin, eoverpMin,eoverpMax);

    h_EoverPTracks_[1][0] = dbe_->book1D(histname+"All"," photons conversion E/p: all Ecal ",100, 0., 5.);
    h_EoverPTracks_[1][1] = dbe_->book1D(histname+"Barrel"," photons conversion E/p: Barrel Ecal",100, 0., 5.);
    h_EoverPTracks_[1][2] = dbe_->book1D(histname+"Endcap"," photons conversion E/p: Endcap Ecal ",100, 0., 5.);

    histname="PoverEtracks";
    h_PoverETracks_[1][0] = dbe_->book1D(histname+"All"," photons conversion p/E: all Ecal ",povereBin, povereMin, povereMax);
    h_PoverETracks_[1][1] = dbe_->book1D(histname+"Barrel"," photons conversion p/E: Barrel Ecal",povereBin, povereMin, povereMax);
    h_PoverETracks_[1][2] = dbe_->book1D(histname+"Endcap"," photons conversion p/E: Endcap Ecal ",povereBin, povereMin, povereMax);



    histname="EoverEtrueVsEoverP";
    h2_EoverEtrueVsEoverP_[0] = dbe_->book2D(histname+"All"," photons conversion E/Etrue vs E/P: all Ecal ",100, 0., 5., 100, 0.5, 1.5);
    h2_EoverEtrueVsEoverP_[1] = dbe_->book2D(histname+"Barrel"," photons conversion  E/Etrue vs E/: Barrel Ecal",100, 0., 5.,100, 0.5, 1.5);
    h2_EoverEtrueVsEoverP_[2] = dbe_->book2D(histname+"Endcap"," photons conversion  E/Etrue vs E/: Endcap Ecal ",100, 0., 5., 100, 0.5, 1.5);
    histname="PoverPtrueVsEoverP";
    h2_PoverPtrueVsEoverP_[0] = dbe_->book2D(histname+"All"," photons conversion P/Ptrue vs E/P: all Ecal ",100, 0., 5., 100, 0., 2.5);
    h2_PoverPtrueVsEoverP_[1] = dbe_->book2D(histname+"Barrel"," photons conversion  P/Ptrue vs E/: Barrel Ecal",100, 0., 5.,100, 0., 2.5);
    h2_PoverPtrueVsEoverP_[2] = dbe_->book2D(histname+"Endcap"," photons conversion  P/Ptrue vs E/: Endcap Ecal ",100, 0., 5., 100, 0., 2.5);

    histname="EoverEtrueVsEta";
    h2_EoverEtrueVsEta_[0] = dbe_->book2D(histname+"All"," photons conversion with 2 (associated) reco tracks  E/Etrue vs #eta: all Ecal ",etaBin2,etaMin, etaMax,100, 0., 2.5);
    histname="pEoverEtrueVsEta";
    p_EoverEtrueVsEta_[0] = dbe_->book1D(histname+"All"," photons conversion with 2 (associated) reco tracks E/Etrue vs #eta: all Ecal ",etaBin2,etaMin, etaMax);

    histname="EoverEtrueVsEta";
    h2_EoverEtrueVsEta_[1] = dbe_->book2D(histname+"All2"," photons conversion  2 reco tracks  E/Etrue vs #eta: all Ecal ",etaBin2,etaMin, etaMax,100, 0., 2.5);
    histname="pEoverEtrueVsEta";
    p_EoverEtrueVsEta_[1] = dbe_->book1D(histname+"All2"," photons conversion  2 reco tracks  E/Etrue vs #eta: all Ecal ",etaBin2,etaMin, etaMax);


    histname="EoverEtrueVsR";
    h2_EoverEtrueVsR_[0] = dbe_->book2D(histname+"All"," photons conversion E/Etrue vs R: all Ecal ",rBin,rMin, rMax,100, 0., 2.5);
    histname="pEoverEtrueVsR";
    p_EoverEtrueVsR_[0] = dbe_->book1D(histname+"All"," photons conversion E/Etrue vs R: all Ecal ",rBin,rMin,rMax);



    histname="PoverPtrueVsEta";
    h2_PoverPtrueVsEta_[0] = dbe_->book2D(histname+"All"," photons conversion P/Ptrue vs #eta: all Ecal ",etaBin2,etaMin, etaMax,100, 0., 5.);
    histname="pPoverPtrueVsEta";
    p_PoverPtrueVsEta_[0] = dbe_->book1D(histname+"All"," photons conversion P/Ptrue vs #eta: all Ecal ",etaBin2,etaMin, etaMax);

    histname="EoverPVsEta";
    h2_EoverPVsEta_[0] = dbe_->book2D(histname+"All"," photons conversion E/P vs #eta: all Ecal ",etaBin2,etaMin, etaMax,100, 0., 5.);
    histname="pEoverPVsEta";
    p_EoverPVsEta_[0] = dbe_->book1D(histname+"All"," photons conversion E/P vs #eta: all Ecal ",etaBin2,etaMin, etaMax);

    histname="EoverPVsR";
    h2_EoverPVsR_[0] = dbe_->book2D(histname+"All"," photons conversion E/P vs R: all Ecal ",rBin,rMin, rMax,100, 0., 5.);
    histname="pEoverPVsR";
    p_EoverPVsR_[0] = dbe_->book1D(histname+"All"," photons conversion E/P vs R: all Ecal ",rBin,rMin,rMax);


    

    histname="hInvMass";
    h_invMass_[0][0]= dbe_->book1D(histname+"All_AllTracks"," Photons:Tracks from conversion: Pair invariant mass: all Ecal ",100, 0., 1.5);
    h_invMass_[0][1]= dbe_->book1D(histname+"Barrel_AllTracks"," Photons:Tracks from conversion: Pair invariant mass: Barrel Ecal ",100, 0., 1.5);
    h_invMass_[0][2]= dbe_->book1D(histname+"Endcap_AllTracks"," Photons:Tracks from conversion: Pair invariant mass: Endcap Ecal ",100, 0., 1.5);
    histname="hInvMass";
    h_invMass_[1][0]= dbe_->book1D(histname+"All_AssTracks"," Photons:Tracks from conversion: Pair invariant mass: all Ecal ",100, 0., 1.5);
    h_invMass_[1][1]= dbe_->book1D(histname+"Barrel_AssTracks"," Photons:Tracks from conversion: Pair invariant mass: Barrel Ecal ",100, 0., 1.5);
    h_invMass_[1][2]= dbe_->book1D(histname+"Endcap_AssTracks"," Photons:Tracks from conversion: Pair invariant mass: Endcap Ecal ",100, 0., 1.5);


    histname="hDPhiTracksAtVtx";
    h_DPhiTracksAtVtx_[1][0] =dbe_->book1D(histname+"All", " Photons:Tracks from conversions: #delta#phi Tracks at vertex: all Ecal",dPhiTracksBin,dPhiTracksMin,dPhiTracksMax); 
    h_DPhiTracksAtVtx_[1][1] =dbe_->book1D(histname+"Barrel", " Photons:Tracks from conversions: #delta#phi Tracks at vertex: Barrel Ecal",dPhiTracksBin,dPhiTracksMin,dPhiTracksMax); 
    h_DPhiTracksAtVtx_[1][2] =dbe_->book1D(histname+"Endcap", " Photons:Tracks from conversions: #delta#phi Tracks at vertex: Endcap Ecal",dPhiTracksBin,dPhiTracksMin,dPhiTracksMax); 

    histname="hDPhiTracksAtVtxVsEta";
    h2_DPhiTracksAtVtxVsEta_ = dbe_->book2D(histname+"All","  Photons:Tracks from conversions: #delta#phi Tracks at vertex vs #eta",etaBin2,etaMin, etaMax,100, -0.5, 0.5);
    histname="pDPhiTracksAtVtxVsEta";
    p_DPhiTracksAtVtxVsEta_ = dbe_->book1D(histname+"All"," Photons:Tracks from conversions: #delta#phi Tracks at vertex vs #eta ",etaBin2,etaMin, etaMax);

    histname="hDPhiTracksAtVtxVsR";
    h2_DPhiTracksAtVtxVsR_ = dbe_->book2D(histname+"All","  Photons:Tracks from conversions: #delta#phi Tracks at vertex vs R",rBin,rMin, rMax,100, -0.5, 0.5);
    histname="pDPhiTracksAtVtxVsR";
    p_DPhiTracksAtVtxVsR_ = dbe_->book1D(histname+"All"," Photons:Tracks from conversions: #delta#phi Tracks at vertex vs R ",rBin,rMin, rMax);


    histname="hDCotTracks";
    h_DCotTracks_[1][0]= dbe_->book1D(histname+"All"," Photons:Tracks from conversions #delta cotg(#Theta) Tracks: all Ecal ",dCotTracksBin,dCotTracksMin,dCotTracksMax); 
    h_DCotTracks_[1][1]= dbe_->book1D(histname+"Barrel"," Photons:Tracks from conversions #delta cotg(#Theta) Tracks: Barrel Ecal ",dCotTracksBin,dCotTracksMin,dCotTracksMax); 
    h_DCotTracks_[1][2]= dbe_->book1D(histname+"Endcap"," Photons:Tracks from conversions #delta cotg(#Theta) Tracks: Endcap Ecal ",dCotTracksBin,dCotTracksMin,dCotTracksMax); 


    histname="hDCotTracksVsEta";
    h2_DCotTracksVsEta_ = dbe_->book2D(histname+"All","  Photons:Tracks from conversions:  #delta cotg(#Theta) Tracks vs #eta",etaBin2,etaMin, etaMax,100, -0.2, 0.2);
    histname="pDCotTracksVsEta";
    p_DCotTracksVsEta_ = dbe_->book1D(histname+"All"," Photons:Tracks from conversions:  #delta cotg(#Theta) Tracks vs #eta ",etaBin2,etaMin, etaMax);

    histname="hDCotTracksVsR";
    h2_DCotTracksVsR_ = dbe_->book2D(histname+"All","  Photons:Tracks from conversions:  #delta cotg(#Theta)  Tracks at vertex vs R",rBin,rMin, rMax,100, -0.2, 0.2);
    histname="pDCotTracksVsR";
    p_DCotTracksVsR_ = dbe_->book1D(histname+"All"," Photons:Tracks from conversions:  #delta cotg(#Theta) Tracks at vertex vs R ",rBin,rMin, rMax);


    histname="hDistMinAppTracks";
    h_distMinAppTracks_[1][0]= dbe_->book1D(histname+"All"," Photons:Tracks from conversions Min Approach Dist Tracks: all Ecal ",dEtaTracksBin,-0.1,0.6); 
    h_distMinAppTracks_[1][1]= dbe_->book1D(histname+"Barrel"," Photons:Tracks from conversions Min Approach Dist Tracks: Barrel Ecal ",dEtaTracksBin,-0.1,0.6); 
    h_distMinAppTracks_[1][2]= dbe_->book1D(histname+"Endcap"," Photons:Tracks from conversions Min Approach Dist Tracks: Endcap Ecal ",dEtaTracksBin,-0.1,0.6); 



    histname="hDPhiTracksAtEcal";
    h_DPhiTracksAtEcal_[1][0]= dbe_->book1D(histname+"All"," Photons:Tracks from conversions:  #delta#phi at Ecal : all Ecal ",dPhiTracksBin,0.,dPhiTracksMax); 
    h_DPhiTracksAtEcal_[1][1]= dbe_->book1D(histname+"Barrel"," Photons:Tracks from conversions:  #delta#phi at Ecal : Barrel Ecal ",dPhiTracksBin,0.,dPhiTracksMax); 
    h_DPhiTracksAtEcal_[1][2]= dbe_->book1D(histname+"Endcap"," Photons:Tracks from conversions:  #delta#phi at Ecal : Endcap Ecal ",dPhiTracksBin,0.,dPhiTracksMax); 
    histname="h2_DPhiTracksAtEcalVsR";
    h2_DPhiTracksAtEcalVsR_= dbe_->book2D(histname+"All"," Photons:Tracks from conversions:  #delta#phi at Ecal vs R : all Ecal ",rBin,rMin, rMax, dPhiTracksBin,0.,dPhiTracksMax);
    histname="pDPhiTracksAtEcalVsR";
    p_DPhiTracksAtEcalVsR_ = dbe_->book1D(histname+"All"," Photons:Tracks from conversions:  #delta#phi at Ecal  vs R ",rBin,rMin, rMax);

    histname="h2_DPhiTracksAtEcalVsEta";
    h2_DPhiTracksAtEcalVsEta_= dbe_->book2D(histname+"All"," Photons:Tracks from conversions:  #delta#phi at Ecal vs #eta : all Ecal ",etaBin2,etaMin, etaMax, dPhiTracksBin,0.,dPhiTracksMax);
    histname="pDPhiTracksAtEcalVsEta";
    p_DPhiTracksAtEcalVsEta_ = dbe_->book1D(histname+"All"," Photons:Tracks from conversions:  #delta#phi at Ecal  vs #eta ",etaBin2,etaMin, etaMax);



    histname="hDEtaTracksAtEcal";
    h_DEtaTracksAtEcal_[1][0]= dbe_->book1D(histname+"All"," Photons:Tracks from conversions:  #delta#eta at Ecal : all Ecal ",dEtaTracksBin,dEtaTracksMin,dEtaTracksMax); 
    h_DEtaTracksAtEcal_[1][1]= dbe_->book1D(histname+"Barrel"," Photons:Tracks from conversions:  #delta#eta at Ecal : Barrel Ecal ",dEtaTracksBin,dEtaTracksMin,dEtaTracksMax); 
    h_DEtaTracksAtEcal_[1][2]= dbe_->book1D(histname+"Endcap"," Photons:Tracks from conversions:  #delta#eta at Ecal : Endcap Ecal ",dEtaTracksBin,dEtaTracksMin,dEtaTracksMax); 
   

    h_convVtxRvsZ_[0] =   dbe_->book2D("convVtxRvsZAll"," Photon Reco conversion vtx position",zBinForXray, zMinForXray, zMaxForXray, rBinForXray, rMinForXray, rMaxForXray); 
    h_convVtxRvsZ_[1] =   dbe_->book2D("convVtxRvsZBarrel"," Photon Reco conversion vtx position",zBinForXray, zMinForXray, zMaxForXray, rBinForXray, rMinForXray, rMaxForXray); 
    h_convVtxRvsZ_[2] =   dbe_->book2D("convVtxRvsZEndcap"," Photon Reco conversion vtx position",zBin2ForXray, zMinForXray, zMaxForXray, rBinForXray, rMinForXray, rMaxForXray); 

    h_convVtxdX_ =   dbe_->book1D("convVtxdX"," Photon Reco conversion vtx dX",100, -20.,20.);
    h_convVtxdY_ =   dbe_->book1D("convVtxdY"," Photon Reco conversion vtx dY",100, -20.,20.);
    h_convVtxdZ_ =   dbe_->book1D("convVtxdZ"," Photon Reco conversion vtx dZ",100, -20.,20.);
    h_convVtxdR_ =   dbe_->book1D("convVtxdR"," Photon Reco conversion vtx dR",100, -20.,20.);

    h2_convVtxdRVsR_ =  dbe_->book2D("h2ConvVtxdRVsR","Photon Reco conversion vtx dR vsR" ,rBin,rMin, rMax,100, -20.,20.);
    p_convVtxdRVsR_ =  dbe_->book1D("pConvVtxdRVsR","Photon Reco conversion vtx dR vsR" ,rBin,rMin, rMax);
    h2_convVtxdRVsEta_ =  dbe_->book2D("h2ConvVtxdRVsEta","Photon Reco conversion vtx dR vs Eta" ,etaBin2,etaMin, etaMax,100, -20.,20.);
    p_convVtxdRVsEta_ =  dbe_->book1D("pConvVtxdRVsEta","Photon Reco conversion vtx dR vs Eta" ,etaBin2,etaMin, etaMax);
    
    h2_convVtxRrecVsTrue_ =  dbe_->book2D("h2ConvVtxRrecVsTrue","Photon Reco conversion vtx R rec vs true" ,rBin,rMin, rMax,rBin,rMin, rMax);

    histname="vtxChi2";
    h_vtxChi2_[0] = dbe_->book1D(histname+"All","vertex #chi^{2} all", 100, chi2Min, chi2Max); 
    h_vtxChi2_[1] = dbe_->book1D(histname+"Barrel","vertex #chi^{2} barrel", 100, chi2Min, chi2Max); 
    h_vtxChi2_[2] = dbe_->book1D(histname+"Endcap","vertex #chi^{2} endcap", 100, chi2Min, chi2Max); 
    histname="vtxChi2Prob";
    h_vtxChi2Prob_[0] = dbe_->book1D(histname+"All","vertex #chi^{2} all", 100, 0., 1.);
    h_vtxChi2Prob_[1] = dbe_->book1D(histname+"Barrel","vertex #chi^{2} barrel", 100, 0., 1.);
    h_vtxChi2Prob_[2] = dbe_->book1D(histname+"Endcap","vertex #chi^{2} endcap", 100, 0., 1.);

    h_zPVFromTracks_[1] =  dbe_->book1D("zPVFromTracks"," Photons: PV z from conversion tracks",100, -25., 25.);
    h_dzPVFromTracks_[1] =  dbe_->book1D("dzPVFromTracks"," Photons: PV Z_rec - Z_true from conversion tracks",100, -5., 5.);
    h2_dzPVVsR_ =  dbe_->book2D("h2dzPVVsR","Photon Reco conversions: dz(PV) vs R" ,rBin,rMin, rMax,100, -3.,3.);
    p_dzPVVsR_ =  dbe_->book1D("pdzPVVsR","Photon Reco conversions: dz(PV) vs R" ,rBin,rMin, rMax);



    //////////////////// plots per track
    histname="nHitsVsEta";
    nHitsVsEta_[0] =  dbe_->book2D(histname+"AllTracks","Photons:Tracks from conversions: # of hits vs #eta all tracks",etaBin,etaMin, etaMax,25,0., 25.);
    histname="h_nHitsVsEta";
    h_nHitsVsEta_[0] =  dbe_->book1D(histname+"AllTracks","Photons:Tracks from conversions: # of hits vs #eta all tracks",etaBin,etaMin, etaMax);

    histname="nHitsVsEta";
    nHitsVsEta_[1] =  dbe_->book2D(histname+"AssTracks","Photons:Tracks from conversions: # of hits vs #eta associated tracks",etaBin,etaMin, etaMax,25,0., 25.);
    histname="h_nHitsVsEta";
    h_nHitsVsEta_[1] =  dbe_->book1D(histname+"AssTracks","Photons:Tracks from conversions: # of hits vs #eta associated tracks",etaBin,etaMin, etaMax);


    histname="nHitsVsR";
    nHitsVsR_[0] =  dbe_->book2D(histname+"AllTracks","Photons:Tracks from conversions: # of hits vs radius all tracks" ,rBin,rMin, rMax,25,0.,25);
    histname="h_nHitsVsR";
    h_nHitsVsR_[0] =  dbe_->book1D(histname+"AllTracks","Photons:Tracks from conversions: # of hits vs radius all tracks",rBin,rMin, rMax);
    histname="tkChi2";
    h_tkChi2_[0] = dbe_->book1D(histname+"AllTracks","Photons:Tracks from conversions: #chi^{2} of all tracks", 100, chi2Min, chi2Max);  
    histname="tkChi2Large";
    h_tkChi2Large_[0] = dbe_->book1D(histname+"AllTracks","Photons:Tracks from conversions: #chi^{2} of all tracks", 1000, 0., 5000.0); 

    histname="nHitsVsR";
    nHitsVsR_[1] =  dbe_->book2D(histname+"AssTracks","Photons:Tracks from conversions: # of hits vs radius associated tracks" ,rBin,rMin, rMax,25,0.,25);
    histname="h_nHitsVsR";
    h_nHitsVsR_[1] =  dbe_->book1D(histname+"AssTracks","Photons:Tracks from conversions: # of hits vs radius associated tracks",rBin,rMin, rMax);

    histname="tkChi2";
    h_tkChi2_[1] = dbe_->book1D(histname+"AssTracks","Photons:Tracks from conversions: #chi^{2} of associated  tracks", 100, chi2Min, chi2Max); 
    histname="tkChi2Large";
    h_tkChi2Large_[1] = dbe_->book1D(histname+"AssTracks","Photons:Tracks from conversions: #chi^{2} of associated  tracks", 1000, 0., 5000.0); 

    histname="h2Chi2VsEta";
    h2_Chi2VsEta_[0]=dbe_->book2D(histname+"All"," Reco Track  #chi^{2} vs #eta: All ",etaBin2,etaMin, etaMax,100, chi2Min, chi2Max); 
    histname="pChi2VsEta";
    p_Chi2VsEta_[0]=dbe_->book1D(histname+"All"," Reco Track #chi^{2} vs #eta : All ",etaBin2,etaMin, etaMax);


    histname="h2Chi2VsR";
    h2_Chi2VsR_[0]=dbe_->book2D(histname+"All"," Reco Track  #chi^{2} vs R: All ",rBin,rMin, rMax,100,chi2Min, chi2Max);
    histname="pChi2VsR";
    p_Chi2VsR_[0]=dbe_->book1D(histname+"All"," Reco Track #chi^{2} vas R : All ",rBin,rMin,rMax);



    histname="hTkD0";
    h_TkD0_[0]=dbe_->book1D(histname+"All"," Reco Track D0*q: All ",100,-0.1,0.6);
    h_TkD0_[1]=dbe_->book1D(histname+"Barrel"," Reco Track D0*q: Barrel ",100,-0.1,0.6);
    h_TkD0_[2]=dbe_->book1D(histname+"Endcap"," Reco Track D0*q: Endcap ",100,-0.1,0.6);



    histname="hTkPtPull";
    h_TkPtPull_[0]=dbe_->book1D(histname+"All"," Reco Track Pt pull: All ",100, -10., 10.);
    histname="hTkPtPull";
    h_TkPtPull_[1]=dbe_->book1D(histname+"Barrel"," Reco Track Pt pull: Barrel ",100, -10., 10.);
    histname="hTkPtPull";
    h_TkPtPull_[2]=dbe_->book1D(histname+"Endcap"," Reco Track Pt pull: Endcap ",100, -10., 10.);

    histname="h2TkPtPullEta";
    h2_TkPtPull_[0]=dbe_->book2D(histname+"All"," Reco Track Pt pull: All ",etaBin2,etaMin, etaMax,100, -10., 10.);
    histname="pTkPtPullEta";
    p_TkPtPull_[0]=dbe_->book1D(histname+"All"," Reco Track Pt pull: All ",etaBin2,etaMin, etaMax);


    histname="PtRecVsPtSim";
    h2_PtRecVsPtSim_[0]=dbe_->book2D(histname+"All", "Pt Rec vs Pt sim: All ", etBin,etMin,etMax,etBin,etMin, etMax);
    h2_PtRecVsPtSim_[1]=dbe_->book2D(histname+"Barrel", "Pt Rec vs Pt sim: Barrel ", etBin,etMin,etMax,etBin,etMin, etMax);
    h2_PtRecVsPtSim_[2]=dbe_->book2D(histname+"Endcap", "Pt Rec vs Pt sim: Endcap ", etBin,etMin,etMax,etBin,etMin, etMax);
    histname="PtRecVsPtSimMixProv";
    h2_PtRecVsPtSimMixProv_ =dbe_->book2D(histname+"All", "Pt Rec vs Pt sim All for mix with general tracks ", etBin,etMin,etMax,etBin,etMin, etMax);


    histname="eBcOverTkPout";
    hBCEnergyOverTrackPout_[0] = dbe_->book1D(histname+"All","Matrching BC E/P_out: all Ecal ",100, 0., 5.);
    hBCEnergyOverTrackPout_[1] = dbe_->book1D(histname+"Barrel","Matrching BC E/P_out: Barrel ",100, 0., 5.);
    hBCEnergyOverTrackPout_[2] = dbe_->book1D(histname+"Endcap","Matrching BC E/P_out: Endcap ",100, 0., 5.);
    

    ////////////// test on OutIn tracks
    h_OIinnermostHitR_ = dbe_->book1D("OIinnermostHitR"," R innermost hit for OI tracks ",50, 0., 25);
    h_IOinnermostHitR_ = dbe_->book1D("IOinnermostHitR"," R innermost hit for IO tracks ",50, 0., 25);

    /// test track provenance
    h_trkProv_[0] = dbe_->book1D("allTrkProv"," Track pair provenance ",4, 0., 4.);
    h_trkProv_[1] = dbe_->book1D("assTrkProv"," Track pair provenance ",4, 0., 4.);

  }



}


 void  PhotonValidator::beginRun (edm::Run const & r, edm::EventSetup const & theEventSetup) {
   
   //get magnetic field
  edm::LogInfo("ConvertedPhotonProducer") << " get magnetic field" << "\n";
  theEventSetup.get<IdealMagneticFieldRecord>().get(theMF_);  


  edm::ESHandle<TrackAssociatorBase> theHitsAssociator;
  theEventSetup.get<TrackAssociatorRecord>().get("TrackAssociatorByHits",theHitsAssociator);
  theTrackAssociator_ = (TrackAssociatorBase *) theHitsAssociator.product();


  

}





void PhotonValidator::analyze( const edm::Event& e, const edm::EventSetup& esup )
{
  
  
  using namespace edm;
  const float etaPhiDistance=0.01;
  // Fiducial region
  const float TRK_BARL =0.9;
  const float BARL = 1.4442; // DAQ TDR p.290
  const float END_LO = 1.566;
  const float END_HI = 2.5;
  // Electron mass
  const Float_t mElec= 0.000511;


  nEvt_++;  
  LogInfo("PhotonValidator") << "PhotonValidator Analyzing event number: " << e.id() << " Global Counter " << nEvt_ <<"\n";
  //  LogDebug("PhotonValidator") << "PhotonValidator Analyzing event number: "  << e.id() << " Global Counter " << nEvt_ <<"\n";
  std::cout << "PhotonValidator Analyzing event number: "  << e.id() << " Global Counter " << nEvt_ <<"\n";


  // get the  calo topology  from the event setup:
  edm::ESHandle<CaloTopology> pTopology;
  esup.get<CaloTopologyRecord>().get(theCaloTopo_);
  const CaloTopology *topology = theCaloTopo_.product();

  // get the geometry from the event setup:
  esup.get<CaloGeometryRecord>().get(theCaloGeom_);


 // Transform Track into TransientTrack (needed by the Vertex fitter)
  edm::ESHandle<TransientTrackBuilder> theTTB;
  esup.get<TransientTrackRecord>().get("TransientTrackBuilder",theTTB);


  ///// Get the recontructed  photons
  Handle<reco::PhotonCollection> photonHandle; 
  e.getByLabel(photonCollectionProducer_, photonCollection_ , photonHandle);
  const reco::PhotonCollection photonCollection = *(photonHandle.product());
  if (!photonHandle.isValid()) {
    edm::LogError("PhotonProducer") << "Error! Can't get the Photon collection "<< std::endl;
    return; 
  }


  //// Get the Out In CKF tracks from conversions 
  Handle< edm::View<reco::Track> > outInTrkHandle;
  e.getByLabel(conversionOITrackProducer_,  outInTrkHandle);
  //std::cout << "ConvPhoAnalyzerWithOfficialAssociation  outInTrack collection size " << (*outInTrkHandle).size() << "\n";
  
  //// Get the In Out  CKF tracks from conversions 
  Handle< edm::View<reco::Track> > inOutTrkHandle;
  e.getByLabel(conversionIOTrackProducer_, inOutTrkHandle);
  //std::cout  << " ConvPhoAnalyzerWithOfficialAssociation inOutTrack collection size " << (*inOutTrkHandle).size() << "\n";
  


 // Loop over Out In Tracks 
  int iTrk=0;
  int nHits=0;
  for( View<reco::Track>::const_iterator    iTk =  (*outInTrkHandle).begin(); iTk !=  (*outInTrkHandle).end(); iTk++) {
    std::cout  << " Barrel  Out In Track charge " << iTk->charge() << " Num of RecHits " << iTk->recHitsSize() << " inner momentum " << sqrt( iTk->innerMomentum().Mag2() ) << "\n";  
    std::cout  << " Barrel Out In Track Extra inner momentum  " << sqrt(iTk->extra()->innerMomentum().Mag2()) <<  " inner position R " <<  sqrt( iTk->innerPosition().Perp2() ) << "\n";  
    h_OIinnermostHitR_ ->Fill ( sqrt( iTk->innerPosition().Perp2() ) );
    for (  trackingRecHit_iterator itHits=iTk->extra()->recHitsBegin();  itHits!=iTk->extra()->recHitsEnd(); ++itHits ) {
      if ( (*itHits)->isValid() ) {
	nHits++;
	//	cout <<nHits <<") RecHit in GP " <<  trackerGeom->idToDet((*itHits)->geographicalId())->surface().toGlobal((*itHits)->localPosition()) << " R "<< trackerGeom->idToDet((*itHits)->geographicalId())->surface().toGlobal((*itHits)->localPosition()).perp() << " Z " << trackerGeom->idToDet((*itHits)->geographicalId())->surface().toGlobal((*itHits)->localPosition()).z() << "\n";
      }
      
     
    }
    
    iTrk++;
    
    
  }
  
// Loop over In Out Tracks Barrel
  iTrk=0;
  for( View<reco::Track>::const_iterator    iTk =  (*inOutTrkHandle).begin(); iTk !=  (*inOutTrkHandle).end(); iTk++) {
    std::cout  << " Barrel In Out Track charge " << iTk->charge() << " Num of RecHits " << iTk->recHitsSize() << " inner momentum " << sqrt( iTk->innerMomentum().Mag2())  << "\n";  
    std::cout   << " Barrel In Out  Track Extra inner momentum  " << sqrt(iTk->extra()->innerMomentum().Mag2()) << "\n"; 
    h_IOinnermostHitR_ ->Fill ( sqrt( iTk->innerPosition().Perp2() ) );  
    nHits=0;
    for (  trackingRecHit_iterator itHits=iTk->extra()->recHitsBegin();  itHits!=iTk->extra()->recHitsEnd(); ++itHits ) {
      if ( (*itHits)->isValid() ) {
	nHits++;
	//cout <<nHits <<") RecHit in GP " << trackerGeom->idToDet((*itHits)->geographicalId())->surface().toGlobal((*itHits)->localPosition())  << " R "<< trackerGeom->idToDet((*itHits)->geographicalId())->surface().toGlobal((*itHits)->localPosition()).perp() << " Z " << trackerGeom->idToDet((*itHits)->geographicalId())->surface().toGlobal((*itHits)->localPosition()).z() << "\n";
	
      }
    }
    
    
    
    iTrk++;
  }



  


  //////////////////// Get the MC truth
  //get simtrack info
  std::vector<SimTrack> theSimTracks;
  std::vector<SimVertex> theSimVertices;
  
  edm::Handle<SimTrackContainer> SimTk;
  edm::Handle<SimVertexContainer> SimVtx;
  e.getByLabel("g4SimHits",SimTk);
  e.getByLabel("g4SimHits",SimVtx);
  
  theSimTracks.insert(theSimTracks.end(),SimTk->begin(),SimTk->end());
  theSimVertices.insert(theSimVertices.end(),SimVtx->begin(),SimVtx->end());
  std::vector<PhotonMCTruth> mcPhotons=thePhotonMCTruthFinder_->find (theSimTracks,  theSimVertices);  

  // Get electron tracking truth
  edm::Handle<TrackingParticleCollection> ElectronTPHandle;
  e.getByLabel(label_tp_,ElectronTPHandle);
  //  e.getByLabel("mergedtruth","MergedTrackTruth",ElectronTPHandle);
  const TrackingParticleCollection trackingParticles = *(ElectronTPHandle.product());

  //// Track association with TrackingParticles
  std::vector<reco::PhotonCollection::const_iterator> StoRMatchedConvertedPhotons;
  // Sim to Reco
  reco::SimToRecoCollection OISimToReco = theTrackAssociator_->associateSimToReco(outInTrkHandle, ElectronTPHandle, &e);
  reco::SimToRecoCollection IOSimToReco = theTrackAssociator_->associateSimToReco(inOutTrkHandle, ElectronTPHandle, &e);
  // Reco to Sim
  reco::RecoToSimCollection OIRecoToSim = theTrackAssociator_->associateRecoToSim(outInTrkHandle, ElectronTPHandle, &e);
  reco::RecoToSimCollection IORecoToSim = theTrackAssociator_->associateRecoToSim(inOutTrkHandle, ElectronTPHandle, &e);
  //
  vector<reco::SimToRecoCollection*> StoRCollPtrs;
  StoRCollPtrs.push_back(&OISimToReco);
  StoRCollPtrs.push_back(&IOSimToReco);
  vector<reco::RecoToSimCollection*> RtoSCollPtrs;
  RtoSCollPtrs.push_back(&OIRecoToSim);
  RtoSCollPtrs.push_back(&IORecoToSim);
  //
  for (int i=0; i<2; i++)  
    nSimPho_[i]=0;
  for (int i=0; i<2; i++)  
    nSimConv_[i]=0;

  //////////////////////////////////////////////////////////////////////
  for( reco::PhotonCollection::const_iterator  iPho = photonCollection.begin(); iPho != photonCollection.end(); iPho++) {
    if ( iPho->pt() < minPhoEtCut_ ) continue;
    if ( ! (  fabs(iPho->eta() ) <= BARL || ( fabs(iPho->eta()) >= END_LO && fabs(iPho->eta() ) <=END_HI ) ) ) 
      continue;  // all ecal fiducial region

    bool  phoIsInBarrel=false;
    bool  phoIsInEndcap=false;
    if ( fabs(iPho->eta() ) < 1.479 ) {
      phoIsInBarrel=true;
    } else {
      phoIsInEndcap=true;
    }
    
    for (reco::PhotonCollection::const_iterator iPho2=iPho+1; iPho2!=photonCollection.end(); iPho2++){
      math::XYZTLorentzVector p12 = iPho->p4()+iPho2->p4();
      float gamgamMass2 = p12.Dot(p12);
      
      h_gamgamMass_[0][0] -> Fill(sqrt( gamgamMass2 ));
      if ( phoIsInBarrel ) h_gamgamMass_[0][1] -> Fill(sqrt( gamgamMass2 ));
      if ( phoIsInEndcap ) h_gamgamMass_[0][2] -> Fill(sqrt( gamgamMass2 ));
      
      if ( iPho->hasConversionTracks() ) {
	h_gamgamMass_[1][0] -> Fill(sqrt( gamgamMass2 ));
	if ( phoIsInBarrel ) h_gamgamMass_[1][1] -> Fill(sqrt( gamgamMass2 ));
	if ( phoIsInEndcap ) h_gamgamMass_[1][2] -> Fill(sqrt( gamgamMass2 ));
      } else {
	h_gamgamMass_[2][0] -> Fill(sqrt( gamgamMass2 ));
	if ( phoIsInBarrel ) h_gamgamMass_[2][1] -> Fill(sqrt( gamgamMass2 ));
	if ( phoIsInEndcap ) h_gamgamMass_[2][2] -> Fill(sqrt( gamgamMass2 ));
      }

    }
  }
  

  cout << " PhotonValidator mcPhotons.size() " << mcPhotons.size() << endl;
  for ( std::vector<PhotonMCTruth>::const_iterator mcPho=mcPhotons.begin(); mcPho !=mcPhotons.end(); mcPho++) {
    if ( (*mcPho).fourMomentum().et() < minPhoEtCut_ ) continue;
    if ( signal_ ) if ( (*mcPho).motherType() != -1 ) continue;

    float mcPhi= (*mcPho).fourMomentum().phi();
    mcPhi_= phiNormalization(mcPhi);
    mcEta_= (*mcPho).fourMomentum().pseudoRapidity();   
    mcEta_ = etaTransformation(mcEta_, (*mcPho).primaryVertex().z() ); 
    
    mcConvR_= (*mcPho).vertex().perp();   
    mcConvX_= (*mcPho).vertex().x();    
    mcConvY_= (*mcPho).vertex().y();    
    mcConvZ_= (*mcPho).vertex().z();    



    
    if ( ! (  fabs(mcEta_) <= BARL || ( fabs(mcEta_) >= END_LO && fabs(mcEta_) <=END_HI ) ) ) 
      continue;  // all ecal fiducial region


    nSimPho_[0]++;
    h_SimPhoE_[0]->Fill(  (*mcPho).fourMomentum().e());
    h_SimPhoEt_[0]->Fill(  (*mcPho).fourMomentum().et());
    h_SimPhoEta_[0]->Fill( mcEta_ ) ;
    h_SimPhoPhi_[0]->Fill( mcPhi_ );

    h_SimPhoMotherEt_[0]->Fill(  (*mcPho).motherMomentum().et()  );
    h_SimPhoMotherEta_[0]->Fill(  (*mcPho).motherMomentum().pseudoRapidity());

    for (unsigned int f=0; f<etaintervals_.size()-1; f++){
      if (mcEta_>etaintervals_[f]&&
	  mcEta_<etaintervals_[f+1]) {
	totSimPhoEta_[f]++;
      }
    }
    for (unsigned int f=0; f<phiintervals_.size()-1; f++){
      if (mcPhi_>phiintervals_[f]&&
	  mcPhi_<phiintervals_[f+1]) {
	totSimPhoPhi_[f]++;
      }
    }
    for (unsigned int f=0; f<etintervals_.size()-1; f++){
      if ((*mcPho).fourMomentum().et() >etintervals_[f]&&
	  (*mcPho).fourMomentum().et()<etintervals_[f+1]) {
	totSimPhoEt_[f]++;
      }
    }


    ////////////////////////////////// extract info about simulated conversions 
    
    bool goodSimConversion=false;
    bool visibleConversion=false;
    bool visibleConversionsWithTwoSimTracks=false;
    if (  (*mcPho).isAConversion() == 1 ) {
      nSimConv_[0]++;
      h_SimConvE_[0]->Fill(  (*mcPho).fourMomentum().e());
      h_SimConvEt_[0]->Fill(  (*mcPho).fourMomentum().et());
      h_SimConvEta_[0]->Fill( mcEta_ ) ;

      if ( mcConvR_ <15) h_SimConvEtaPix_[0]->Fill( mcEta_ ) ;

      h_SimConvPhi_[0]->Fill( mcPhi_ );
      h_SimConvR_[0]->Fill( mcConvR_ );
      h_SimConvZ_[0]->Fill( mcConvZ_ );

      if ( ( fabs(mcEta_) <= BARL && mcConvR_ <85 )  || 
           ( fabs(mcEta_) > BARL && fabs(mcEta_) <=END_HI && fabs( (*mcPho).vertex().z() ) < 210 )  ) visibleConversion=true;
      


      theConvTP_.clear(); 
      std::cout << " PhotonValidator TrackingParticles   TrackingParticleCollection size "<<  trackingParticles.size() <<  "\n";
      for(size_t i = 0; i < trackingParticles.size(); ++i){
	TrackingParticleRef tp (ElectronTPHandle,i);
	//std::cout << "  Electron pt " << tp -> pt() << " charge " << tp -> charge() << " pdgId " << tp->pdgId() << " Hits for this track: " << tp -> trackPSimHit().size() << std::endl;      
	//std::cout << " track vertex position x " <<  tp->vertex().x() << " y " << tp->vertex().y() << " z " << tp->vertex().z() << std::endl;
	//std::cout << " track vertex position x " <<  tp->vx() << " y " << tp->vy() << " z " << tp->vz() << std::endl;
	//std::cout << " conversion vertex position x " <<  (*mcPho).vertex().x() << " y " << (*mcPho).vertex().y() << " z " << (*mcPho).vertex().z() << std::endl;
	if ( fabs( tp->vx() - (*mcPho).vertex().x() ) < 0.001   &&
	     fabs( tp->vy() - (*mcPho).vertex().y() ) < 0.001   &&
	     fabs( tp->vz() - (*mcPho).vertex().z() ) < 0.001) {
	  
	  //std::cout << " From conversion Electron pt " << tp -> pt() << " charge " << tp -> charge() << " pdgId " << tp->pdgId() << " Hits for this track: " << tp -> trackPSimHit().size() << std::endl;      
	  //	std::cout << " track vertex position x " <<  tp->vertex().x() << " y " << tp->vertex().y() << " z " << tp->vertex().z() << std::endl;
	  //std::cout << " conversion vertex position x " <<  (*mcPho).vertex().x() << " y " << (*mcPho).vertex().y() << " z " << (*mcPho).vertex().z() << "  R " <<  (*mcPho).vertex().perp() << std::endl;
	  theConvTP_.push_back( tp );	
	}
      }
      //      std::cout << " PhotonValidator  theConvTP_ size " <<   theConvTP_.size() << std::endl;	

      if ( theConvTP_.size() == 2 )   visibleConversionsWithTwoSimTracks=true;
      goodSimConversion=false;

      if (   visibleConversion && visibleConversionsWithTwoSimTracks )  goodSimConversion=true;
      if ( goodSimConversion ) {
	nSimConv_[1]++;	
	h_SimConvE_[1]->Fill(  (*mcPho).fourMomentum().e());
	h_SimConvEt_[1]->Fill(  (*mcPho).fourMomentum().et());
	h_SimConvEta_[1]->Fill( mcEta_ ) ;
	h_SimConvPhi_[1]->Fill( mcPhi_ );
	h_SimConvR_[1]->Fill( mcConvR_ );
	h_SimConvZ_[1]->Fill( mcConvZ_ );
	
	for ( vector<TrackingParticleRef>::iterator iTrk=theConvTP_.begin(); iTrk!=theConvTP_.end(); ++iTrk) {
	  h_simTkPt_ -> Fill ( (*iTrk)->pt() );
	  h_simTkEta_ -> Fill ( (*iTrk)->eta() );

	}

	////////// Denominators for conversion efficiencies
	for (unsigned int f=0; f<etaintervalslarge_.size()-1; f++){
	  if (mcEta_>etaintervalslarge_[f]&&
	      mcEta_<etaintervalslarge_[f+1]) {
	    totSimConvEta_[f]++;
	  }
	}
	for (unsigned int f=0; f<phiintervals_.size()-1; f++){
	  if (mcPhi_>phiintervals_[f]&&
	      mcPhi_<phiintervals_[f+1]) {
	    totSimConvPhi_[f]++;
	  }
	}
	for (unsigned int f=0; f<rintervals_.size()-1; f++){
	  if (mcConvR_>rintervals_[f]&&
	      mcConvR_<rintervals_[f+1]) {
	    totSimConvR_[f]++;
	  }
	}
	for (unsigned int f=0; f<zintervals_.size()-1; f++){
	  if (mcConvZ_>zintervals_[f]&&
	      mcConvZ_<zintervals_[f+1]) {
	    totSimConvZ_[f]++;
	  }
	}
	for (unsigned int f=0; f<etintervals_.size()-1; f++){
	  if ((*mcPho).fourMomentum().et() >etintervals_[f]&&
	      (*mcPho).fourMomentum().et()<etintervals_[f+1]) {
	    totSimConvEt_[f]++;
	  }
	}


      
      }
      
    }  ////////////// End of info from conversions //////////////////////////////////////////////////


    
    float minDelta=10000.;
    std::vector<reco::Photon> thePhotons;
    int index=0;
    int iMatch=-1;
    bool matched=false;
    
    for( reco::PhotonCollection::const_iterator  iPho = photonCollection.begin(); iPho != photonCollection.end(); iPho++) {
      reco::Photon aPho = reco::Photon(*iPho);
      thePhotons.push_back(aPho);
      float phiPho=aPho.phi();
      float etaPho=aPho.eta();
      float deltaPhi = phiPho-mcPhi_;
      float deltaEta = etaPho-mcEta_;
      if ( deltaPhi > pi )  deltaPhi -= twopi;
      if ( deltaPhi < -pi) deltaPhi += twopi;
      deltaPhi=pow(deltaPhi,2);
      deltaEta=pow(deltaEta,2);
      float delta = sqrt( deltaPhi+deltaEta); 
      if ( delta<0.1 && delta < minDelta ) {
	minDelta=delta;
	iMatch=index;
         
      }
      index++;
    }  // end loop over reco photons
    if ( iMatch>-1 ) matched=true; 

  
    if ( matched ) {
      for (unsigned int f=0; f<etaintervals_.size()-1; f++){
	if (mcEta_>etaintervals_[f]&&
	    mcEta_<etaintervals_[f+1]) {
	  totMatchedSimPhoEta_[f]++;
	}
      }
      for (unsigned int f=0; f<phiintervals_.size()-1; f++){
	if (mcPhi_>phiintervals_[f]&&
	    mcPhi_<phiintervals_[f+1]) {
	  totMatchedSimPhoPhi_[f]++;
	}
      }
      for (unsigned int f=0; f<etintervals_.size()-1; f++){
	if ((*mcPho).fourMomentum().et()>etintervals_[f]&&
	    (*mcPho).fourMomentum().et()<etintervals_[f+1]) {
	  totMatchedSimPhoEt_[f]++;
	}
      }



      nSimPho_[1]++;
      h_SimPhoE_[1]->Fill(  (*mcPho).fourMomentum().e());
      h_SimPhoEt_[1]->Fill(  (*mcPho).fourMomentum().et());
      h_SimPhoEta_[1]->Fill( mcEta_ ) ;
      h_SimPhoPhi_[1]->Fill( mcPhi_ );
      
      h_SimPhoMotherEt_[1]->Fill(  (*mcPho).motherMomentum().et()  );
      h_SimPhoMotherEta_[1]->Fill(  (*mcPho).motherMomentum().pseudoRapidity());

    }




    if ( ! matched) continue;

    bool  phoIsInBarrel=false;
    bool  phoIsInEndcap=false;

    reco::Photon matchingPho = thePhotons[iMatch];

    if ( fabs(matchingPho.superCluster()->position().eta() ) < 1.479 ) {
      phoIsInBarrel=true;
    } else {
      phoIsInEndcap=true;
    }
    edm::Handle<EcalRecHitCollection>   ecalRecHitHandle;
    if ( phoIsInBarrel ) {
      
      // Get handle to rec hits ecal barrel 
      e.getByLabel(barrelEcalHits_, ecalRecHitHandle);
      if (!ecalRecHitHandle.isValid()) {
	edm::LogError("PhotonProducer") << "Error! Can't get the product "<<barrelEcalHits_.label();
	return;
      }
      
      
    } else if ( phoIsInEndcap ) {    
      
      // Get handle to rec hits ecal encap 
      e.getByLabel(endcapEcalHits_, ecalRecHitHandle);
      if (!ecalRecHitHandle.isValid()) {
	edm::LogError("PhotonProducer") << "Error! Can't get the product "<<endcapEcalHits_.label();
	return;
      }
      
    }

    int type=0;
    const EcalRecHitCollection ecalRecHitCollection = *(ecalRecHitHandle.product());
    float photonE = matchingPho.energy();
    float photonEt= matchingPho.energy()/cosh( matchingPho.eta()) ;
    float r9 = matchingPho.r9();
    float r1 = matchingPho.r1x5();
    float r2 = matchingPho.r2x5();
    float sigmaIetaIeta =  matchingPho.sigmaIetaIeta();
    float hOverE = matchingPho.hadronicOverEm();
    float ecalIso = matchingPho.ecalRecHitSumEtConeDR04();
    float hcalIso = matchingPho.hcalTowerSumEtConeDR04();
    float trkIso =  matchingPho.trkSumPtSolidConeDR04();
    float nIsoTrk   =  matchingPho.nTrkSolidConeDR04();
  
    h_scEta_[type]->Fill( matchingPho.superCluster()->eta() );
    h_scPhi_[type]->Fill( matchingPho.superCluster()->phi() );
    h_scEtaWidth_[type]->Fill( matchingPho.superCluster()->etaWidth() );
    h_scPhiWidth_[type]->Fill( matchingPho.superCluster()->phiWidth() );
    h_scE_[type][0]->Fill( matchingPho.superCluster()->energy() );
    h_scEt_[type][0]->Fill( matchingPho.superCluster()->energy()/cosh( matchingPho.superCluster()->eta()) );
    if ( phoIsInEndcap ) h_psE_->Fill( matchingPho.superCluster()->preshowerEnergy() ) ;
    //
    h_r9_[type][0]->Fill( r9 );
    h2_r9VsEta_[0] -> Fill (mcEta_, r9);      
    h2_r9VsEt_[0] -> Fill ((*mcPho).fourMomentum().et(), r9);      
    //
    h_r1_[type][0]->Fill( r1 );
    h2_r1VsEta_[0] -> Fill (mcEta_, r1);      
    h2_r1VsEt_[0] -> Fill ((*mcPho).fourMomentum().et(), r1);      
    //
    h_r2_[type][0]->Fill( r2 );
    h2_r2VsEta_[0] -> Fill (mcEta_, r2);      
    h2_r2VsEt_[0] -> Fill ((*mcPho).fourMomentum().et(), r2);      
    //
    h_sigmaIetaIeta_[type][0]->Fill( sigmaIetaIeta );
    h2_sigmaIetaIetaVsEta_[0] -> Fill (mcEta_, sigmaIetaIeta );      
    h2_sigmaIetaIetaVsEt_[0] -> Fill ((*mcPho).fourMomentum().et(), sigmaIetaIeta);      
    //    
    h_hOverE_[type][0]->Fill( hOverE );
    h2_hOverEVsEta_[0] -> Fill (mcEta_, hOverE );      
    h2_hOverEVsEt_[0] -> Fill ((*mcPho).fourMomentum().et(), hOverE);      
    //
    h_ecalRecHitSumEtConeDR04_[type][0]->Fill( ecalIso );
    h2_ecalRecHitSumEtConeDR04VsEta_[0] -> Fill (mcEta_, ecalIso );      
    h2_ecalRecHitSumEtConeDR04VsEt_[0] -> Fill ((*mcPho).fourMomentum().et(), ecalIso);      
    //
    h_hcalTowerSumEtConeDR04_[type][0]->Fill( hcalIso );
    h2_hcalTowerSumEtConeDR04VsEta_[0] -> Fill (mcEta_, hcalIso );      
    h2_hcalTowerSumEtConeDR04VsEt_[0] -> Fill ((*mcPho).fourMomentum().et(), hcalIso);      
    //
    h_isoTrkSolidConeDR04_[type][0]->Fill( trkIso );
    h2_isoTrkSolidConeDR04VsEta_[0] -> Fill (mcEta_, trkIso );      
    h2_isoTrkSolidConeDR04VsEt_[0] -> Fill ((*mcPho).fourMomentum().et(), trkIso);      
    //
    h_nTrkSolidConeDR04_[type][0]->Fill( nIsoTrk );
    h2_nTrkSolidConeDR04VsEta_[0] -> Fill (mcEta_, nIsoTrk );      
    h2_nTrkSolidConeDR04VsEt_[0] -> Fill ((*mcPho).fourMomentum().et(), nIsoTrk);      
    //
    h_phoEta_[type]->Fill( matchingPho.eta() );
    h_phoPhi_[type]->Fill( matchingPho.phi() );
    h_phoDEta_[0]->Fill (  matchingPho.eta() - (*mcPho).fourMomentum().eta() );
    h_phoDPhi_[0]->Fill (  matchingPho.phi() - mcPhi_ );
    h_phoE_[type][0]->Fill( photonE );
    h_phoEt_[type][0]->Fill( photonEt);
    //
    h_phoERes_[0][0]->Fill( photonE / (*mcPho).fourMomentum().e() );
    h2_eResVsEta_[0]->Fill (mcEta_, photonE/(*mcPho).fourMomentum().e()  ) ;
    //
    if (  (*mcPho).isAConversion() == 0 ) {
      h2_eResVsEta_[1]->Fill (mcEta_, photonE/ (*mcPho).fourMomentum().e()  ) ;
      h2_r9VsEta_[1] -> Fill (mcEta_, r9);
      h2_r9VsEt_[1] -> Fill ((*mcPho).fourMomentum().et(), r9);     
      //
      h2_r1VsEta_[1] -> Fill (mcEta_, r1);      
      h2_r1VsEt_[1] -> Fill ((*mcPho).fourMomentum().et(), r1);      
      //
      h2_r2VsEta_[1] -> Fill (mcEta_, r2);      
      h2_r2VsEt_[1] -> Fill ((*mcPho).fourMomentum().et(), r2);      
      //
      h2_sigmaIetaIetaVsEta_[1] -> Fill (mcEta_, sigmaIetaIeta );      
      h2_sigmaIetaIetaVsEt_[1] -> Fill ((*mcPho).fourMomentum().et(), sigmaIetaIeta);
      //
      h2_hOverEVsEta_[1] -> Fill (mcEta_, hOverE );      
      h2_hOverEVsEt_[1] -> Fill ((*mcPho).fourMomentum().et(), hOverE);
      //
      h2_ecalRecHitSumEtConeDR04VsEta_[1] -> Fill (mcEta_, ecalIso );      
      //
      h2_hcalTowerSumEtConeDR04VsEta_[1] -> Fill (mcEta_, hcalIso );      
      //
      h2_isoTrkSolidConeDR04VsEta_[1] -> Fill (mcEta_, trkIso );      
      h2_isoTrkSolidConeDR04VsEt_[1] -> Fill ((*mcPho).fourMomentum().et(), trkIso);      
      //
      h2_nTrkSolidConeDR04VsEta_[1] -> Fill (mcEta_, nIsoTrk );      
      h2_nTrkSolidConeDR04VsEt_[1] -> Fill ((*mcPho).fourMomentum().et(), nIsoTrk);      

    }





 
    if ( photonE/(*mcPho).fourMomentum().e()  < 0.3 &&   photonE/(*mcPho).fourMomentum().e() > 0.1 ) {
      //      std::cout << " Eta sim " << mcEta_ << " sc eta " << matchingPho.superCluster()->eta() << " pho eta " << matchingPho.eta() << std::endl;
 
    }


    if ( r9 > 0.93 )  h_phoERes_[1][0]->Fill( photonE / (*mcPho).fourMomentum().e() );
    if ( r9 <= 0.93 )  h_phoERes_[2][0]->Fill(photonE / (*mcPho).fourMomentum().e() );
	 
          
    if ( phoIsInBarrel ) {
      h_scE_[type][1]->Fill( matchingPho.superCluster()->energy() );
      h_scEt_[type][1]->Fill( matchingPho.superCluster()->energy()/cosh( matchingPho.superCluster()->eta()) );
      h_r9_[type][1]->Fill( r9 );
      h_r1_[type][1]->Fill( r1 );
      h_r2_[type][1]->Fill( r2 );
      h_sigmaIetaIeta_[type][1]->Fill( sigmaIetaIeta );
      h_hOverE_[type][1]->Fill( hOverE );
      h_ecalRecHitSumEtConeDR04_[type][1]->Fill( ecalIso );
      h2_ecalRecHitSumEtConeDR04VsEt_[1] -> Fill ((*mcPho).fourMomentum().et(), ecalIso);      
      h_hcalTowerSumEtConeDR04_[type][1]->Fill( hcalIso );
      h2_hcalTowerSumEtConeDR04VsEt_[1] -> Fill ((*mcPho).fourMomentum().et(), hcalIso);      
      h_isoTrkSolidConeDR04_[type][1]->Fill( trkIso );
      h_nTrkSolidConeDR04_[type][1]->Fill( nIsoTrk );

      h_phoE_[type][1]->Fill( photonE );
      h_phoEt_[type][1]->Fill( photonEt );
      h_nConv_[type][1]->Fill(float( matchingPho.conversions().size()));
      
      h_phoERes_[0][1]->Fill( photonE / (*mcPho).fourMomentum().e() );
      if ( r9 > 0.93 ) {  
	h_phoERes_[1][1]->Fill(  photonE  / (*mcPho).fourMomentum().e() );
      }
      if ( r9 <= 0.93 )  { 
	h_phoERes_[2][1]->Fill( photonE / (*mcPho).fourMomentum().e() );
      }
    }
    if ( phoIsInEndcap ) {
      h_scE_[type][2]->Fill( matchingPho.superCluster()->energy() );
      h_scEt_[type][2]->Fill( matchingPho.superCluster()->energy()/cosh( matchingPho.superCluster()->eta()) );
      h_r9_[type][2]->Fill( r9 );
      h_r1_[type][2]->Fill( r1 );
      h_r2_[type][2]->Fill( r2 );
      h_sigmaIetaIeta_[type][2]->Fill( sigmaIetaIeta );
      h_hOverE_[type][2]->Fill( hOverE );
      h_ecalRecHitSumEtConeDR04_[type][2]->Fill( ecalIso );
      h2_ecalRecHitSumEtConeDR04VsEt_[2] -> Fill ((*mcPho).fourMomentum().et(), ecalIso);      
      h_hcalTowerSumEtConeDR04_[type][2]->Fill( hcalIso );
      h2_hcalTowerSumEtConeDR04VsEt_[2] -> Fill ((*mcPho).fourMomentum().et(), hcalIso);      
      h_isoTrkSolidConeDR04_[type][2]->Fill( trkIso );
      h_nTrkSolidConeDR04_[type][2]->Fill( nIsoTrk );
      h_phoE_[type][2]->Fill( photonE );
      h_phoEt_[type][2]->Fill( photonEt );
      h_nConv_[type][2]->Fill(float( matchingPho.conversions().size()));
      h_phoERes_[0][2]->Fill( photonE / (*mcPho).fourMomentum().e() );
      if ( r9 > 0.93 ) {  

	h_phoERes_[1][2]->Fill( photonE / (*mcPho).fourMomentum().e() );
      }
      if ( r9 <= 0.93 ) {
	h_phoERes_[2][2]->Fill( photonE / (*mcPho).fourMomentum().e() );
      }
    }
      



    
    if ( ! (visibleConversion &&  visibleConversionsWithTwoSimTracks ) ) continue;
    h_r9_[1][0]->Fill( r9 );
    if ( phoIsInBarrel ) h_r9_[1][1]->Fill( r9 );
    if ( phoIsInEndcap ) h_r9_[1][2]->Fill( r9 );
    
    
    h_nConv_[type][0]->Fill(float( matchingPho.conversions().size()));          
    
    
    ////////////////// plot quantitied related to conversions
    reco::ConversionRefVector conversions = matchingPho.conversions();
    for (unsigned int iConv=0; iConv<conversions.size(); iConv++) {
      
      
      h2_EoverEtrueVsEta_[1]->Fill (mcEta_,matchingPho.superCluster()->energy()/ (*mcPho).fourMomentum().e()  ) ;
      
      reco::ConversionRef aConv=conversions[iConv];
      std::vector<reco::TrackRef> tracks = aConv->tracks();
      if (tracks.size() < 2 ) continue;
      if ( sqrt( aConv->tracksPin()[0].Perp2()) < convTrackMinPtCut_ || sqrt( aConv->tracksPin()[1].Perp2()) < convTrackMinPtCut_) continue;
      
      
      if ( dCotCutOn_ ) {
	if (  (fabs(mcEta_) > 1.1 && fabs (mcEta_)  < 1.4  )  &&
	      fabs( aConv->pairCotThetaSeparation() ) > dCotHardCutValue_ ) continue;
	if ( fabs( aConv->pairCotThetaSeparation() ) > dCotCutValue_ ) continue;
      }
      
      //std::cout << " PhotonValidator converison algo name " << aConv->algoName() << " " << aConv->algo() << std::endl;
      
      nRecConv_++;
      
      h_convEta_[0]->Fill( aConv->caloCluster()[0]->eta() );
      h_convPhi_[0]->Fill( aConv->caloCluster()[0]->phi() );
      h_convERes_[0][0]->Fill( aConv->caloCluster()[0]->energy() / (*mcPho).fourMomentum().e() );
      h_r9VsNofTracks_[0][0]->Fill( r9, aConv->nTracks() ) ; 
      
      if ( phoIsInBarrel )  {
	h_convERes_[0][1]->Fill(aConv->caloCluster()[0]->energy() / (*mcPho).fourMomentum().e() );
	h_r9VsNofTracks_[0][1]->Fill( r9, aConv->nTracks() ) ; 
      }
      if ( phoIsInEndcap ) {
	h_convERes_[0][2]->Fill(aConv->caloCluster()[0]->energy() / (*mcPho).fourMomentum().e() );
	h_r9VsNofTracks_[0][2]->Fill( r9, aConv->nTracks() ) ; 
      }
      
      
      std::map<reco::TrackRef,TrackingParticleRef> myAss;
      std::map<reco::TrackRef,TrackingParticleRef>::const_iterator itAss;
      std::map<reco::TrackRef,TrackingParticleRef>::const_iterator itAssMin;
      std::map<reco::TrackRef,TrackingParticleRef>::const_iterator itAssMax;
      //     
      
      int nAssT2=0;
      int nAssT1=0;
      float px=0;
      float py=0;
      float pz=0;
      float e=0;
      std::cout << " Before loop on tracks  tracks size " << tracks.size() << " or " << aConv->tracks().size() <<  " nAssT2 " << nAssT2 << std::endl;
      for (unsigned int i=0; i<tracks.size(); i++) {

        type =0;
	nHitsVsEta_[type] ->Fill (mcEta_,   float(tracks[i]->numberOfValidHits()) );
	nHitsVsR_[type] ->Fill (mcConvR_,   float(tracks[i]->numberOfValidHits()) );
	h_tkChi2_[type] ->Fill (tracks[i]->normalizedChi2() ); 

        px+= tracks[i]->innerMomentum().x();
        py+= tracks[i]->innerMomentum().y();
        pz+= tracks[i]->innerMomentum().z();
        e +=  sqrt (  tracks[i]->innerMomentum().x()*tracks[i]->innerMomentum().x() +
		      tracks[i]->innerMomentum().y()*tracks[i]->innerMomentum().y() +
		      tracks[i]->innerMomentum().z()*tracks[i]->innerMomentum().z() +
		      +  mElec*mElec ) ;

	
	/////////// fill my local track - trackingparticle association map
	TrackingParticleRef myTP;
	for (size_t j = 0; j < RtoSCollPtrs.size(); j++) {          
	  reco::RecoToSimCollection q = *(RtoSCollPtrs[j]);
	  
	  RefToBase<reco::Track> myTk( aConv->tracks()[i] );
	  
	  if( q.find(myTk ) != q.end() ) {
	    std::vector<std::pair<TrackingParticleRef, double> > tp = q[myTk];
	    for (int itp=0; itp<tp.size(); itp++) {
	      myTP=tp[itp].first;
	      //      std::cout << " associated with TP " << myTP->pdgId() << " pt " << sqrt(myTP->momentum().perp2()) << std::endl;
	      myAss.insert( std::make_pair ( aConv->tracks()[i]  , myTP) );
	      nAssT2++;
	    }
	  }
	}
      }


      float totP = sqrt(px*px +py*py + pz*pz);
      float invM=  (e + totP) * (e-totP) ;
      if ( invM> 0.) {
	invM= sqrt( invM);
      } else {
	invM=-1;
      }

      type=0;
      h_invMass_[type][0] ->Fill( invM);
      if ( phoIsInBarrel ) h_invMass_[type][1] ->Fill(invM);
      if ( phoIsInEndcap ) h_invMass_[type][2] ->Fill(invM);


      ////////// Numerators for conversion absolute efficiency 
      if ( tracks.size() ==1  ) {
	for (unsigned int f=0; f<etaintervalslarge_.size()-1; f++){
	  if (mcEta_>etaintervalslarge_[f]&&
	      mcEta_<etaintervalslarge_[f+1]) {
	    totMatchedSimConvEtaOneTrack_[f]++;
	    
	  }
	}
	
	for (unsigned int f=0; f<rintervals_.size()-1; f++){
	  if (mcConvR_>rintervals_[f]&&
	      mcConvR_<rintervals_[f+1]) {
	    totMatchedSimConvROneTrack_[f]++;
	  }
	}
	
	for (unsigned int f=0; f<etintervals_.size()-1; f++){
	  if ((*mcPho).fourMomentum().et()>etintervals_[f]&&
	      (*mcPho).fourMomentum().et()<etintervals_[f+1]) {
	    totMatchedSimConvEtOneTrack_[f]++;
	  }
	}
      }




      if ( tracks.size() ==2 ) {
	h_convEta_[1]->Fill( aConv->caloCluster()[0]->eta() );	
	
	for (unsigned int f=0; f<etaintervalslarge_.size()-1; f++){
	  if (mcEta_>etaintervalslarge_[f]&&
	      mcEta_<etaintervalslarge_[f+1]) {
	    totMatchedSimConvEtaAllTwoTracks_[f]++;
	  }
	}
	for (unsigned int f=0; f<rintervals_.size()-1; f++){
	  if (mcConvR_>rintervals_[f]&&
	      mcConvR_<rintervals_[f+1]) {
	    totMatchedSimConvRAllTwoTracks_[f]++;
	  }
	}
	
	float trkProvenance=3;
	if ( tracks[0]->algoName() == "outInEcalSeededConv"  &&  tracks[1]->algoName() == "outInEcalSeededConv" ) trkProvenance=0;
	if ( tracks[0]->algoName() == "inOutEcalSeededConv"  &&  tracks[1]->algoName() == "inOutEcalSeededConv" ) trkProvenance=1;
	if ( ( tracks[0]->algoName() == "outInEcalSeededConv"  &&  tracks[1]->algoName() == "inOutEcalSeededConv") || 
	     ( tracks[1]->algoName() == "outInEcalSeededConv"  &&  tracks[0]->algoName() == "inOutEcalSeededConv") ) trkProvenance=2;
	if ( trkProvenance==3 ) std::cout << " PhotonValidator provenance of tracks is " << tracks[0]->algoName() << " and " << tracks[1]->algoName() << std::endl;
	h_trkProv_[0]->Fill( trkProvenance );
	
	
	
	////////// Numerators for conversion efficiencies: both tracks are associated
	if ( nAssT2 ==2 ) {
	  
	  


	  h_r9_[2][0]->Fill( r9 );
	  if ( phoIsInBarrel ) h_r9_[2][1]->Fill( r9 );
	  if ( phoIsInEndcap ) h_r9_[2][2]->Fill( r9 );

	  h_convEta_[2]->Fill( aConv->caloCluster()[0]->eta() );	

	  nRecConvAss_++;
	
	  for (unsigned int f=0; f<etaintervalslarge_.size()-1; f++){
	    if (mcEta_>etaintervalslarge_[f]&&
		mcEta_<etaintervalslarge_[f+1]) {
	      totMatchedSimConvEtaTwoTracks_[f]++;
	      if ( aConv->conversionVertex().isValid() ) {
		if ( trkProvenance==3 ) std::cout << " PhotonValidator provenance of tracks is mixed and vertex is valid " << std::endl;
                totMatchedConvEtaTwoTracksWithVtx_[f]++;
		float chi2Prob = ChiSquaredProbability( aConv->conversionVertex().normalizedChi2(),  aConv->conversionVertex().ndof() );
                if (   chi2Prob > 0)                
		  totMatchedConvEtaTwoTracksWithVtxProbGT0_[f]++;
                if (   chi2Prob > 0.005)                
		  totMatchedConvEtaTwoTracksWithVtxProbGT005_[f]++;
                if (   chi2Prob > 0.01)                
		  totMatchedConvEtaTwoTracksWithVtxProbGT01_[f]++;
		
	      }   
	    }
	  }
	  for (unsigned int f=0; f<phiintervals_.size()-1; f++){
	    if (mcPhi_>phiintervals_[f]&&
		mcPhi_<phiintervals_[f+1]) {
	      totMatchedSimConvPhiTwoTracks_[f]++;
	    }
	  }
	  for (unsigned int f=0; f<rintervals_.size()-1; f++){
	    if (mcConvR_>rintervals_[f]&&
		mcConvR_<rintervals_[f+1]) {
	      totMatchedSimConvRTwoTracks_[f]++;
	    }
	  }
	  for (unsigned int f=0; f<zintervals_.size()-1; f++){
	    if (mcConvZ_>zintervals_[f]&&
		mcConvZ_<zintervals_[f+1]) {
	      totMatchedSimConvZTwoTracks_[f]++;
	    }
	  }
	  for (unsigned int f=0; f<etintervals_.size()-1; f++){
	    if ((*mcPho).fourMomentum().et()>etintervals_[f]&&
		(*mcPho).fourMomentum().et()<etintervals_[f+1]) {
	      totMatchedSimConvEtTwoTracks_[f]++;
	    }
	  }


	  ///////////  Quantities per conversion
	  type =1;

	  h_trkProv_[1]->Fill( trkProvenance );


	  float eoverp= aConv->EoverP();

	  h_invMass_[type][0] ->Fill( invM);
	  h_convPRes_[type][0]->Fill( totP / (*mcPho).fourMomentum().e() );
	  h_EoverPTracks_[type][0] ->Fill( eoverp ) ;
	  h_PoverETracks_[type][0] ->Fill( 1./eoverp ) ;

	  h2_EoverEtrueVsEoverP_[0] ->Fill( eoverp,matchingPho.superCluster()->energy()/ (*mcPho).fourMomentum().e()  ) ;
	  h2_PoverPtrueVsEoverP_[0] ->Fill( eoverp, totP/ (*mcPho).fourMomentum().e()  ) ;
	  h2_EoverEtrueVsEta_[0]->Fill (mcEta_,matchingPho.superCluster()->energy()/ (*mcPho).fourMomentum().e()  ) ;
	  h2_EoverEtrueVsR_[0]->Fill (mcConvR_,matchingPho.superCluster()->energy()/ (*mcPho).fourMomentum().e()  ) ;

	  h2_PoverPtrueVsEta_[0]->Fill (mcEta_,totP/ (*mcPho).fourMomentum().e()  ) ;

	  h2_EoverPVsEta_[0]->Fill (mcEta_, eoverp);
	  h2_EoverPVsR_[0]->Fill (mcConvR_, eoverp);

        

	  reco::TrackRef track1 = tracks[0];
	  reco::TrackRef track2 = tracks[1];
	  reco::TransientTrack tt1 = (*theTTB).build( &track1);
	  reco::TransientTrack tt2 = (*theTTB).build( &track2);
	  TwoTrackMinimumDistance md;
	  md.calculate  (  tt1.initialFreeState(),  tt2.initialFreeState() );
	  if (md.status() )  {
	    //cout << " Min Dist " << md.distance() << std::endl;
	    h_distMinAppTracks_[1][0]->Fill ( md.distance() );
	  }  else {
	    nInvalidPCA_++;

	  }

	  float  dPhiTracksAtVtx = -99;
	  float phiTk1=  tracks[0]->innerMomentum().phi();
	  float phiTk2=  tracks[1]->innerMomentum().phi();
	  dPhiTracksAtVtx = phiTk1-phiTk2;
	  dPhiTracksAtVtx = phiNormalization( dPhiTracksAtVtx );

	  h_DPhiTracksAtVtx_[type][0]->Fill( dPhiTracksAtVtx);
	  h2_DPhiTracksAtVtxVsEta_->Fill( mcEta_, dPhiTracksAtVtx);
	  h2_DPhiTracksAtVtxVsR_->Fill( mcConvR_, dPhiTracksAtVtx);


	  h_DCotTracks_[type][0] ->Fill ( aConv->pairCotThetaSeparation() );
	  h2_DCotTracksVsEta_->Fill( mcEta_, aConv->pairCotThetaSeparation() );
	  h2_DCotTracksVsR_->Fill( mcConvR_, aConv->pairCotThetaSeparation() );





	  if ( phoIsInBarrel ) {
	    h_invMass_[type][1] ->Fill(invM);
	    h_convPRes_[type][1]->Fill( totP / (*mcPho).fourMomentum().e() );
	    h_EoverPTracks_[type][1] ->Fill( eoverp ) ;
	    if (  mcConvR_ < 15 )                 h_EoverPTracks_[0][0] ->Fill( eoverp ) ;
	    if (  mcConvR_ > 15 && mcConvR_< 58 ) h_EoverPTracks_[0][1] ->Fill( eoverp ) ;
	    if (  mcConvR_ > 58 )                 h_EoverPTracks_[0][2] ->Fill( eoverp ) ;
	    h_PoverETracks_[type][1] ->Fill( 1./eoverp ) ;
	    h_DPhiTracksAtVtx_[type][1]->Fill( dPhiTracksAtVtx);
	    h_DCotTracks_[type][1] ->Fill ( aConv->pairCotThetaSeparation() );

	    h2_EoverEtrueVsEoverP_[1] ->Fill( eoverp,matchingPho.superCluster()->energy()/ (*mcPho).fourMomentum().e()  ) ;
	    h2_PoverPtrueVsEoverP_[1] ->Fill( eoverp, totP/ (*mcPho).fourMomentum().e()  ) ;
	  }


	  if ( phoIsInEndcap ) {
	    h_invMass_[type][2] ->Fill(invM);
	    h_convPRes_[type][2]->Fill( totP / (*mcPho).fourMomentum().e() );
	    h_EoverPTracks_[type][2] ->Fill( eoverp ) ;
	    h_PoverETracks_[type][2] ->Fill( 1./eoverp ) ;
	    h_DPhiTracksAtVtx_[type][2]->Fill( dPhiTracksAtVtx);
	    h_DCotTracks_[type][2] ->Fill ( aConv->pairCotThetaSeparation() );
	    h2_EoverEtrueVsEoverP_[2] ->Fill( eoverp,matchingPho.superCluster()->energy()/ (*mcPho).fourMomentum().e()  ) ;
	    h2_PoverPtrueVsEoverP_[2] ->Fill( eoverp, totP/ (*mcPho).fourMomentum().e()  ) ;
	  }



	  if ( aConv->conversionVertex().isValid() ) {
	    float chi2Prob = ChiSquaredProbability( aConv->conversionVertex().normalizedChi2(),  aConv->conversionVertex().ndof() );
	    
	    h_convVtxRvsZ_[0] ->Fill ( fabs (aConv->conversionVertex().position().z() ),  sqrt(aConv->conversionVertex().position().perp2())  ) ;
	    h_convVtxdX_ ->Fill ( aConv->conversionVertex().position().x() - mcConvX_);
	    h_convVtxdY_ ->Fill ( aConv->conversionVertex().position().y() - mcConvY_);
	    h_convVtxdZ_ ->Fill ( aConv->conversionVertex().position().z() - mcConvZ_);
	    h_convVtxdR_ ->Fill ( sqrt(aConv->conversionVertex().position().perp2()) - mcConvR_);
	    h2_convVtxdRVsR_ ->Fill (mcConvR_, sqrt(aConv->conversionVertex().position().perp2()) - mcConvR_ );
	    h2_convVtxdRVsEta_ ->Fill (mcEta_, sqrt(aConv->conversionVertex().position().perp2()) - mcConvR_ );
	    h2_convVtxRrecVsTrue_ -> Fill (mcConvR_, sqrt(aConv->conversionVertex().position().perp2()) );

	    if ( fabs(matchingPho.superCluster()->position().eta() ) <= 1.)	h_convVtxRvsZ_[1] ->Fill ( fabs (aConv->conversionVertex().position().z() ),  sqrt(aConv->conversionVertex().position().perp2())  ) ;
	    if ( fabs(matchingPho.superCluster()->position().eta() ) > 1.)      h_convVtxRvsZ_[2] ->Fill ( fabs (aConv->conversionVertex().position().z() ),  sqrt(aConv->conversionVertex().position().perp2())  ) ;

	    h_vtxChi2Prob_[0]->Fill( chi2Prob ); 
	    h_vtxChi2_[0]->Fill(  aConv->conversionVertex().normalizedChi2() );
	    if ( phoIsInBarrel ) {
	      h_vtxChi2Prob_[1]->Fill( chi2Prob );
	      h_vtxChi2_[1]->Fill( aConv->conversionVertex().normalizedChi2() ); 
	    }
	    if ( phoIsInEndcap ) {   
	      h_vtxChi2Prob_[2]->Fill(  chi2Prob );
	      h_vtxChi2_[2]->Fill( aConv->conversionVertex().normalizedChi2() );
	    }

	  } // end conversion vertex valid



	  h_zPVFromTracks_[type]->Fill ( aConv->zOfPrimaryVertexFromTracks() );
	  h_dzPVFromTracks_[type]->Fill ( aConv->zOfPrimaryVertexFromTracks() - (*mcPho).primaryVertex().z() );
	  h2_dzPVVsR_ ->Fill(mcConvR_, aConv->zOfPrimaryVertexFromTracks() - (*mcPho).primaryVertex().z() );


	  float  dPhiTracksAtEcal=-99;
	  float  dEtaTracksAtEcal=-99;
	  if (aConv->bcMatchingWithTracks()[0].isNonnull() && aConv->bcMatchingWithTracks()[1].isNonnull() ) {
	    nRecConvAssWithEcal_++;
	    float recoPhi1 = aConv->ecalImpactPosition()[0].phi();
	    float recoPhi2 = aConv->ecalImpactPosition()[1].phi();
	    float recoEta1 = aConv->ecalImpactPosition()[0].eta();
	    float recoEta2 = aConv->ecalImpactPosition()[1].eta();
	    float bcPhi1 = aConv->bcMatchingWithTracks()[0]->phi();
	    float bcPhi2 = aConv->bcMatchingWithTracks()[1]->phi();
	    float bcEta1 = aConv->bcMatchingWithTracks()[0]->eta();
	    float bcEta2 = aConv->bcMatchingWithTracks()[1]->eta();
	    recoPhi1 = phiNormalization(recoPhi1);
	    recoPhi2 = phiNormalization(recoPhi2);
	    bcPhi1 = phiNormalization(bcPhi1);
	    bcPhi2 = phiNormalization(bcPhi2);
	    dPhiTracksAtEcal = recoPhi1 -recoPhi2;
	    dPhiTracksAtEcal = phiNormalization( dPhiTracksAtEcal );
	    dEtaTracksAtEcal = recoEta1 -recoEta2;
	  

	    h_DPhiTracksAtEcal_[type][0]->Fill( fabs(dPhiTracksAtEcal));
	    h2_DPhiTracksAtEcalVsR_ ->Fill (mcConvR_, fabs(dPhiTracksAtEcal));
	    h2_DPhiTracksAtEcalVsEta_ ->Fill (mcEta_, fabs(dPhiTracksAtEcal));

	    h_DEtaTracksAtEcal_[type][0]->Fill( dEtaTracksAtEcal);

	    if ( phoIsInBarrel ) {
	      h_DPhiTracksAtEcal_[type][1]->Fill( fabs(dPhiTracksAtEcal));
	      h_DEtaTracksAtEcal_[type][1]->Fill( dEtaTracksAtEcal);
	    }
	    if ( phoIsInEndcap ) {
	      h_DPhiTracksAtEcal_[type][2]->Fill( fabs(dPhiTracksAtEcal));
	      h_DEtaTracksAtEcal_[type][2]->Fill( dEtaTracksAtEcal);
	    }
	  
	  }
	
	
	
	  
	  ///////////  Quantities per track
	  for (unsigned int i=0; i<tracks.size(); i++) {
	    itAss= myAss.find(  aConv->tracks()[i] );
	    if ( itAss == myAss.end()  ) continue;

	    float trkProvenance=3;
	    if ( tracks[0]->algoName() == "outInEcalSeededConv"  &&  tracks[1]->algoName() == "outInEcalSeededConv" ) trkProvenance=0;
	    if ( tracks[0]->algoName() == "inOutEcalSeededConv"  &&  tracks[1]->algoName() == "inOutEcalSeededConv" ) trkProvenance=1;
	    if ( ( tracks[0]->algoName() == "outInEcalSeededConv"  &&  tracks[1]->algoName() == "inOutEcalSeededConv") || 
		 ( tracks[1]->algoName() == "outInEcalSeededConv"  &&  tracks[0]->algoName() == "inOutEcalSeededConv") ) trkProvenance=2;


	    nHitsVsEta_[type] ->Fill (mcEta_,   float(tracks[i]->numberOfValidHits()) );
	    nHitsVsR_[type] ->Fill (mcConvR_,   float(tracks[i]->numberOfValidHits()) );
	    h_tkChi2_[type] ->Fill (tracks[i]->normalizedChi2() ); 
	    h_tkChi2Large_[type] ->Fill (tracks[i]->normalizedChi2() ); 
	    h2_Chi2VsEta_[0] ->Fill(  mcEta_, tracks[i]->normalizedChi2() ); 
	    h2_Chi2VsR_[0] ->Fill(  mcConvR_, tracks[i]->normalizedChi2() ); 

	  

	    float simPt = sqrt( ((*itAss).second)->momentum().perp2() );
	    float recPt =   sqrt( aConv->tracks()[i]->innerMomentum().Perp2() ) ;
	    float ptres= recPt - simPt ;
	    float pterror = aConv->tracks()[i]->ptError();
	    h2_PtRecVsPtSim_[0]->Fill ( simPt, recPt);
	    if ( trkProvenance ==3 ) h2_PtRecVsPtSimMixProv_->Fill ( simPt, recPt);


	    h_TkPtPull_[0] ->Fill(ptres/pterror);
	    h2_TkPtPull_[0] ->Fill(mcEta_, ptres/pterror);

	    h_TkD0_[0]->Fill ( tracks[i]->d0()* tracks[i]->charge() );


	    if ( aConv->bcMatchingWithTracks()[i].isNonnull() ) hBCEnergyOverTrackPout_[0]->Fill  ( aConv->bcMatchingWithTracks()[i]->energy()/sqrt(aConv->tracks()[i]->outerMomentum().Mag2())  );

	    if ( phoIsInBarrel ) {
	      h_TkD0_[1]->Fill ( tracks[i]->d0()* tracks[i]->charge() );
	      h_TkPtPull_[1] ->Fill(ptres/pterror);
	      h2_PtRecVsPtSim_[1]->Fill ( simPt, recPt);
	      if ( aConv->bcMatchingWithTracks()[i].isNonnull() ) hBCEnergyOverTrackPout_[1]->Fill  ( aConv->bcMatchingWithTracks()[i]->energy()/sqrt(aConv->tracks()[i]->outerMomentum().Mag2())  );

	    }
	    if ( phoIsInEndcap ) { 
	      h_TkD0_[2]->Fill ( tracks[i]->d0()* tracks[i]->charge() );
	      h_TkPtPull_[2] ->Fill(ptres/pterror);
	      h2_PtRecVsPtSim_[2]->Fill ( simPt, recPt);
	      if ( aConv->bcMatchingWithTracks()[i].isNonnull() ) hBCEnergyOverTrackPout_[2]->Fill  ( aConv->bcMatchingWithTracks()[i]->energy()/sqrt(aConv->tracks()[i]->outerMomentum().Mag2())  );
	    }
	  

	  
	  } // end loop over track
	} // end analysis of two associated tracks
      } // end analysis of two  tracks
      
    } // loop over conversions
  } // End loop over simulated Photons
 
  h_nSimPho_[0]->Fill(float(nSimPho_[0]));
  h_nSimPho_[1]->Fill(float(nSimPho_[1]));
  h_nSimConv_[0]->Fill(float(nSimConv_[0]));
  h_nSimConv_[1]->Fill(float(nSimConv_[1]));
  



  ///////////////////  Measure fake rate
  for( reco::PhotonCollection::const_iterator  iPho = photonCollection.begin(); iPho != photonCollection.end(); iPho++) {
    reco::Photon aPho = reco::Photon(*iPho);
    float et= aPho.superCluster()->energy()/cosh( aPho.superCluster()->eta()) ;    
    reco::ConversionRefVector conversions = aPho.conversions();
    for (unsigned int iConv=0; iConv<conversions.size(); iConv++) {
    

      reco::ConversionRef aConv=conversions[iConv];
      std::vector<reco::TrackRef> tracks = aConv->tracks();

      if (tracks.size() < 2 ) continue;

      if ( dCotCutOn_ ) {
	if ( ( fabs(mcEta_) > 1.1 && fabs (mcEta_)  < 1.4  )  &&
	     fabs( aConv->pairCotThetaSeparation() ) > dCotHardCutValue_ )  continue;
	if ( fabs( aConv->pairCotThetaSeparation() ) > dCotCutValue_ ) continue;
      }


      
      for (unsigned int f=0; f<etaintervalslarge_.size()-1; f++){
	if (aPho.eta()>etaintervalslarge_[f]&&
	    aPho.eta()<etaintervalslarge_[f+1]) {
	  totMatchedRecConvEtaTwoTracks_[f]++;
 	}

      }
      for (unsigned int f=0; f<phiintervals_.size()-1; f++){
	if (aPho.phi()>phiintervals_[f]&&
	    aPho.phi()<phiintervals_[f+1]) {
	  totMatchedRecConvPhiTwoTracks_[f]++;
	}
      }

      for (unsigned int f=0; f<rintervals_.size()-1; f++){
	if (mcConvR_>rintervals_[f]&&
	    mcConvR_<rintervals_[f+1]) {
	  totMatchedRecConvRTwoTracks_[f]++;
	}
      }
      for (unsigned int f=0; f<zintervals_.size()-1; f++){
	if (mcConvZ_>zintervals_[f]&&
	    mcConvZ_<zintervals_[f+1]) {
	  totMatchedRecConvZTwoTracks_[f]++;
	}
      }

      for (unsigned int f=0; f<etintervals_.size()-1; f++){
	if ( et >etintervals_[f]&&
	     et <etintervals_[f+1]) {
	  totMatchedRecConvEtTwoTracks_[f]++;
	}
      }

      
      
      int  nAssT2=0;

      std::map<reco::TrackRef,TrackingParticleRef> myAss;
      for (unsigned int i=0; i<tracks.size(); i++) {
	
	TrackingParticleRef myTP;
	for (size_t j = 0; j < RtoSCollPtrs.size(); j++) {          
	  reco::RecoToSimCollection q = *(RtoSCollPtrs[j]);
	  
	  RefToBase<reco::Track> myTk( aConv->tracks()[i] );
	  
	  if( q.find(myTk ) != q.end() ) {
	    std::vector<std::pair<TrackingParticleRef, double> > tp = q[myTk];
	    for (int itp=0; itp<tp.size(); itp++) {
	      myTP=tp[itp].first;
	      //	      std::cout << " associated with TP " << myTP->pdgId() << " pt " << sqrt(myTP->momentum().perp2()) << std::endl;
	      myAss.insert( std::make_pair ( aConv->tracks()[i]  , myTP) );
	      nAssT2++;
	    }
	  }
	}
	
	if ( nAssT2 == 2) {
	  
	  for (unsigned int f=0; f<etaintervalslarge_.size()-1; f++){
	    if (aPho.eta()>etaintervalslarge_[f]&&
		aPho.eta()<etaintervalslarge_[f+1]) {
	      totRecAssConvEtaTwoTracks_[f]++;
	    }

	  }
	  for (unsigned int f=0; f<phiintervals_.size()-1; f++){
	    if (aPho.phi()>phiintervals_[f]&&
		aPho.phi()<phiintervals_[f+1]) {
	      totRecAssConvPhiTwoTracks_[f]++;
	    }
	  }
	  
	  for (unsigned int f=0; f<rintervals_.size()-1; f++){
	    if (mcConvR_>rintervals_[f]&&
		mcConvR_<rintervals_[f+1]) {
	      totRecAssConvRTwoTracks_[f]++;
	    }
	  }
	  for (unsigned int f=0; f<zintervals_.size()-1; f++){
	    if (mcConvZ_>zintervals_[f]&&
		mcConvZ_<zintervals_[f+1]) {
	      totRecAssConvZTwoTracks_[f]++;
	    }
	  }
	  for (unsigned int f=0; f<etintervals_.size()-1; f++){
	    if ( et >etintervals_[f]&&
		 et <etintervals_[f+1]) {
	      totRecAssConvEtTwoTracks_[f]++;
	    }
	  }


	  
	  
	}
      }
    }
  }
  




}



 void  PhotonValidator::fillPlotFromVectors(MonitorElement* h, std::vector<int>& numerator, std::vector<int>& denominator,std::string type){
    double value,err;
    for (unsigned int j=0; j<numerator.size(); j++){
      if (denominator[j]!=0){
	if (type=="effic")
	  value = ((double) numerator[j])/((double) denominator[j]);
	else if (type=="fakerate")
	  value = 1-((double) numerator[j])/((double) denominator[j]);
	else return;
	err = sqrt( value*(1-value)/(double) denominator[j] );
	h->setBinContent(j+1, value);
	h->setBinError(j+1,err);
      }
      else {
	h->setBinContent(j+1, 0);
      }
    }
  }


void PhotonValidator::endJob()
{


  for (int i=0; i<2; i++) {
    doProfileX( nHitsVsEta_[i], h_nHitsVsEta_[i]);
    doProfileX( nHitsVsR_[i], h_nHitsVsR_[i]);
  }

 
  doProfileX( h2_TkPtPull_[0],p_TkPtPull_[0]);
 


  doProfileX( h2_eResVsEta_[0], p_eResVsEta_[0] );
  doProfileX( h2_eResVsEta_[1], p_eResVsEta_[1] );

  doProfileX( h2_EoverEtrueVsEta_[0], p_EoverEtrueVsEta_[0] );
  doProfileX( h2_EoverEtrueVsEta_[1], p_EoverEtrueVsEta_[1] );

  doProfileX( h2_EoverEtrueVsR_[0], p_EoverEtrueVsR_[0] );
  doProfileX( h2_PoverPtrueVsEta_[0], p_PoverPtrueVsEta_[0] );
  doProfileX( h2_EoverPVsEta_[0], p_EoverPVsEta_[0] );
  doProfileX( h2_EoverPVsR_[0], p_EoverPVsR_[0] );


  doProfileX( h2_Chi2VsEta_[0], p_Chi2VsEta_[0] ); 
  doProfileX( h2_Chi2VsR_[0], p_Chi2VsR_[0] ); 
  doProfileX( h2_DPhiTracksAtVtxVsEta_,p_DPhiTracksAtVtxVsEta_);
  doProfileX( h2_DPhiTracksAtVtxVsR_,p_DPhiTracksAtVtxVsR_);
  doProfileX( h2_DCotTracksVsEta_,p_DCotTracksVsEta_);
  doProfileX( h2_DCotTracksVsR_,p_DCotTracksVsR_);

  doProfileX( h2_DPhiTracksAtEcalVsR_, p_DPhiTracksAtEcalVsR_);
  doProfileX( h2_DPhiTracksAtEcalVsEta_, p_DPhiTracksAtEcalVsEta_);

  doProfileX( h2_convVtxdRVsR_,  p_convVtxdRVsR_);
  doProfileX( h2_convVtxdRVsEta_,  p_convVtxdRVsEta_);
  doProfileX( h2_dzPVVsR_, p_dzPVVsR_); 


 
  fillPlotFromVectors(phoEffEta_,totMatchedSimPhoEta_,totSimPhoEta_,"effic");
  fillPlotFromVectors(phoEffPhi_,totMatchedSimPhoPhi_,totSimPhoPhi_,"effic");
  fillPlotFromVectors(phoEffEt_,totMatchedSimPhoEt_,totSimPhoEt_,"effic"); 
  //
  fillPlotFromVectors(convEffEtaOneTrack_,totMatchedSimConvEtaOneTrack_,totSimConvEta_,"effic");
  fillPlotFromVectors(convEffROneTrack_,totMatchedSimConvROneTrack_,totSimConvR_,"effic");
  fillPlotFromVectors(convEffEtOneTrack_,totMatchedSimConvEtOneTrack_,totSimConvEt_,"effic");
  //
  fillPlotFromVectors(convEffEtaTwoTracks_,totMatchedSimConvEtaTwoTracks_,totSimConvEta_,"effic");
  fillPlotFromVectors(convEffEtaTwoTracksAndVtx_,totMatchedConvEtaTwoTracksWithVtx_,totSimConvEta_,"effic");
  fillPlotFromVectors(convEffEtaTwoTracksAndVtx2_,totMatchedConvEtaTwoTracksWithVtx_,totMatchedSimConvEtaTwoTracks_,"effic"); 
  fillPlotFromVectors(convEffEtaTwoTracksAndVtxProbGT0_,totMatchedConvEtaTwoTracksWithVtxProbGT0_,totMatchedSimConvEtaTwoTracks_,"effic"); 
  fillPlotFromVectors(convEffEtaTwoTracksAndVtxProbGT005_,totMatchedConvEtaTwoTracksWithVtxProbGT005_,totMatchedSimConvEtaTwoTracks_,"effic"); 
  fillPlotFromVectors(convEffEtaTwoTracksAndVtxProbGT01_,totMatchedConvEtaTwoTracksWithVtxProbGT01_,totMatchedSimConvEtaTwoTracks_,"effic"); 
  fillPlotFromVectors(convEffEtaAllTwoTracks_,totMatchedSimConvEtaAllTwoTracks_,totSimConvEta_,"effic");

  fillPlotFromVectors(convEffPhiTwoTracks_,totMatchedSimConvPhiTwoTracks_,totSimConvPhi_,"effic");
  fillPlotFromVectors(convEffRTwoTracks_,totMatchedSimConvRTwoTracks_,totSimConvR_,"effic");
  fillPlotFromVectors(convEffRAllTwoTracks_,totMatchedSimConvRAllTwoTracks_,totSimConvR_,"effic");
  fillPlotFromVectors(convEffZTwoTracks_,totMatchedSimConvZTwoTracks_,totSimConvZ_,"effic");
  fillPlotFromVectors(convEffEtTwoTracks_,totMatchedSimConvEtTwoTracks_,totSimConvEt_,"effic");  

  fillPlotFromVectors(convFakeRateEtaTwoTracks_,totRecAssConvEtaTwoTracks_,totMatchedRecConvEtaTwoTracks_,"fakerate");
  fillPlotFromVectors(convFakeRatePhiTwoTracks_,totRecAssConvPhiTwoTracks_,totMatchedRecConvPhiTwoTracks_,"fakerate");
  fillPlotFromVectors(convFakeRateRTwoTracks_, totRecAssConvRTwoTracks_,totMatchedRecConvRTwoTracks_,"fakerate");
  fillPlotFromVectors(convFakeRateZTwoTracks_,totRecAssConvZTwoTracks_,totMatchedRecConvZTwoTracks_,"fakerate");
  fillPlotFromVectors(convFakeRateEtTwoTracks_,totRecAssConvEtTwoTracks_,totMatchedRecConvEtTwoTracks_,"fakerate");



  dbe_->showDirStructure();
  bool outputMEsInRootFile = parameters_.getParameter<bool>("OutputMEsInRootFile");
  std::string outputFileName = parameters_.getParameter<std::string>("OutputFileName");
  if(outputMEsInRootFile){
    dbe_->save(outputFileName);
  }
  
  edm::LogInfo("PhotonValidator") << "Analyzed " << nEvt_  << "\n";
  // std::cout  << "::endJob Analyzed " << nEvt_ << " events " << " with total " << nPho_ << " Photons " << "\n";
  std::cout  << "PhotonValidator::endJob Analyzed " << nEvt_ << " events " << "\n";
  std::cout << " Total number of photons " << nEntry_ << std::endl;
  std::cout << " Fraction of conv with two tracks having both BC matching " << float(nRecConvAssWithEcal_)/nRecConvAss_ << std::endl;
   
  return ;
}
 
float PhotonValidator::phiNormalization(float & phi)
{
  //---Definitions
  const float PI    = 3.1415927;
  const float TWOPI = 2.0*PI;


  if(phi >  PI) {phi = phi - TWOPI;}
  if(phi < -PI) {phi = phi + TWOPI;}

  //  cout << " Float_t PHInormalization out " << PHI << endl;
  return phi;

}


float PhotonValidator::etaTransformation(  float EtaParticle , float Zvertex)  {

  //---Definitions
  const float PI    = 3.1415927;
  const float TWOPI = 2.0*PI;

  //---Definitions for ECAL
  const float R_ECAL           = 136.5;
  const float Z_Endcap         = 328.0;
  const float etaBarrelEndcap  = 1.479; 
   
  //---ETA correction

  float Theta = 0.0  ; 
  float ZEcal = R_ECAL*sinh(EtaParticle)+Zvertex;

  if(ZEcal != 0.0) Theta = atan(R_ECAL/ZEcal);
  if(Theta<0.0) Theta = Theta+PI ;
  float ETA = - log(tan(0.5*Theta));
         
  if( fabs(ETA) > etaBarrelEndcap )
    {
      float Zend = Z_Endcap ;
      if(EtaParticle<0.0 )  Zend = -Zend ;
      float Zlen = Zend - Zvertex ;
      float RR = Zlen/sinh(EtaParticle); 
      Theta = atan(RR/Zend);
      if(Theta<0.0) Theta = Theta+PI ;
      ETA = - log(tan(0.5*Theta));		      
    } 
  //---Return the result
  return ETA;
  //---end
}


  std::vector<float> PhotonValidator::errors( TH1* histo1, TH1* histo2 ) {
  std::vector<float>  nEntryNum, nEntryDen;
  std::vector<float> ratio; 
  std::vector<float> erro;

  ///// calculate statistical errors
 
 
  const int nBin1=histo1->GetNbinsX();
  const int nBin2=histo2->GetNbinsX();
   
  //  std::cout << " histo 1 has " << nBin1 << " bins " << std::endl;
  // std::cout << " histo 2 has " << nBin2 << " bins " << std::endl;
   
  static const int sega1=nBin1;  
  static const int sega2=nBin2;  

  
  nEntryNum.clear();
  nEntryDen.clear();
  ratio.clear();
  erro.clear();
 
  //  cout << " sizes after clear " << nEntryNum.size() << " " << erro.size() << endl;

  for ( int i=1; i<=nBin1; i++) {
    //    cout << " Bin # " << i << " has " << histo1->GetBinContent(i) << " entries " << endl; 
    // nEntryNum[i]=histo1->GetBinContent(i);
    nEntryNum.push_back( histo1->GetBinContent(i) ) ;
  }
  for ( int i=1; i<=nBin1; i++) {
    // cout << " Bin # " << i << " has " << histo2->GetBinContent(i) << " entries " << endl; 
    //  nEntryDen[i]=histo2->GetBinContent(i);
   nEntryDen.push_back( histo2->GetBinContent(i) ) ; 

  }
  
  //  cout << " nEntryNum size " << nEntryNum.size() << " nEntryDen size " << nEntryDen.size() << " erro size " << erro.size() << endl;
 
  float val, delta;
  for ( int i=0; i<nEntryNum.size() ; i++) { 
    //   cout << " nEntryNum " << nEntryNum[i] << " nEntryDen " << nEntryDen[i] << endl;
    val = 0, delta=0;
    if ( nEntryDen[i] !=0 ) {
      val= nEntryNum[i]/nEntryDen[i];
      delta= sqrt ( val*(1.-val)/ nEntryDen[i]   );
      if (delta==0) delta=0.0003; 
     
    } 

    

    ratio.push_back( val  ) ;
    erro.push_back(delta);
    //    cout << " ratio " << ratio[i] << " erro " << erro[i] << endl; 
  }


  return erro;


 }


 void PhotonValidator::doProfileX(TH2 * th2, MonitorElement* me){
    if (th2->GetNbinsX()==me->getNbinsX()){
      TH1F * h1 = (TH1F*) th2->ProfileX();
      for (int bin=0;bin!=h1->GetNbinsX();bin++){
	me->setBinContent(bin+1,h1->GetBinContent(bin+1));
	me->setBinError(bin+1,h1->GetBinError(bin+1));
      }
      delete h1;
    } else {
      throw cms::Exception("MultiTrackValidator") << "Different number of bins!";
    }
  }

  void PhotonValidator::doProfileX(MonitorElement * th2m, MonitorElement* me) {
    doProfileX(th2m->getTH2F(), me);
  }
