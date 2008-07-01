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
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
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
//
#include "CLHEP/Units/PhysicalConstants.h"
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
 **  $Date: 2008/06/27 12:22:05 $ 
 **  $Revision: 1.2 $
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

   
    barrelEcalHits_   = pset.getParameter<edm::InputTag>("barrelEcalHits");
    endcapEcalHits_   = pset.getParameter<edm::InputTag>("endcapEcalHits");

    conversionOITrackProducer_ = pset.getParameter<std::string>("conversionOITrackProducer");
    conversionIOTrackProducer_ = pset.getParameter<std::string>("conversionIOTrackProducer");



    minPhoEtCut_ = pset.getParameter<double>("minPhoEtCut");   


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
      totMatchedSimConvEtaTwoTracks_.push_back(0);     
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
    }   
    step=(rMax-rMin)/rBin;
    rintervals_.push_back(rMin);
    for (int k=1;k<rBin+1;k++) {
      double d=rMin+k*step;
      rintervals_.push_back(d);
      totSimConvR_.push_back(0);     
      totMatchedSimConvRTwoTracks_.push_back(0);     
    }   
    step=(zMax-zMin)/zBin;
    zintervals_.push_back(zMin);
    for (int k=1;k<zBin+1;k++) {
      double d=zMin+k*step;
      zintervals_.push_back(d);
      totSimConvZ_.push_back(0);     
      totMatchedSimConvZTwoTracks_.push_back(0);     
    }   



}



void PhotonValidator::beginJob( const edm::EventSetup& setup)
{




  nEvt_=0;
  nEntry_=0;



   

  
  dbe_ = 0;
  dbe_ = edm::Service<DQMStore>().operator->();
  

  edm::ESHandle<TrackAssociatorBase> theHitsAssociator;
  setup.get<TrackAssociatorRecord>().get("TrackAssociatorByHits",theHitsAssociator);
  theTrackAssociator_ = (TrackAssociatorBase *) theHitsAssociator.product();



  if (dbe_) {
    if (verbosity_ > 0 ) {
      dbe_->setVerbose(1);
    } else {
      dbe_->setVerbose(0);
    }
  }
  if (dbe_) {
    if (verbosity_ > 0 ) dbe_->showDirStructure();
  }



  double resMin = parameters_.getParameter<double>("resMin");
  double resMax = parameters_.getParameter<double>("resMax");
  int resBin = parameters_.getParameter<int>("resBin");

  double eMin = parameters_.getParameter<double>("eMin");
  double eMax = parameters_.getParameter<double>("eMax");
  int eBin = parameters_.getParameter<int>("eBin");

  double etMin = parameters_.getParameter<double>("etMin");
  double etMax = parameters_.getParameter<double>("etMax");
  int etBin = parameters_.getParameter<int>("etBin");

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

  //////// set up vectors
  initVectors();



  if (dbe_) {  
    //// All MC photons
    // SC from reco photons

    dbe_->setCurrentFolder("Egamma/PhotonValidator/SimulationInfo");

    std::string histname = "nOfSimPhotons";    
    h_nSimPho_ = dbe_->book1D(histname,"# of Sim photons per event ",20,-0.5,19.5);
    histname = "SimPhoE";    
    h_SimPhoE_ = dbe_->book1D(histname,"Sim photon energy spectrum",eBin,eMin,eMax);
    histname = "SimPhoEt";    
    h_SimPhoEt_ = dbe_->book1D(histname,"Sim photon tranverse energy spectrum",etBin,etMin,etMax);
    h_SimPhoEta_ = dbe_->book1D("SimPhoEta"," Sim Photon Eta ",etaBin,etaMin, etaMax) ;
    h_SimPhoPhi_ = dbe_->book1D("SimPhoPhi"," Sim Photon  Phi ",phiBin,phiMin,phiMax) ;
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
    h_SimConvPhi_[1] = dbe_->book1D("VisSimConvPhi"," Visible Sim Conversion  Phi ",phiBin,phiMin,phiMax) ;
    h_SimConvR_[1] = dbe_->book1D("VisSimConvR"," Visible Sim Conversion Radius ",rBin,rMin,rMax) ;
    h_SimConvZ_[1] = dbe_->book1D("VisSimConvZ"," Visible Sim Conversion Z ",zBin,zMin,zMax) ;
    h_simTkPt_ = dbe_->book1D("simTkPt","Sim conversion tracks pt ",etBin,0.,etMax);
   

    dbe_->setCurrentFolder("Egamma/PhotonValidator/Photons");

    //// Reconstructed photons
    histname = "recoEffVsEta";
    phoEffEta_ =  dbe_->book1D(histname,histname,etaBin,etaMin, etaMax);
    histname = "recoEffVsPhi";
    phoEffPhi_ =  dbe_->book1D(histname,histname,phiBin,phiMin, phiMax);


    h_phoEta_[0] = dbe_->book1D("phoEta"," Photon Eta ",etaBin,etaMin, etaMax) ;
    h_phoPhi_[0] = dbe_->book1D("phoPhi"," Photon  Phi ",phiBin,phiMin,phiMax) ;

    h_phoDEta_[0] = dbe_->book1D("phoDEta"," Photon Eta(rec)-Eta(true) ",dEtaBin,dEtaMin, dEtaMax) ;
    h_phoDPhi_[0] = dbe_->book1D("phoDPhi"," Photon  Phi(rec)-Phi(true) ",dPhiBin,dPhiMin,dPhiMax) ;

    h_scEta_[0] =   dbe_->book1D("scEta"," SC Eta ",etaBin,etaMin, etaMax);
    h_scPhi_[0] =   dbe_->book1D("scPhi"," SC Phi ",phiBin,phiMin,phiMax);


    histname = "scE";
    h_scE_[0][0] = dbe_->book1D(histname+"All"," SC Energy: All Ecal  ",eBin,eMin, eMax);
    h_scE_[0][1] = dbe_->book1D(histname+"Barrel"," SC Energy: Barrel ",eBin,eMin, eMax);
    h_scE_[0][2] = dbe_->book1D(histname+"Endcap"," SC Energy: Endcap ",eBin,eMin, eMax);

    histname = "scEt";
    h_scEt_[0][0] = dbe_->book1D(histname+"All"," SC Et: All Ecal ",etBin,etMin, etMax) ;
    h_scEt_[0][1] = dbe_->book1D(histname+"Barrel"," SC Et: Barrel",etBin,etMin, etMax) ;
    h_scEt_[0][2] = dbe_->book1D(histname+"Endcap"," SC Et: Endcap",etBin,etMin, etMax) ;



    histname = "r9";
    h_r9_[0][0] = dbe_->book1D(histname+"All",   " r9: All Ecal",r9Bin,r9Min, r9Max) ;
    h_r9_[0][1] = dbe_->book1D(histname+"Barrel"," r9: Barrel ",r9Bin,r9Min, r9Max) ;
    h_r9_[0][2] = dbe_->book1D(histname+"Endcap"," r9: Endcap ",r9Bin,r9Min, r9Max) ;
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





    dbe_->setCurrentFolder("Egamma/PhotonValidator/ConversionInfo");

    histname = "convEffVsEtaTwoTracks";
    convEffEtaTwoTracks_ =  dbe_->book1D(histname,histname,etaBin2,etaMin, etaMax);
    histname = "convEffVsPhiTwoTracks";
    convEffPhiTwoTracks_ =  dbe_->book1D(histname,histname,phiBin,phiMin,phiMax);
    histname = "convEffVsRTwoTracks";
    convEffRTwoTracks_ =  dbe_->book1D(histname,histname,rBin,rMin, rMax);
    histname = "convEffVsZTwoTracks";
    convEffZTwoTracks_ =  dbe_->book1D(histname,histname,zBin,zMin,zMax);



    histname="nConv";
    h_nConv_[0][0] = dbe_->book1D(histname+"All","Number Of Conversions per isolated candidates per events: All Ecal  ",10,-0.5, 9.5);
    h_nConv_[0][1] = dbe_->book1D(histname+"Barrel","Number Of Conversions per isolated candidates per events: Ecal Barrel  ",10,-0.5, 9.5);
    h_nConv_[0][2] = dbe_->book1D(histname+"Endcap","Number Of Conversions per isolated candidates per events: Ecal Endcap ",10,-0.5, 9.5);

    h_convEta_[0] = dbe_->book1D("convEta"," converted Photon Eta ",etaBin,etaMin, etaMax) ;
    h_convPhi_[0] = dbe_->book1D("convPhi"," converted Photon  Phi ",phiBin,phiMin,phiMax) ;

    histname = "eConvRes";
    h_convERes_[0][0] = dbe_->book1D(histname+"All"," Conversion rec/true Energy: All ecal ", resBin,resMin, resMax);
    h_convERes_[0][1] = dbe_->book1D(histname+"Barrel"," Conversion rec/true Energy: Barrel ",resBin,resMin, resMax);
    h_convERes_[0][2] = dbe_->book1D(histname+"Endcap"," Conversion rec/true Energy: Endcap ",resBin,resMin, resMax);



    histname="r9VsTracks";
    h_r9VsNofTracks_[0][0] = dbe_->book2D(histname+"All"," photons r9 vs nTracks from conversions: All Ecal",r9Bin,r9Min, r9Max, 3, -0.5, 2.5) ;
    h_r9VsNofTracks_[0][1] = dbe_->book2D(histname+"Barrel"," photons r9 vs nTracks from conversions: Barrel Ecal",r9Bin,r9Min, r9Max, 3, -0.5, 2.5) ;
    h_r9VsNofTracks_[0][2] = dbe_->book2D(histname+"Endcap"," photons r9 vs nTracks from conversions: Endcap Ecal",r9Bin,r9Min, r9Max, 3, -0.5, 2.5) ;

    histname="EoverPtracks";
    h_EoverPTracks_[0][0] = dbe_->book1D(histname+"All"," photons conversion E/p: all Ecal ",100, 0., 5.);
    h_EoverPTracks_[0][1] = dbe_->book1D(histname+"Barrel"," photons conversion E/p: Barrel Ecal",100, 0., 5.);
    h_EoverPTracks_[0][2] = dbe_->book1D(histname+"All"," photons conversion E/p: Endcap Ecal ",100, 0., 5.);


    histname="nHitsVsEta";
    nHitsVsEta_[0] =  dbe_->book2D(histname,histname,etaBin,etaMin, etaMax,25,0., 25.);
    histname="h_nHitsVsEta";
    h_nHitsVsEta_[0] =  dbe_->book1D(histname,histname,etaBin,etaMin, etaMax);

    histname="nHitsVsR";
    nHitsVsR_[0] =  dbe_->book2D(histname,histname,rBin,rMin, rMax,25,0.,25);
    histname="h_nHitsVsR";
    h_nHitsVsR_[0] =  dbe_->book1D(histname,histname,rBin,rMin, rMax);


    h_tkChi2_[0] = dbe_->book1D("tkChi2"," Photons:Tracks from conversions: #chi^{2} of tracks", 100, 0., 20.0); 


    histname="hDPhiTracksAtVtx";
    h_DPhiTracksAtVtx_[0][0] =dbe_->book1D(histname+"All", " Photons:Tracks from conversions: #delta#phi Tracks at vertex: all Ecal",dPhiTracksBin,dPhiTracksMin,dPhiTracksMax); 
    h_DPhiTracksAtVtx_[0][1] =dbe_->book1D(histname+"Barrel", " Photons:Tracks from conversions: #delta#phi Tracks at vertex: Barrel Ecal",dPhiTracksBin,dPhiTracksMin,dPhiTracksMax); 
    h_DPhiTracksAtVtx_[0][2] =dbe_->book1D(histname+"Endcap", " Photons:Tracks from conversions: #delta#phi Tracks at vertex: Endcap Ecal",dPhiTracksBin,dPhiTracksMin,dPhiTracksMax); 
    histname="hDCotTracks";
    h_DCotTracks_[0][0]= dbe_->book1D(histname+"All"," Photons:Tracks from conversions #delta cotg(#Theta) Tracks: all Ecal ",dEtaTracksBin,dEtaTracksMin,dEtaTracksMax); 
    h_DCotTracks_[0][1]= dbe_->book1D(histname+"Barrel"," Photons:Tracks from conversions #delta cotg(#Theta) Tracks: Barrel Ecal ",dEtaTracksBin,dEtaTracksMin,dEtaTracksMax); 
    h_DCotTracks_[0][2]= dbe_->book1D(histname+"Encap"," Photons:Tracks from conversions #delta cotg(#Theta) Tracks: Endcap Ecal ",dEtaTracksBin,dEtaTracksMin,dEtaTracksMax); 
    histname="hInvMass";
    h_invMass_[0][0]= dbe_->book1D(histname+"All"," Photons:Tracks from conversion: Pair invariant mass: all Ecal ",100, 0., 1.5);
    h_invMass_[0][1]= dbe_->book1D(histname+"Barrel"," Photons:Tracks from conversion: Pair invariant mass: Barrel Ecal ",100, 0., 1.5);
    h_invMass_[0][2]= dbe_->book1D(histname+"Endcap"," Photons:Tracks from conversion: Pair invariant mass: Endcap Ecal ",100, 0., 1.5);
    histname="hDPhiTracksAtEcal";
    h_DPhiTracksAtEcal_[0][0]= dbe_->book1D(histname+"All"," Photons:Tracks from conversions:  #delta#phi at Ecal : all Ecal ",dPhiTracksBin,dPhiTracksMin,dPhiTracksMax); 
    h_DPhiTracksAtEcal_[0][1]= dbe_->book1D(histname+"Barrel"," Photons:Tracks from conversions:  #delta#phi at Ecal : Barrel Ecal ",dPhiTracksBin,dPhiTracksMin,dPhiTracksMax); 
    h_DPhiTracksAtEcal_[0][2]= dbe_->book1D(histname+"Endcap"," Photons:Tracks from conversions:  #delta#phi at Ecal : Endcap Ecal ",dPhiTracksBin,dPhiTracksMin,dPhiTracksMax); 
    histname="hDEtaTracksAtEcal";
    h_DEtaTracksAtEcal_[0][0]= dbe_->book1D(histname+"All"," Photons:Tracks from conversions:  #delta#eta at Ecal : all Ecal ",dEtaTracksBin,dEtaTracksMin,dEtaTracksMax); 
    h_DEtaTracksAtEcal_[0][1]= dbe_->book1D(histname+"Barrel"," Photons:Tracks from conversions:  #delta#eta at Ecal : Barrel Ecal ",dEtaTracksBin,dEtaTracksMin,dEtaTracksMax); 
    h_DEtaTracksAtEcal_[0][2]= dbe_->book1D(histname+"Endcap"," Photons:Tracks from conversions:  #delta#eta at Ecal : Endcap Ecal ",dEtaTracksBin,dEtaTracksMin,dEtaTracksMax); 
   
    h_convVtxRvsZ_[0] =   dbe_->book2D("convVtxRvsZ"," Photon Reco conversion vtx position",100, 0., 280.,200,0., 120.);
    h_zPVFromTracks_[0] =  dbe_->book1D("zPVFromTracks"," Photons: PV z from conversion tracks",100, -25., 25.);



  }

  
  return ;
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
  e.getByLabel("mergedtruth","MergedTrackTruth",ElectronTPHandle);
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

  nSimPho_=0;
  nSimConv_[0]=0;
  nSimConv_[1]=0;
  for ( std::vector<PhotonMCTruth>::const_iterator mcPho=mcPhotons.begin(); mcPho !=mcPhotons.end(); mcPho++) {
    if ( (*mcPho).fourMomentum().et() < minPhoEtCut_ ) continue;
  
    float mcPhi= (*mcPho).fourMomentum().phi();
    mcPhi_= phiNormalization(mcPhi);
    mcEta_= (*mcPho).fourMomentum().pseudoRapidity();   
    mcEta_ = etaTransformation(mcEta_, (*mcPho).primaryVertex().z() ); 
    
    mcConvR_= (*mcPho).vertex().perp();    
    mcConvZ_= (*mcPho).vertex().z();    
    mcConvY_= (*mcPho).vertex().y();    

    
    if ( ! (  fabs(mcEta_) <= BARL || ( fabs(mcEta_) >= END_LO && fabs(mcEta_) <=END_HI ) ) ) 
      continue;  // all ecal fiducial region


    nSimPho_++;
    h_SimPhoE_->Fill(  (*mcPho).fourMomentum().e());
    h_SimPhoEt_->Fill(  (*mcPho).fourMomentum().et());
    h_SimPhoEta_->Fill( mcEta_ ) ;
    h_SimPhoPhi_->Fill( mcPhi_ );

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


    ////////////////////////////////// extract info about simulated conversions 
    bool goodSimConversion=false;
    bool visibleConversion=false;
    bool visibleConversionsWithTwoSimTracks=false;
    if (  (*mcPho).isAConversion() == 1 ) {
      nSimConv_[0]++;
      h_SimConvE_[0]->Fill(  (*mcPho).fourMomentum().e());
      h_SimConvEt_[0]->Fill(  (*mcPho).fourMomentum().et());
      h_SimConvEta_[0]->Fill( mcEta_ ) ;
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
      std::cout << " PhotonValidator  theConvTP_ size " <<   theConvTP_.size() << std::endl;	

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
      for (unsigned int f=0; f<etaintervalslarge_.size()-1; f++){
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
    float e3x3=   EcalClusterTools::e3x3(  *(  matchingPho.superCluster()->seed()  ), &ecalRecHitCollection, &(*topology)); 
    float r9 =e3x3/( matchingPho.superCluster()->rawEnergy()+ matchingPho.superCluster()->preshowerEnergy());

    h_scEta_[type]->Fill( matchingPho.superCluster()->eta() );
    h_scPhi_[type]->Fill( matchingPho.superCluster()->phi() );
    h_scE_[type][0]->Fill( matchingPho.superCluster()->energy() );
    h_scEt_[type][0]->Fill( matchingPho.superCluster()->energy()/cosh( matchingPho.superCluster()->eta()) );
    h_r9_[type][0]->Fill( r9 );
      
      
    h_phoEta_[type]->Fill( matchingPho.eta() );
    h_phoPhi_[type]->Fill( matchingPho.phi() );
    h_phoDEta_[0]->Fill (  matchingPho.eta() - (*mcPho).fourMomentum().eta() );
    h_phoDPhi_[0]->Fill (  matchingPho.phi() - mcPhi_ );
    h_phoE_[type][0]->Fill( matchingPho.energy() );
    h_phoEt_[type][0]->Fill( matchingPho.energy()/ cosh( matchingPho.eta()) );
      

      

    h_phoERes_[0][0]->Fill(matchingPho.energy() / (*mcPho).fourMomentum().e() );
    if ( r9 > 0.93 )  h_phoERes_[1][0]->Fill(matchingPho.energy() / (*mcPho).fourMomentum().e() );
    if ( r9 <= 0.93 )  h_phoERes_[2][0]->Fill(matchingPho.energy() / (*mcPho).fourMomentum().e() );
	 
          
    if ( phoIsInBarrel ) {
      h_scE_[type][1]->Fill( matchingPho.superCluster()->energy() );
      h_scEt_[type][1]->Fill( matchingPho.superCluster()->energy()/cosh( matchingPho.superCluster()->eta()) );
      h_r9_[type][1]->Fill( r9 );
      h_phoE_[type][1]->Fill( matchingPho.energy() );
      h_phoEt_[type][1]->Fill( matchingPho.energy()/ cosh( matchingPho.eta()) );
      h_nConv_[type][1]->Fill(float( matchingPho.conversions().size()));
      
      h_phoERes_[0][1]->Fill(matchingPho.energy() / (*mcPho).fourMomentum().e() );
      if ( r9 > 0.93 )  h_phoERes_[1][1]->Fill(matchingPho.energy() / (*mcPho).fourMomentum().e() );
      if ( r9 <= 0.93 )  h_phoERes_[2][1]->Fill(matchingPho.energy() / (*mcPho).fourMomentum().e() );
	
    }
    if ( phoIsInEndcap ) {
      h_scE_[type][2]->Fill( matchingPho.superCluster()->energy() );
      h_scEt_[type][2]->Fill( matchingPho.superCluster()->energy()/cosh( matchingPho.superCluster()->eta()) );
      h_r9_[type][2]->Fill( r9 );
      h_phoE_[type][2]->Fill( matchingPho.energy() );
      h_phoEt_[type][2]->Fill( matchingPho.energy()/ cosh( matchingPho.eta()) );
      h_nConv_[type][2]->Fill(float( matchingPho.conversions().size()));
      h_phoERes_[0][2]->Fill(matchingPho.energy() / (*mcPho).fourMomentum().e() );
      if ( r9 > 0.93 )  h_phoERes_[1][2]->Fill(matchingPho.energy() / (*mcPho).fourMomentum().e() );
      if ( r9 <= 0.93 )  h_phoERes_[2][2]->Fill(matchingPho.energy() / (*mcPho).fourMomentum().e() );
      
    }
      

    if ( ! (visibleConversion &&  visibleConversionsWithTwoSimTracks ) ) continue;

    h_nConv_[type][0]->Fill(float( matchingPho.conversions().size()));          
    ////////////////// plot quantitied related to conversions
    std::vector<reco::ConversionRef> conversions = matchingPho.conversions();
    for (unsigned int iConv=0; iConv<conversions.size(); iConv++) {

      reco::ConversionRef aConv=conversions[iConv];
       
      h_convEta_[type]->Fill( aConv->caloCluster()[0]->eta() );
      h_convPhi_[type]->Fill( aConv->caloCluster()[0]->phi() );
      h_convERes_[0][0]->Fill( aConv->caloCluster()[0]->energy() / (*mcPho).fourMomentum().e() );
      if ( phoIsInBarrel ) h_convERes_[0][1]->Fill(aConv->caloCluster()[0]->energy() / (*mcPho).fourMomentum().e() );
      if ( phoIsInEndcap ) h_convERes_[0][2]->Fill(aConv->caloCluster()[0]->energy() / (*mcPho).fourMomentum().e() );
   

      h_r9VsNofTracks_[type][0]->Fill( r9, aConv->nTracks() ) ; 
      if ( phoIsInBarrel ) h_r9VsNofTracks_[type][1]->Fill( r9, aConv->nTracks() ) ; 
      if ( phoIsInEndcap ) h_r9VsNofTracks_[type][2]->Fill( r9, aConv->nTracks() ) ; 


      std::map<reco::TrackRef,TrackingParticleRef> myAss;
      std::map<reco::TrackRef,TrackingParticleRef>::const_iterator itAss;
      std::map<reco::TrackRef,TrackingParticleRef>::const_iterator itAssMin;
      std::map<reco::TrackRef,TrackingParticleRef>::const_iterator itAssMax;
      //     
      std::vector<reco::TrackRef> tracks = aConv->tracks();
      int nAssT2=0;
      int nAssT1=0;
      float px=0;
      float py=0;
      float pz=0;
      float e=0;
      for (unsigned int i=0; i<tracks.size(); i++) {

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

	/////////// check track association
	TrackingParticleRef myTP;
	for (size_t j = 0; j < RtoSCollPtrs.size(); j++) {          
	  reco::RecoToSimCollection q = *(RtoSCollPtrs[j]);
	  
	  RefToBase<reco::Track> myTk( aConv->tracks()[i] );
	  
	  if( q.find(myTk ) != q.end() ) {
	    std::vector<std::pair<TrackingParticleRef, double> > tp = q[myTk];
	    for (int itp=0; itp<tp.size(); itp++) {
	      myTP=tp[itp].first;
	      //std::cout << " associated with TP " << myTP->pdgId() << " pt " << sqrt(myTP->momentum().perp2()) << std::endl;
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

      h_invMass_[type][0] ->Fill( invM);
      if ( phoIsInBarrel ) h_invMass_[type][1] ->Fill(invM);
      if ( phoIsInEndcap ) h_invMass_[type][2] ->Fill(invM);

      


      ////////// Numerators for conversion efficiencies: both tracks are associated
      if ( nAssT2 ==2 ) {
	for (unsigned int f=0; f<etaintervalslarge_.size()-1; f++){
	  if (mcEta_>etaintervalslarge_[f]&&
	      mcEta_<etaintervalslarge_[f+1]) {
	    totMatchedSimConvEtaTwoTracks_[f]++;
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
      }
      





    } // loop over conversions

    
    
  } // End loop over simulated Photons
 

  h_nSimPho_->Fill(float(nSimPho_));
  h_nSimConv_[0]->Fill(float(nSimConv_[0]));
  h_nSimConv_[1]->Fill(float(nSimConv_[1]));
  

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


  doProfileX( nHitsVsEta_[0], h_nHitsVsEta_[0]);
  doProfileX( nHitsVsR_[0], h_nHitsVsR_[0]);

  fillPlotFromVectors(phoEffEta_,totMatchedSimPhoEta_,totSimPhoEta_,"effic");
  fillPlotFromVectors(phoEffPhi_,totMatchedSimPhoPhi_,totSimPhoPhi_,"effic"); 
 //
  fillPlotFromVectors(convEffEtaTwoTracks_,totMatchedSimConvEtaTwoTracks_,totSimConvEta_,"effic");
  fillPlotFromVectors(convEffPhiTwoTracks_,totMatchedSimConvPhiTwoTracks_,totSimConvPhi_,"effic");
  fillPlotFromVectors(convEffRTwoTracks_,totMatchedSimConvRTwoTracks_,totSimConvR_,"effic");
  fillPlotFromVectors(convEffZTwoTracks_,totMatchedSimConvZTwoTracks_,totSimConvZ_,"effic");
  



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
