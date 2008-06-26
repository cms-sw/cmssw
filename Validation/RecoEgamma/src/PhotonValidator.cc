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
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
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
 **  $Date:  $ 
 **  $Revision: $
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

  double dEtaMin = parameters_.getParameter<double>("dEtaMin");
  double dEtaMax = parameters_.getParameter<double>("dEtaMax");
  int dEtaBin = parameters_.getParameter<int>("dEtaBin");
 
  double phiMin = parameters_.getParameter<double>("phiMin");
  double phiMax = parameters_.getParameter<double>("phiMax");
  int    phiBin = parameters_.getParameter<int>("phiBin");

  double dPhiMin = parameters_.getParameter<double>("dPhiMin");
  double dPhiMax = parameters_.getParameter<double>("dPhiMax");
  int    dPhiBin = parameters_.getParameter<int>("dPhiBin");
 

 
  double r9Min = parameters_.getParameter<double>("r9Min"); 
  double r9Max = parameters_.getParameter<double>("r9Max"); 
  int r9Bin = parameters_.getParameter<int>("r9Bin");

  double dPhiTracksMin = parameters_.getParameter<double>("dPhiTracksMin"); 
  double dPhiTracksMax = parameters_.getParameter<double>("dPhiTracksMax"); 
  int dPhiTracksBin = parameters_.getParameter<int>("dPhiTracksBin"); 
  
  double dEtaTracksMin = parameters_.getParameter<double>("dEtaTracksMin"); 
  double dEtaTracksMax = parameters_.getParameter<double>("dEtaTracksMax"); 
  int    dEtaTracksBin = parameters_.getParameter<int>("dEtaTracksBin"); 

  if (dbe_) {  
    //// All MC photons
    // SC from reco photons

    dbe_->setCurrentFolder("Egamma/PhotonValidator/SimulationInfo");

    std::string histname = "nIsoTracks";    



    dbe_->setCurrentFolder("Egamma/PhotonValidator/Photons");

    //// Reconstructed photons
    histname = "recoEffVsEta";
    effEta_ =  dbe_->bookProfile(histname,histname,etaBin,etaMin, etaMax,etaBin,0, 1.);
    histname = "recoEffVsPhi";
    effPhi_ =  dbe_->bookProfile(histname,histname,phiBin,phiMin, phiMax,phiBin,0, 1.);

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

    histname="nConv";
    h_nConv_[0][0] = dbe_->book1D(histname+"All","Number Of Conversions per isolated candidates per events: All Ecal  ",10,-0.5, 9.5);
    h_nConv_[0][1] = dbe_->book1D(histname+"Barrel","Number Of Conversions per isolated candidates per events: Ecal Barrel  ",10,-0.5, 9.5);
    h_nConv_[0][2] = dbe_->book1D(histname+"Endcap","Number Of Conversions per isolated candidates per events: Ecal Endcap ",10,-0.5, 9.5);

    h_convEta_[0] = dbe_->book1D("convEta"," converted Photon Eta ",etaBin,etaMin, etaMax) ;
    h_convPhi_[0] = dbe_->book1D("convPhi"," converted Photon  Phi ",phiBin,phiMin,phiMax) ;

    histname="r9VsTracks";
    h_r9VsNofTracks_[0][0] = dbe_->book2D(histname+"All"," photons r9 vs nTracks from conversions: All Ecal",r9Bin,r9Min, r9Max, 3, -0.5, 2.5) ;
    h_r9VsNofTracks_[0][1] = dbe_->book2D(histname+"Barrel"," photons r9 vs nTracks from conversions: Barrel Ecal",r9Bin,r9Min, r9Max, 3, -0.5, 2.5) ;
    h_r9VsNofTracks_[0][2] = dbe_->book2D(histname+"Endcap"," photons r9 vs nTracks from conversions: Endcap Ecal",r9Bin,r9Min, r9Max, 3, -0.5, 2.5) ;

    histname="EoverPtracks";
    h_EoverPTracks_[0][0] = dbe_->book1D(histname+"All"," photons conversion E/p: all Ecal ",100, 0., 5.);
    h_EoverPTracks_[0][1] = dbe_->book1D(histname+"Barrel"," photons conversion E/p: Barrel Ecal",100, 0., 5.);
    h_EoverPTracks_[0][2] = dbe_->book1D(histname+"All"," photons conversion E/p: Endcap Ecal ",100, 0., 5.);

    histname="pTknHitsVsEta";
    // p_tk_nHitsVsEta_[0] =  dbe_->bookProfile(histname," Photons:Tracks from conversions: mean numb of  Hits vs Eta",etaBin,etaMin, etaMax);
    p_tk_nHitsVsEta_[0] =  dbe_->bookProfile(histname,histname,etaBin,etaMin, etaMax,etaBin,0, 16);
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



    double wt=0.;
    if (matched )  wt=1.; 
    effEta_ ->Fill ( mcEta_, wt);
    effPhi_ ->Fill ( mcPhi_, wt);

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
      
    h_nConv_[type][0]->Fill(float( matchingPho.conversions().size()));          
 
    ////////////////// plot quantitied related to conversions
    std::vector<reco::ConversionRef> conversions = matchingPho.conversions();
    for (unsigned int iConv=0; iConv<conversions.size(); iConv++) {
      if ( conversions[iConv]->nTracks() <2 ) continue; 
    } // loop over conversions

    
    
  } // End loop over simulated Photons
 
  

}




void PhotonValidator::endJob()
{



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



