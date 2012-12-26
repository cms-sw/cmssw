#include <iostream>
//
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
//
#include "Validation/RecoEgamma/plugins/TkConvValidator.h"

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
#include "MagneticField/Engine/interface/MagneticField.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"

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
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/GeometrySurface/interface/BoundCylinder.h"
#include "DataFormats/GeometrySurface/interface/BoundDisk.h"
#include "DataFormats/GeometrySurface/interface/SimpleCylinderBounds.h"
#include "DataFormats/GeometrySurface/interface/SimpleDiskBounds.h"
#include "DataFormats/Math/interface/deltaPhi.h"

//
#include "RecoEgamma/EgammaMCTools/interface/PhotonMCTruthFinder.h"
#include "RecoEgamma/EgammaMCTools/interface/PhotonMCTruth.h"
#include "RecoEgamma/EgammaMCTools/interface/ElectronMCTruth.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"
#include "RecoCaloTools/MetaCollections/interface/CaloRecHitMetaCollections.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"

#include "RecoEgamma/EgammaPhotonAlgos/interface/ConversionHitChecker.h"

//
//
#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TTree.h"
#include "TVector3.h"
#include "TProfile.h"
//
/** \class TkConvValidator
 **
 **
 **  $Id: TkConvValidator
 **  $Date: 2012/02/01 21:27:39 $
 **  $Revision: 1.5 $
 **  \author N.Marinelli - Univ. of Notre Dame
 **
 ***/

using namespace std;


TkConvValidator::TkConvValidator( const edm::ParameterSet& pset )
  {

    fName_     = pset.getUntrackedParameter<std::string>("Name");
    verbosity_ = pset.getUntrackedParameter<int>("Verbosity");
    parameters_ = pset;
    
    photonCollectionProducer_ = pset.getParameter<std::string>("phoProducer");
    photonCollection_ = pset.getParameter<std::string>("photonCollection");

    conversionCollectionProducer_ = pset.getParameter<std::string>("convProducer");
    conversionCollection_ = pset.getParameter<std::string>("conversionCollection");
    // conversionTrackProducer_ = pset.getParameter<std::string>("trackProducer");
    dqmpath_ = pset.getParameter<std::string>("dqmpath");
    minPhoEtCut_ = pset.getParameter<double>("minPhoEtCut");
    generalTracksOnly_ = pset.getParameter<bool>("generalTracksOnly");
    arbitratedMerged_ = pset.getParameter<bool>("arbitratedMerged");
    arbitratedEcalSeeded_ = pset.getParameter<bool>("arbitratedEcalSeeded");
    ecalalgotracks_ = pset.getParameter<bool>("ecalalgotracks");
    highPurity_ = pset.getParameter<bool>("highPurity");
    minProb_ = pset.getParameter<double>("minProb");
    maxHitsBeforeVtx_ = pset.getParameter<uint>("maxHitsBeforeVtx");
    minLxy_           = pset.getParameter<double>("minLxy");
    isRunCentrally_=   pset.getParameter<bool>("isRunCentrally");
  }





TkConvValidator::~TkConvValidator() {}




void  TkConvValidator::beginJob() {

  nEvt_=0;
  nEntry_=0;
  nRecConv_=0;
  nRecConvAss_=0;
  nRecConvAssWithEcal_=0;

  nInvalidPCA_=0;

  dbe_ = 0;
  dbe_ = edm::Service<DQMStore>().operator->();


  double etMin = parameters_.getParameter<double>("etMin");
  double etMax = parameters_.getParameter<double>("etMax");
  int etBin = parameters_.getParameter<int>("etBin");


  double resMin = parameters_.getParameter<double>("resMin");
  double resMax = parameters_.getParameter<double>("resMax");
  int resBin = parameters_.getParameter<int>("resBin");

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

  double dPhiTracksMin = parameters_.getParameter<double>("dPhiTracksMin");
  double dPhiTracksMax = parameters_.getParameter<double>("dPhiTracksMax");
  int dPhiTracksBin = parameters_.getParameter<int>("dPhiTracksBin");

  double eoverpMin = parameters_.getParameter<double>("eoverpMin");
  double eoverpMax = parameters_.getParameter<double>("eoverpMax");
  int    eoverpBin = parameters_.getParameter<int>("eoverpBin");


  //  double dEtaTracksMin = parameters_.getParameter<double>("dEtaTracksMin");  // unused
  //  double dEtaTracksMax = parameters_.getParameter<double>("dEtaTracksMax"); // unused
  //  int    dEtaTracksBin = parameters_.getParameter<int>("dEtaTracksBin");  // unused

  double dCotTracksMin = parameters_.getParameter<double>("dCotTracksMin");
  double dCotTracksMax = parameters_.getParameter<double>("dCotTracksMax");
  int    dCotTracksBin = parameters_.getParameter<int>("dCotTracksBin");


  double chi2Min = parameters_.getParameter<double>("chi2Min");
  double chi2Max = parameters_.getParameter<double>("chi2Max");


  double rMinForXray = parameters_.getParameter<double>("rMinForXray");
  double rMaxForXray = parameters_.getParameter<double>("rMaxForXray");
  int    rBinForXray = parameters_.getParameter<int>("rBinForXray");
  double zMinForXray = parameters_.getParameter<double>("zMinForXray");
  double zMaxForXray = parameters_.getParameter<double>("zMaxForXray");
  int    zBinForXray = parameters_.getParameter<int>("zBinForXray");
  int    zBin2ForXray = parameters_.getParameter<int>("zBin2ForXray");

  minPhoPtForEffic = parameters_.getParameter<double>("minPhoPtForEffic");
  maxPhoEtaForEffic = parameters_.getParameter<double>("maxPhoEtaForEffic");
  maxPhoZForEffic = parameters_.getParameter<double>("maxPhoZForEffic");
  maxPhoRForEffic = parameters_.getParameter<double>("maxPhoRForEffic");
  minPhoPtForPurity = parameters_.getParameter<double>("minPhoPtForPurity");
  maxPhoEtaForPurity = parameters_.getParameter<double>("maxPhoEtaForPurity");
  maxPhoZForPurity = parameters_.getParameter<double>("maxPhoZForPurity");
  maxPhoRForPurity = parameters_.getParameter<double>("maxPhoRForPurity");

  if (dbe_) {

    //// All MC photons
    // SC from reco photons

    //TString simfolder = TString(
    std::string simpath = dqmpath_ + "SimulationInfo";
    dbe_->setCurrentFolder(simpath);
    //
    // simulation information about conversions
    /// Histograms for efficiencies
    std::string histname = "nOfSimConversions";
    h_nSimConv_[0] = dbe_->book1D(histname,"# of Sim conversions per event ",20,-0.5,19.5);
    /// Denominators
    histname = "h_AllSimConvEta";
    h_AllSimConv_[0] =  dbe_->book1D(histname," All conversions: simulated #eta",etaBin2,etaMin,etaMax);
    histname = "h_AllSimConvPhi";
    h_AllSimConv_[1] =  dbe_->book1D(histname," All conversions: simulated #phi",phiBin,phiMin,phiMax);
    histname = "h_AllSimConvR";
    h_AllSimConv_[2] =  dbe_->book1D(histname," All conversions: simulated R",rBin,rMin,rMax);
    histname = "h_AllSimConvZ";
    h_AllSimConv_[3] =  dbe_->book1D(histname," All conversions: simulated Z",zBin,zMin,zMax);
    histname = "h_AllSimConvEt";
    h_AllSimConv_[4] =  dbe_->book1D(histname," All conversions: simulated Et",etBin,etMin,etMax);
    //
    histname = "nOfVisSimConversions";
    h_nSimConv_[1] = dbe_->book1D(histname,"# of Sim conversions per event ",20,-0.5,19.5);
    histname = "h_VisSimConvEta";
    h_VisSimConv_[0] =  dbe_->book1D(histname," All vis conversions: simulated #eta",etaBin2,etaMin, etaMax);
    histname = "h_VisSimConvPhi";
    h_VisSimConv_[1] =  dbe_->book1D(histname," All vis conversions: simulated #phi",phiBin,phiMin, phiMax);
    histname = "h_VisSimConvR";
    h_VisSimConv_[2] =  dbe_->book1D(histname," All vis conversions: simulated R",rBin,rMin,rMax);
    histname = "h_VisSimConvZ";
    h_VisSimConv_[3] =  dbe_->book1D(histname," All vis conversions: simulated Z",zBin,zMin, zMax);
    histname = "h_VisSimConvEt";
    h_VisSimConv_[4] =  dbe_->book1D(histname," All vis conversions: simulated Et",etBin,etMin, etMax);

    //
    histname = "h_SimConvTwoMTracksEta";
    h_SimConvTwoMTracks_[0] =  dbe_->book1D(histname," All vis conversions with 2 reco-matching tracks: simulated #eta",etaBin2,etaMin, etaMax);
    histname = "h_SimConvTwoMTracksPhi";
    h_SimConvTwoMTracks_[1] =  dbe_->book1D(histname," All vis conversions with 2 reco-matching tracks: simulated #phi",phiBin,phiMin, phiMax);
    histname = "h_SimConvTwoMTracksR";
    h_SimConvTwoMTracks_[2] =  dbe_->book1D(histname," All vis conversions with 2 reco-matching tracks: simulated R",rBin,rMin, rMax);
    histname = "h_SimConvTwoMTracksZ";
    h_SimConvTwoMTracks_[3] =  dbe_->book1D(histname," All vis conversions with 2 reco-matching tracks: simulated Z",zBin,zMin, zMax);
    histname = "h_SimConvTwoMTracksEt";
    h_SimConvTwoMTracks_[4] =  dbe_->book1D(histname," All vis conversions with 2 reco-matching tracks: simulated Et",etBin,etMin, etMax);
    //
    histname = "h_SimConvTwoTracksEta";
    h_SimConvTwoTracks_[0] =  dbe_->book1D(histname," All vis conversions with 2 reco  tracks: simulated #eta",etaBin2,etaMin, etaMax);
    histname = "h_SimConvTwoTracksPhi";
    h_SimConvTwoTracks_[1] =  dbe_->book1D(histname," All vis conversions with 2 reco tracks: simulated #phi",phiBin,phiMin, phiMax);
    histname = "h_SimConvTwoTracksR";
    h_SimConvTwoTracks_[2] =  dbe_->book1D(histname," All vis conversions with 2 reco tracks: simulated R",rBin,rMin, rMax);
    histname = "h_SimConvTwoTracksZ";
    h_SimConvTwoTracks_[3] =  dbe_->book1D(histname," All vis conversions with 2 reco tracks: simulated Z",zBin,zMin, zMax);
    histname = "h_SimConvTwoTracksEt";
    h_SimConvTwoTracks_[4] =  dbe_->book1D(histname," All vis conversions with 2 reco tracks: simulated Et",etBin,etMin, etMax);
    //
    histname = "h_SimConvTwoMTracksEtaAndVtxPGT0";
    h_SimConvTwoMTracksAndVtxPGT0_[0] =  dbe_->book1D(histname," All vis conversions with 2 reco-matching tracks + vertex: simulated #eta",etaBin2,etaMin, etaMax);
    histname = "h_SimConvTwoMTracksPhiAndVtxPGT0";
    h_SimConvTwoMTracksAndVtxPGT0_[1] =  dbe_->book1D(histname," All vis conversions with 2 reco-matching tracks + vertex: simulated #phi",phiBin,phiMin, phiMax);
    histname = "h_SimConvTwoMTracksRAndVtxPGT0";
    h_SimConvTwoMTracksAndVtxPGT0_[2] =  dbe_->book1D(histname," All vis conversions with 2 reco-matching tracks + vertex: simulated R",rBin,rMin, rMax);
    histname = "h_SimConvTwoMTracksZAndVtxPGT0";
    h_SimConvTwoMTracksAndVtxPGT0_[3] =  dbe_->book1D(histname," All vis conversions with 2 reco-matching tracks + vertex: simulated Z",zBin,zMin, zMax);
    histname = "h_SimConvTwoMTracksEtAndVtxPGT0";
    h_SimConvTwoMTracksAndVtxPGT0_[4] =  dbe_->book1D(histname," All vis conversions with 2 reco-matching tracks + vertex: simulated Et",etBin,etMin, etMax);

    //
    histname = "h_SimConvTwoMTracksEtaAndVtxPGT0005";
    h_SimConvTwoMTracksAndVtxPGT0005_[0] =  dbe_->book1D(histname," All vis conversions with 2 reco-matching tracks + vertex: simulated #eta",etaBin2,etaMin, etaMax);
    histname = "h_SimConvTwoMTracksPhiAndVtxPGT0005";
    h_SimConvTwoMTracksAndVtxPGT0005_[1] =  dbe_->book1D(histname," All vis conversions with 2 reco-matching tracks + vertex: simulated #phi",phiBin,phiMin, phiMax);
    histname = "h_SimConvTwoMTracksRAndVtxPGT0005";
    h_SimConvTwoMTracksAndVtxPGT0005_[2] =  dbe_->book1D(histname," All vis conversions with 2 reco-matching tracks + vertex: simulated R",rBin,rMin, rMax);
    histname = "h_SimConvTwoMTracksZAndVtxPGT0005";
    h_SimConvTwoMTracksAndVtxPGT0005_[3] =  dbe_->book1D(histname," All vis conversions with 2 reco-matching tracks + vertex: simulated Z",zBin,zMin, zMax);
    histname = "h_SimConvTwoMTracksEtAndVtxPGT0005";
    h_SimConvTwoMTracksAndVtxPGT0005_[4] =  dbe_->book1D(histname," All vis conversions with 2 reco-matching tracks + vertex: simulated Et",etBin,etMin, etMax);

    histname = "h_SimRecConvTwoMTracksEta";
    h_SimRecConvTwoMTracks_[0] =  dbe_->book1D(histname," All vis conversions with 2 reco-matching tracks: simulated #eta",etaBin2,etaMin, etaMax);
    histname = "h_SimRecConvTwoMTracksPhi";
    h_SimRecConvTwoMTracks_[1] =  dbe_->book1D(histname," All vis conversions with 2 reco-matching tracks: simulated #phi",phiBin,phiMin, phiMax);
    histname = "h_SimRecConvTwoMTracksR";
    h_SimRecConvTwoMTracks_[2] =  dbe_->book1D(histname," All vis conversions with 2 reco-matching tracks: simulated R",rBin,rMin, rMax);
    histname = "h_SimRecConvTwoMTracksZ";
    h_SimRecConvTwoMTracks_[3] =  dbe_->book1D(histname," All vis conversions with 2 reco-matching tracks: simulated Z",zBin,zMin, zMax);
    histname = "h_SimRecConvTwoMTracksEt";
    h_SimRecConvTwoMTracks_[4] =  dbe_->book1D(histname," All vis conversions with 2 reco-matching tracks: simulated Et",etBin,etMin, etMax);
    //


    h_SimConvEtaPix_[0] = dbe_->book1D("simConvEtaPix"," sim converted Photon Eta: Pix ",etaBin,etaMin, etaMax) ;
    h_simTkPt_ = dbe_->book1D("simTkPt","Sim conversion tracks pt ",etBin*3,0.,etMax);
    h_simTkEta_ = dbe_->book1D("simTkEta","Sim conversion tracks eta ",etaBin,etaMin,etaMax);

    h_simConvVtxRvsZ_[0] =   dbe_->book2D("simConvVtxRvsZAll"," Photon Sim conversion vtx position",zBinForXray, zMinForXray, zMaxForXray, rBinForXray, rMinForXray, rMaxForXray);
    h_simConvVtxRvsZ_[1] =   dbe_->book2D("simConvVtxRvsZBarrel"," Photon Sim conversion vtx position",zBinForXray, zMinForXray, zMaxForXray, rBinForXray, rMinForXray, rMaxForXray);
    h_simConvVtxRvsZ_[2] =   dbe_->book2D("simConvVtxRvsZEndcap"," Photon Sim conversion vtx position",zBin2ForXray, zMinForXray, zMaxForXray, rBinForXray, rMinForXray, rMaxForXray);
    h_simConvVtxRvsZ_[3] =   dbe_->book2D("simConvVtxRvsZBarrel2"," Photon Sim conversion vtx position when reco R<4cm",zBinForXray, zMinForXray, zMaxForXray, rBinForXray, rMinForXray, rMaxForXray);
    h_simConvVtxYvsX_ =   dbe_->book2D("simConvVtxYvsXTrkBarrel"," Photon Sim conversion vtx position, (x,y) eta<1 ",100, -80., 80., 100, -80., 80.);

    std::string convpath = dqmpath_ + "ConversionInfo";
    dbe_->setCurrentFolder(convpath);

    histname="nConv";
    h_nConv_[0][0] = dbe_->book1D(histname+"All","Number Of Conversions per isolated candidates per events: All Ecal  ",10,-0.5, 9.5);
    h_nConv_[0][1] = dbe_->book1D(histname+"Barrel","Number Of Conversions per isolated candidates per events: Ecal Barrel  ",10,-0.5, 9.5);
    h_nConv_[0][2] = dbe_->book1D(histname+"Endcap","Number Of Conversions per isolated candidates per events: Ecal Endcap ",10,-0.5, 9.5);
    h_nConv_[1][0] = dbe_->book1D(histname+"All_Ass","Number Of associated Conversions per isolated candidates per events: All Ecal  ",10,-0.5, 9.5);

    h_convEta_[0][0] = dbe_->book1D("convEta"," converted Photon  Eta ",etaBin,etaMin, etaMax) ;
    h_convEtaMatchSC_[0][0] = dbe_->book1D("convEtaMatchSC"," converted Photon  Eta when SC is matched ",etaBin,etaMin, etaMax) ;
    h_convEta2_[0][0] = dbe_->book1D("convEta2"," converted Photon  Eta ",etaBin2,etaMin, etaMax) ;
    h_convPhi_[0][0] = dbe_->book1D("convPhi"," converted Photon  Phi ",phiBin,phiMin,phiMax) ;
    h_convR_[0][0]  =  dbe_->book1D("convR"," converted photon R",rBin,rMin, rMax);
    h_convZ_[0][0] =  dbe_->book1D("convZ"," converted photon Z",zBin,zMin, zMax);
    h_convPt_[0][0] = dbe_->book1D("convPt","  conversions Transverse Energy: all eta ", etBin,etMin, etMax);

    h_convEta_[1][0] = dbe_->book1D("convEtaAss2"," Matched converted Photon  Eta ",etaBin2,etaMin, etaMax) ;
    h_convEta_[1][1] = dbe_->book1D("convEtaAss"," Matched converted Photon  Eta ",etaBin,etaMin, etaMax) ;
    h_convEtaMatchSC_[1][0] = dbe_->book1D("convEtaMatchSCAss"," converted Photon  Eta when SC is matched ",etaBin,etaMin, etaMax) ;
    h_convPhi_[1][0] = dbe_->book1D("convPhiAss"," Matched converted Photon  Phi ",phiBin,phiMin,phiMax) ;
    h_convR_[1][0]  =  dbe_->book1D("convRAss"," Matched converted photon R",rBin,rMin, rMax);
    h_convZ_[1][0] =  dbe_->book1D("convZAss"," Matched converted photon Z",zBin,zMin, zMax);
    h_convPt_[1][0] = dbe_->book1D("convPtAss","Matched conversions Transverse Energy: all eta ", etBin,etMin, etMax);

    h_convEta_[2][0] = dbe_->book1D("convEtaFake2"," Fake converted Photon  Eta ",etaBin2,etaMin, etaMax) ;
    h_convEta_[2][1] = dbe_->book1D("convEtaFake"," Fake converted Photon  Eta ",etaBin,etaMin, etaMax) ;
    h_convEtaMatchSC_[2][0] = dbe_->book1D("convEtaMatchSCFake"," converted Photon  Eta when SC is matched ",etaBin,etaMin, etaMax) ;
    h_convPhi_[2][0] = dbe_->book1D("convPhiFake"," Fake converted Photon  Phi ",phiBin,phiMin,phiMax) ;
    h_convR_[2][0]  =  dbe_->book1D("convRFake"," Fake converted photon R",rBin,rMin, rMax);
    h_convZ_[2][0] =  dbe_->book1D("convZFake"," Fake converted photon Z",zBin,zMin, zMax);
    h_convPt_[2][0] = dbe_->book1D("convPtFake","Fake conversions Transverse Energy: all eta ", etBin,etMin, etMax);

    h_convRplot_  =  dbe_->book1D("convRplot"," converted photon R",600, 0.,120.);
    h_convZplot_  =  dbe_->book1D("convZplot"," converted photon Z",320,-160.,160.);

    histname = "convSCdPhi";
    h_convSCdPhi_[0][0] =   dbe_->book1D(histname+"All","dPhi between SC and conversion",100, -0.1,0.1);
    h_convSCdPhi_[0][1] =   dbe_->book1D(histname+"Barrel"," dPhi between SC and conversion: Barrel",100, -0.1,0.1);
    h_convSCdPhi_[0][2] =   dbe_->book1D(histname+"Endcap"," dPhi between SC and conversion: Endcap",100, -0.1,0.1);
    h_convSCdPhi_[1][0] =   dbe_->book1D(histname+"All_Ass","dPhi between SC and conversion",100, -0.1,0.1);
    h_convSCdPhi_[1][1] =   dbe_->book1D(histname+"Barrel_Ass"," dPhi between SC and conversion: Barrel",100, -0.1,0.1);
    h_convSCdPhi_[1][2] =   dbe_->book1D(histname+"Endcap_Ass"," dPhi between SC and conversion: Endcap",100, -0.1,0.1);
    h_convSCdPhi_[2][0] =   dbe_->book1D(histname+"All_Fakes","dPhi between SC and conversion",100, -0.1,0.1);
    h_convSCdPhi_[2][1] =   dbe_->book1D(histname+"Barrel_Fakes"," dPhi between SC and conversion: Barrel",100, -0.1,0.1);
    h_convSCdPhi_[2][2] =   dbe_->book1D(histname+"Endcap_Fakes"," dPhi between SC and conversion: Endcap",100, -0.1,0.1);
    histname = "convSCdEta";
    h_convSCdEta_[0][0] =   dbe_->book1D(histname+"All"," dEta between SC and conversion",100, -0.1,0.1);
    h_convSCdEta_[0][1] =   dbe_->book1D(histname+"Barrel"," dEta between SC and conversion: Barrel",100, -0.1,0.1);
    h_convSCdEta_[0][2] =   dbe_->book1D(histname+"Endcap"," dEta between SC and conversion: Endcap",100, -0.1,0.1);
    h_convSCdEta_[1][0] =   dbe_->book1D(histname+"All_Ass"," dEta between SC and conversion",100, -0.1,0.1);
    h_convSCdEta_[1][1] =   dbe_->book1D(histname+"Barrel_Ass"," dEta between SC and conversion: Barrel",100, -0.1,0.1);
    h_convSCdEta_[1][2] =   dbe_->book1D(histname+"Endcap_Ass"," dEta between SC and conversion: Endcap",100, -0.1,0.1);
    h_convSCdEta_[2][0] =   dbe_->book1D(histname+"All_Fakes"," dEta between SC and conversion",100, -0.1,0.1);
    h_convSCdEta_[2][1] =   dbe_->book1D(histname+"Barrel_Fakes"," dEta between SC and conversion: Barrel",100, -0.1,0.1);
    h_convSCdEta_[2][2] =   dbe_->book1D(histname+"Endcap_Fakes"," dEta between SC and conversion: Endcap",100, -0.1,0.1);

    histname = "convPtRes";
    h_convPtRes_[0] = dbe_->book1D(histname+"All"," Conversion Pt rec/true : All ecal ", resBin,resMin, resMax);
    h_convPtRes_[1] = dbe_->book1D(histname+"Barrel"," Conversion Pt rec/true : Barrel ",resBin,resMin, resMax);
    h_convPtRes_[2] = dbe_->book1D(histname+"Endcap"," Conversion Pt rec/true : Endcap ",resBin,resMin, resMax);


    histname="hInvMass";
    h_invMass_[0][0]= dbe_->book1D(histname+"All_AllTracks"," Photons:Tracks from conversion: Pair invariant mass: all Ecal ",100, 0., 1.5);
    h_invMass_[0][1]= dbe_->book1D(histname+"Barrel_AllTracks"," Photons:Tracks from conversion: Pair invariant mass: Barrel Ecal ",100, 0., 1.5);
    h_invMass_[0][2]= dbe_->book1D(histname+"Endcap_AllTracks"," Photons:Tracks from conversion: Pair invariant mass: Endcap Ecal ",100, 0., 1.5);
    //
    h_invMass_[1][0]= dbe_->book1D(histname+"All_AssTracks"," Photons:Tracks from conversion: Pair invariant mass: all Ecal ",100, 0., 1.5);
    h_invMass_[1][1]= dbe_->book1D(histname+"Barrel_AssTracks"," Photons:Tracks from conversion: Pair invariant mass: Barrel Ecal ",100, 0., 1.5);
    h_invMass_[1][2]= dbe_->book1D(histname+"Endcap_AssTracks"," Photons:Tracks from conversion: Pair invariant mass: Endcap Ecal ",100, 0., 1.5);
    //
    h_invMass_[2][0]= dbe_->book1D(histname+"All_FakeTracks"," Photons:Tracks from conversion: Pair invariant mass: all Ecal ",100, 0., 1.5);
    h_invMass_[2][1]= dbe_->book1D(histname+"Barrel_FakeTracks"," Photons:Tracks from conversion: Pair invariant mass: Barrel Ecal ",100, 0., 1.5);
    h_invMass_[2][2]= dbe_->book1D(histname+"Endcap_FaleTracks"," Photons:Tracks from conversion: Pair invariant mass: Endcap Ecal ",100, 0., 1.5);



    histname="hDPhiTracksAtVtx";
    h_DPhiTracksAtVtx_[0][0] =dbe_->book1D(histname+"All", " Photons:Tracks from conversions: #delta#phi Tracks at vertex: all Ecal",dPhiTracksBin,dPhiTracksMin,dPhiTracksMax);
    h_DPhiTracksAtVtx_[0][1] =dbe_->book1D(histname+"Barrel", " Photons:Tracks from conversions: #delta#phi Tracks at vertex: Barrel Ecal",dPhiTracksBin,dPhiTracksMin,dPhiTracksMax);
    h_DPhiTracksAtVtx_[0][2] =dbe_->book1D(histname+"Endcap", " Photons:Tracks from conversions: #delta#phi Tracks at vertex: Endcap Ecal",dPhiTracksBin,dPhiTracksMin,dPhiTracksMax);
    h_DPhiTracksAtVtx_[1][0] =dbe_->book1D(histname+"All_Ass", " Photons:Tracks from conversions: #delta#phi Tracks at vertex: all Ecal",dPhiTracksBin,dPhiTracksMin,dPhiTracksMax);
    h_DPhiTracksAtVtx_[1][1] =dbe_->book1D(histname+"Barrel_Ass", " Photons:Tracks from conversions: #delta#phi Tracks at vertex: Barrel Ecal",dPhiTracksBin,dPhiTracksMin,dPhiTracksMax);
    h_DPhiTracksAtVtx_[1][2] =dbe_->book1D(histname+"Endcap_Ass", " Photons:Tracks from conversions: #delta#phi Tracks at vertex: Endcap Ecal",dPhiTracksBin,dPhiTracksMin,dPhiTracksMax);
    h_DPhiTracksAtVtx_[2][0] =dbe_->book1D(histname+"All_Fakes", " Photons:Tracks from conversions: #delta#phi Tracks at vertex: all Ecal",dPhiTracksBin,dPhiTracksMin,dPhiTracksMax);
    h_DPhiTracksAtVtx_[2][1] =dbe_->book1D(histname+"Barrel_Fakes", " Photons:Tracks from conversions: #delta#phi Tracks at vertex: Barrel Ecal",dPhiTracksBin,dPhiTracksMin,dPhiTracksMax);
    h_DPhiTracksAtVtx_[2][2] =dbe_->book1D(histname+"Endcap_Fakes", " Photons:Tracks from conversions: #delta#phi Tracks at vertex: Endcap Ecal",dPhiTracksBin,dPhiTracksMin,dPhiTracksMax);



    histname="hDPhiTracksAtVtxVsEta";
    h2_DPhiTracksAtVtxVsEta_ = dbe_->book2D(histname+"All","  Photons:Tracks from conversions: #delta#phi Tracks at vertex vs #eta",etaBin2,etaMin, etaMax,100, -0.5, 0.5);
    histname="pDPhiTracksAtVtxVsEta";
    p_DPhiTracksAtVtxVsEta_ = dbe_->bookProfile(histname+"All"," Photons:Tracks from conversions: #delta#phi Tracks at vertex vs #eta ",etaBin2,etaMin, etaMax, 100, -0.5, 0.5,"");

    histname="hDPhiTracksAtVtxVsR";
    h2_DPhiTracksAtVtxVsR_ = dbe_->book2D(histname+"All","  Photons:Tracks from conversions: #delta#phi Tracks at vertex vs R",rBin,rMin, rMax,100, -0.5, 0.5);
    histname="pDPhiTracksAtVtxVsR";
    p_DPhiTracksAtVtxVsR_ = dbe_->bookProfile(histname+"All"," Photons:Tracks from conversions: #delta#phi Tracks at vertex vs R ",rBin,rMin, rMax,100, -0.5, 0.5,"");


    histname="hDCotTracks";
    h_DCotTracks_[0][0]= dbe_->book1D(histname+"All"," Photons:Tracks from conversions #delta cotg(#Theta) Tracks: all Ecal ",dCotTracksBin,dCotTracksMin,dCotTracksMax);
    h_DCotTracks_[0][1]= dbe_->book1D(histname+"Barrel"," Photons:Tracks from conversions #delta cotg(#Theta) Tracks: Barrel Ecal ",dCotTracksBin,dCotTracksMin,dCotTracksMax);
    h_DCotTracks_[0][2]= dbe_->book1D(histname+"Endcap"," Photons:Tracks from conversions #delta cotg(#Theta) Tracks: Endcap Ecal ",dCotTracksBin,dCotTracksMin,dCotTracksMax);
    h_DCotTracks_[1][0]= dbe_->book1D(histname+"All_Ass"," Photons:Tracks from conversions #delta cotg(#Theta) Tracks: all Ecal ",dCotTracksBin,dCotTracksMin,dCotTracksMax);
    h_DCotTracks_[1][1]= dbe_->book1D(histname+"Barrel_Ass"," Photons:Tracks from conversions #delta cotg(#Theta) Tracks: Barrel Ecal ",dCotTracksBin,dCotTracksMin,dCotTracksMax);
    h_DCotTracks_[1][2]= dbe_->book1D(histname+"Endcap_Ass"," Photons:Tracks from conversions #delta cotg(#Theta) Tracks: Endcap Ecal ",dCotTracksBin,dCotTracksMin,dCotTracksMax);
    h_DCotTracks_[2][0]= dbe_->book1D(histname+"All_Fakes"," Photons:Tracks from conversions #delta cotg(#Theta) Tracks: all Ecal ",dCotTracksBin,dCotTracksMin,dCotTracksMax);
    h_DCotTracks_[2][1]= dbe_->book1D(histname+"Barrel_Fakes"," Photons:Tracks from conversions #delta cotg(#Theta) Tracks: Barrel Ecal ",dCotTracksBin,dCotTracksMin,dCotTracksMax);
    h_DCotTracks_[2][2]= dbe_->book1D(histname+"Endcap_Fakes"," Photons:Tracks from conversions #delta cotg(#Theta) Tracks: Endcap Ecal ",dCotTracksBin,dCotTracksMin,dCotTracksMax);


    histname="hDCotTracksVsEta";
    h2_DCotTracksVsEta_ = dbe_->book2D(histname+"All","  Photons:Tracks from conversions:  #delta cotg(#Theta) Tracks vs #eta",etaBin2,etaMin, etaMax,100, -0.2, 0.2);
    histname="pDCotTracksVsEta";
    p_DCotTracksVsEta_ = dbe_->bookProfile(histname+"All"," Photons:Tracks from conversions:  #delta cotg(#Theta) Tracks vs #eta ",etaBin2,etaMin, etaMax, 100, -0.2, 0.2,"");

    histname="hDCotTracksVsR";
    h2_DCotTracksVsR_ = dbe_->book2D(histname+"All","  Photons:Tracks from conversions:  #delta cotg(#Theta)  Tracks at vertex vs R",rBin,rMin, rMax,100, -0.2, 0.2);
    histname="pDCotTracksVsR";
    p_DCotTracksVsR_ = dbe_->bookProfile(histname+"All"," Photons:Tracks from conversions:  #delta cotg(#Theta) Tracks at vertex vs R ",rBin,rMin, rMax,100, -0.2, 0.2,"");


    histname="hDistMinAppTracks";
    h_distMinAppTracks_[0][0]= dbe_->book1D(histname+"All"," Photons:Tracks from conversions Min Approach Dist Tracks: all Ecal ",120, -0.5, 1.0);
    h_distMinAppTracks_[0][1]= dbe_->book1D(histname+"Barrel"," Photons:Tracks from conversions Min Approach Dist Tracks: Barrel Ecal ",120, -0.5, 1.0);
    h_distMinAppTracks_[0][2]= dbe_->book1D(histname+"Endcap"," Photons:Tracks from conversions Min Approach Dist Tracks: Endcap Ecal ",120, -0.5, 1.0);
    h_distMinAppTracks_[1][0]= dbe_->book1D(histname+"All_Ass"," Photons:Tracks from conversions Min Approach Dist Tracks: all Ecal ",120, -0.5, 1.0);
    h_distMinAppTracks_[1][1]= dbe_->book1D(histname+"Barrel_Ass"," Photons:Tracks from conversions Min Approach Dist Tracks: Barrel Ecal ",120, -0.5, 1.0);
    h_distMinAppTracks_[1][2]= dbe_->book1D(histname+"Endcap_Ass"," Photons:Tracks from conversions Min Approach Dist Tracks: Endcap Ecal ",120, -0.5, 1.0);
    h_distMinAppTracks_[2][0]= dbe_->book1D(histname+"All_Fakes"," Photons:Tracks from conversions Min Approach Dist Tracks: all Ecal ",120, -0.5, 1.0);
    h_distMinAppTracks_[2][1]= dbe_->book1D(histname+"Barrel_Fakes"," Photons:Tracks from conversions Min Approach Dist Tracks: Barrel Ecal ",120, -0.5, 1.0);
    h_distMinAppTracks_[2][2]= dbe_->book1D(histname+"Endcap_Fakes"," Photons:Tracks from conversions Min Approach Dist Tracks: Endcap Ecal ",120, -0.5, 1.0);


    h_convVtxRvsZ_[0] =   dbe_->book2D("convVtxRvsZAll"," Photon Reco conversion vtx position",zBinForXray, zMinForXray, zMaxForXray, rBinForXray, rMinForXray, rMaxForXray);
    h_convVtxRvsZ_[1] =   dbe_->book2D("convVtxRvsZBarrel"," Photon Reco conversion vtx position",zBinForXray, zMinForXray, zMaxForXray, rBinForXray, rMinForXray, rMaxForXray);
    h_convVtxRvsZ_[2] =   dbe_->book2D("convVtxRvsZEndcap"," Photon Reco conversion vtx position",zBin2ForXray, zMinForXray, zMaxForXray, rBinForXray, rMinForXray, rMaxForXray);
    h_convVtxYvsX_ =   dbe_->book2D("convVtxYvsXTrkBarrel"," Photon Reco conversion vtx position, (x,y) eta<1 ", 1000, -60., 60., 1000, -60., 60.);
    /// zooms
    h_convVtxRvsZ_zoom_[0] =  dbe_->book2D("convVtxRvsZBarrelZoom1"," Photon Reco conversion vtx position",zBinForXray, zMinForXray, zMaxForXray, rBinForXray, -10., 40.);
    h_convVtxRvsZ_zoom_[1] =  dbe_->book2D("convVtxRvsZBarrelZoom2"," Photon Reco conversion vtx position",zBinForXray, zMinForXray, zMaxForXray, rBinForXray, -10., 20.);
    h_convVtxYvsX_zoom_[0] =   dbe_->book2D("convVtxYvsXTrkBarrelZoom1"," Photon Reco conversion vtx position, (x,y) eta<1 ",100, -40., 40., 100, -40., 40.);
    h_convVtxYvsX_zoom_[1] =   dbe_->book2D("convVtxYvsXTrkBarrelZoom2"," Photon Reco conversion vtx position, (x,y) eta<1 ",100, -20., 20., 100, -20., 20.);

    h_convVtxdR_ =   dbe_->book1D("convVtxdR"," Photon Reco conversion vtx dR",100, -10.,10.);
    h_convVtxdX_ =   dbe_->book1D("convVtxdX"," Photon Reco conversion vtx dX",100, -10.,10.);
    h_convVtxdY_ =   dbe_->book1D("convVtxdY"," Photon Reco conversion vtx dY",100, -10.,10.);
    h_convVtxdZ_ =   dbe_->book1D("convVtxdZ"," Photon Reco conversion vtx dZ",100, -20.,20.);

    h_convVtxdPhi_ =   dbe_->book1D("convVtxdPhi"," Photon Reco conversion vtx dPhi",100, -0.01,0.01);
    h_convVtxdEta_ =   dbe_->book1D("convVtxdEta"," Photon Reco conversion vtx dEta",100, -0.5,0.5);

    h_convVtxdR_barrel_ =   dbe_->book1D("convVtxdR_barrel"," Photon Reco conversion vtx dR, |eta|<=1.2",100, -10.,10.);
    h_convVtxdX_barrel_ =   dbe_->book1D("convVtxdX_barrel"," Photon Reco conversion vtx dX, |eta|<=1.2",100, -10.,10.);
    h_convVtxdY_barrel_ =   dbe_->book1D("convVtxdY_barrel"," Photon Reco conversion vtx dY, |eta|<=1.2 ",100, -10.,10.);
    h_convVtxdZ_barrel_ =   dbe_->book1D("convVtxdZ_barrel"," Photon Reco conversion vtx dZ, |eta|<=1.2,",100, -20.,20.);

    h_convVtxdR_endcap_ =   dbe_->book1D("convVtxdR_endcap"," Photon Reco conversion vtx dR,  |eta|>1.2 ",100, -10.,10.);
    h_convVtxdX_endcap_ =   dbe_->book1D("convVtxdX_endcap"," Photon Reco conversion vtx dX,  |eta|>1.2",100, -10.,10.);
    h_convVtxdY_endcap_ =   dbe_->book1D("convVtxdY_endcap"," Photon Reco conversion vtx dY,  |eta|>1.2",100, -10.,10.);
    h_convVtxdZ_endcap_ =   dbe_->book1D("convVtxdZ_endcap"," Photon Reco conversion vtx dZ,  |eta|>1.2",100, -20.,20.);



    h2_convVtxdRVsR_ =  dbe_->book2D("h2ConvVtxdRVsR"," Conversion vtx dR vsR" ,rBin,rMin, rMax,100, -20.,20.);
    h2_convVtxdRVsEta_ =  dbe_->book2D("h2ConvVtxdRVsEta","Conversion vtx dR vs Eta" ,etaBin2,etaMin, etaMax,100, -20.,20.);

    p_convVtxdRVsR_ =  dbe_->bookProfile("pConvVtxdRVsR"," Conversion vtx dR vsR" ,rBin,rMin, rMax ,100, -20.,20., "");
    p_convVtxdRVsEta_ =  dbe_->bookProfile("pConvVtxdRVsEta","Conversion vtx dR vs Eta" ,etaBin2,etaMin, etaMax, 100, -20.,20., "");
    p_convVtxdXVsX_ =  dbe_->bookProfile("pConvVtxdXVsX","Conversion vtx dX vs X" ,120,-60, 60 ,100, -20.,20., "");
    p_convVtxdYVsY_ =  dbe_->bookProfile("pConvVtxdYVsY","Conversion vtx dY vs Y" ,120,-60, 60 ,100, -20.,20., "");
    p_convVtxdZVsZ_ =  dbe_->bookProfile("pConvVtxdZVsZ","Conversion vtx dZ vs Z" ,zBin,zMin,zMax ,100, -20.,20., "");

    p_convVtxdZVsR_ =  dbe_->bookProfile("pConvVtxdZVsR","Conversion vtx dZ vs R" ,rBin,rMin,rMax ,100, -20.,20., "");
    p2_convVtxdRVsRZ_ =  dbe_->bookProfile2D("p2ConvVtxdRVsRZ","Conversion vtx dR vs RZ" ,zBin,zMin, zMax,rBin,rMin,rMax,100, 0.,20.,"s");
    p2_convVtxdZVsRZ_ =  dbe_->bookProfile2D("p2ConvVtxdZVsRZ","Conversion vtx dZ vs RZ" ,zBin,zMin, zMax,rBin,rMin,rMax,100, 0.,20.,"s");


    histname="EoverPtracks";
    h_EoverPTracks_[0][0] = dbe_->book1D(histname+"All"," photons conversion E/p: all Ecal ",       eoverpBin, eoverpMin, eoverpMax );
    h_EoverPTracks_[0][1] = dbe_->book1D(histname+"Barrel"," photons conversion E/p: Barrel Ecal",  eoverpBin, eoverpMin,  eoverpMax);
    h_EoverPTracks_[0][2] = dbe_->book1D(histname+"Endcap"," photons conversion E/p: Endcap Ecal ", eoverpBin, eoverpMin,  eoverpMax);
    h_EoverPTracks_[1][0] = dbe_->book1D(histname+"All_Ass"," photons conversion E/p: all Ecal ",   eoverpBin, eoverpMin,  eoverpMax);
    h_EoverPTracks_[1][1] = dbe_->book1D(histname+"Barrel_Ass"," photons conversion E/p: Barrel Ecal", eoverpBin, eoverpMin,  eoverpMax);
    h_EoverPTracks_[1][2] = dbe_->book1D(histname+"Endcap_Ass"," photons conversion E/p: Endcap Ecal ", eoverpBin, eoverpMin,  eoverpMax);
    h_EoverPTracks_[2][0] = dbe_->book1D(histname+"All_Fakes"," photons conversion E/p: all Ecal ",     eoverpBin, eoverpMin,  eoverpMax);
    h_EoverPTracks_[2][1] = dbe_->book1D(histname+"Barrel_Fakes"," photons conversion E/p: Barrel Ecal", eoverpBin, eoverpMin,  eoverpMax);
    h_EoverPTracks_[2][2] = dbe_->book1D(histname+"Endcap_Fakes"," photons conversion E/p: Endcap Ecal ", eoverpBin, eoverpMin,  eoverpMax);


    h2_convVtxRrecVsTrue_ =  dbe_->book2D("h2ConvVtxRrecVsTrue","Photon Reco conversion vtx R rec vs true" ,rBin,rMin, rMax,rBin,rMin, rMax);

    histname="vtxChi2Prob";
    h_vtxChi2Prob_[0][0] = dbe_->book1D(histname+"All","vertex #chi^{2} all", 100, 0., 1.);
    h_vtxChi2Prob_[0][1] = dbe_->book1D(histname+"Barrel","vertex #chi^{2} barrel", 100, 0., 1.);
    h_vtxChi2Prob_[0][2] = dbe_->book1D(histname+"Endcap","vertex #chi^{2} endcap", 100, 0., 1.);
    h_vtxChi2Prob_[1][0] = dbe_->book1D(histname+"All_Ass","vertex #chi^{2} all", 100, 0., 1.);
    h_vtxChi2Prob_[1][1] = dbe_->book1D(histname+"Barrel_Ass","vertex #chi^{2} barrel", 100, 0., 1.);
    h_vtxChi2Prob_[1][2] = dbe_->book1D(histname+"Endcap_Ass","vertex #chi^{2} endcap", 100, 0., 1.);
    h_vtxChi2Prob_[2][0] = dbe_->book1D(histname+"All_Fakes","vertex #chi^{2} all", 100, 0., 1.);
    h_vtxChi2Prob_[2][1] = dbe_->book1D(histname+"Barrel_Fakes","vertex #chi^{2} barrel", 100, 0., 1.);
    h_vtxChi2Prob_[2][2] = dbe_->book1D(histname+"Endcap_Fakes","vertex #chi^{2} endcap", 100, 0., 1.);


    h_zPVFromTracks_[1] =  dbe_->book1D("zPVFromTracks"," Photons: PV z from conversion tracks",100, -25., 25.);
    h_dzPVFromTracks_[1] =  dbe_->book1D("dzPVFromTracks"," Photons: PV Z_rec - Z_true from conversion tracks",100, -5., 5.);
    h2_dzPVVsR_ =  dbe_->book2D("h2dzPVVsR","Photon Reco conversions: dz(PV) vs R" ,rBin,rMin, rMax,100, -3.,3.);
    p_dzPVVsR_ =  dbe_->bookProfile("pdzPVVsR","Photon Reco conversions: dz(PV) vs R" ,rBin,rMin, rMax, 100, -3.,3.,"");


    histname="lxybs";
    h_lxybs_[0][0] = dbe_->book1D(histname+"All","vertex #chi^{2} all", 200, -100., 100.);
    h_lxybs_[0][1] = dbe_->book1D(histname+"Barrel","vertex #chi^{2} barrel", 200, -100., 100.);
    h_lxybs_[0][2] = dbe_->book1D(histname+"Endcap","vertex #chi^{2} endcap", 200, -100., 100.);
    h_lxybs_[1][0] = dbe_->book1D(histname+"All_Ass","vertex #chi^{2} all", 200, -100., 100.);
    h_lxybs_[1][1] = dbe_->book1D(histname+"Barrel_Ass","vertex #chi^{2} barrel", 200, -100., 100.);
    h_lxybs_[1][2] = dbe_->book1D(histname+"Endcap_Ass","vertex #chi^{2} endcap", 200, -100., 100.);
    h_lxybs_[2][0] = dbe_->book1D(histname+"All_Fakes","vertex #chi^{2} all", 200, -100., 100.);
    h_lxybs_[2][1] = dbe_->book1D(histname+"Barrel_Fakes","vertex #chi^{2} barrel", 200, -100., 100.);
    h_lxybs_[2][2] = dbe_->book1D(histname+"Endcap_Fakes","vertex #chi^{2} endcap", 200, -100., 100.);

    histname="maxNHitsBeforeVtx";
    h_maxNHitsBeforeVtx_[0][0] = dbe_->book1D(histname+"All","vertex #chi^{2} all", 16, -0.5, 15.5);
    h_maxNHitsBeforeVtx_[0][1] = dbe_->book1D(histname+"Barrel","vertex #chi^{2} barrel", 16, -0.5, 15.5);
    h_maxNHitsBeforeVtx_[0][2] = dbe_->book1D(histname+"Endcap","vertex #chi^{2} endcap", 16, -0.5, 15.5);
    h_maxNHitsBeforeVtx_[1][0] = dbe_->book1D(histname+"All_Ass","vertex #chi^{2} all", 16, -0.5, 15.5);
    h_maxNHitsBeforeVtx_[1][1] = dbe_->book1D(histname+"Barrel_Ass","vertex #chi^{2} barrel", 16, -0.5, 15.5);
    h_maxNHitsBeforeVtx_[1][2] = dbe_->book1D(histname+"Endcap_Ass","vertex #chi^{2} endcap", 16, -0.5, 15.5);
    h_maxNHitsBeforeVtx_[2][0] = dbe_->book1D(histname+"All_Fakes","vertex #chi^{2} all", 16, -0.5, 15.5);
    h_maxNHitsBeforeVtx_[2][1] = dbe_->book1D(histname+"Barrel_Fakes","vertex #chi^{2} barrel", 16, -0.5, 15.5);
    h_maxNHitsBeforeVtx_[2][2] = dbe_->book1D(histname+"Endcap_Fakes","vertex #chi^{2} endcap", 16, -0.5, 15.5);

    histname="leadNHitsBeforeVtx";
    h_leadNHitsBeforeVtx_[0][0] = dbe_->book1D(histname+"All","vertex #chi^{2} all", 16, -0.5, 15.5);
    h_leadNHitsBeforeVtx_[0][1] = dbe_->book1D(histname+"Barrel","vertex #chi^{2} barrel", 16, -0.5, 15.5);
    h_leadNHitsBeforeVtx_[0][2] = dbe_->book1D(histname+"Endcap","vertex #chi^{2} endcap", 16, -0.5, 15.5);
    h_leadNHitsBeforeVtx_[1][0] = dbe_->book1D(histname+"All_Ass","vertex #chi^{2} all", 16, -0.5, 15.5);
    h_leadNHitsBeforeVtx_[1][1] = dbe_->book1D(histname+"Barrel_Ass","vertex #chi^{2} barrel", 16, -0.5, 15.5);
    h_leadNHitsBeforeVtx_[1][2] = dbe_->book1D(histname+"Endcap_Ass","vertex #chi^{2} endcap", 16, -0.5, 15.5);
    h_leadNHitsBeforeVtx_[2][0] = dbe_->book1D(histname+"All_Fakes","vertex #chi^{2} all", 16, -0.5, 15.5);
    h_leadNHitsBeforeVtx_[2][1] = dbe_->book1D(histname+"Barrel_Fakes","vertex #chi^{2} barrel", 16, -0.5, 15.5);
    h_leadNHitsBeforeVtx_[2][2] = dbe_->book1D(histname+"Endcap_Fakes","vertex #chi^{2} endcap", 16, -0.5, 15.5);

    histname="trailNHitsBeforeVtx";
    h_trailNHitsBeforeVtx_[0][0] = dbe_->book1D(histname+"All","vertex #chi^{2} all", 16, -0.5, 15.5);
    h_trailNHitsBeforeVtx_[0][1] = dbe_->book1D(histname+"Barrel","vertex #chi^{2} barrel", 16, -0.5, 15.5);
    h_trailNHitsBeforeVtx_[0][2] = dbe_->book1D(histname+"Endcap","vertex #chi^{2} endcap", 16, -0.5, 15.5);
    h_trailNHitsBeforeVtx_[1][0] = dbe_->book1D(histname+"All_Ass","vertex #chi^{2} all", 16, -0.5, 15.5);
    h_trailNHitsBeforeVtx_[1][1] = dbe_->book1D(histname+"Barrel_Ass","vertex #chi^{2} barrel", 16, -0.5, 15.5);
    h_trailNHitsBeforeVtx_[1][2] = dbe_->book1D(histname+"Endcap_Ass","vertex #chi^{2} endcap", 16, -0.5, 15.5);
    h_trailNHitsBeforeVtx_[2][0] = dbe_->book1D(histname+"All_Fakes","vertex #chi^{2} all", 16, -0.5, 15.5);
    h_trailNHitsBeforeVtx_[2][1] = dbe_->book1D(histname+"Barrel_Fakes","vertex #chi^{2} barrel", 16, -0.5, 15.5);
    h_trailNHitsBeforeVtx_[2][2] = dbe_->book1D(histname+"Endcap_Fakes","vertex #chi^{2} endcap", 16, -0.5, 15.5);

    histname="sumNHitsBeforeVtx";
    h_sumNHitsBeforeVtx_[0][0] = dbe_->book1D(histname+"All","vertex #chi^{2} all", 16, -0.5, 15.5);
    h_sumNHitsBeforeVtx_[0][1] = dbe_->book1D(histname+"Barrel","vertex #chi^{2} barrel", 16, -0.5, 15.5);
    h_sumNHitsBeforeVtx_[0][2] = dbe_->book1D(histname+"Endcap","vertex #chi^{2} endcap", 16, -0.5, 15.5);
    h_sumNHitsBeforeVtx_[1][0] = dbe_->book1D(histname+"All_Ass","vertex #chi^{2} all", 16, -0.5, 15.5);
    h_sumNHitsBeforeVtx_[1][1] = dbe_->book1D(histname+"Barrel_Ass","vertex #chi^{2} barrel", 16, -0.5, 15.5);
    h_sumNHitsBeforeVtx_[1][2] = dbe_->book1D(histname+"Endcap_Ass","vertex #chi^{2} endcap", 16, -0.5, 15.5);
    h_sumNHitsBeforeVtx_[2][0] = dbe_->book1D(histname+"All_Fakes","vertex #chi^{2} all", 16, -0.5, 15.5);
    h_sumNHitsBeforeVtx_[2][1] = dbe_->book1D(histname+"Barrel_Fakes","vertex #chi^{2} barrel", 16, -0.5, 15.5);
    h_sumNHitsBeforeVtx_[2][2] = dbe_->book1D(histname+"Endcap_Fakes","vertex #chi^{2} endcap", 16, -0.5, 15.5);

    histname="maxDlClosestHitToVtx";
    h_maxDlClosestHitToVtx_[0][0] = dbe_->book1D(histname+"All","vertex #chi^{2} all", 100, -10., 10.);
    h_maxDlClosestHitToVtx_[0][1] = dbe_->book1D(histname+"Barrel","vertex #chi^{2} barrel", 100, -10., 10.);
    h_maxDlClosestHitToVtx_[0][2] = dbe_->book1D(histname+"Endcap","vertex #chi^{2} endcap", 100, -10., 10.);
    h_maxDlClosestHitToVtx_[1][0] = dbe_->book1D(histname+"All_Ass","vertex #chi^{2} all", 100, -10., 10.);
    h_maxDlClosestHitToVtx_[1][1] = dbe_->book1D(histname+"Barrel_Ass","vertex #chi^{2} barrel", 100, -10., 10.);
    h_maxDlClosestHitToVtx_[1][2] = dbe_->book1D(histname+"Endcap_Ass","vertex #chi^{2} endcap", 100, -10., 10.);
    h_maxDlClosestHitToVtx_[2][0] = dbe_->book1D(histname+"All_Fakes","vertex #chi^{2} all", 100, -10., 10.);
    h_maxDlClosestHitToVtx_[2][1] = dbe_->book1D(histname+"Barrel_Fakes","vertex #chi^{2} barrel", 100, -10., 10.);
    h_maxDlClosestHitToVtx_[2][2] = dbe_->book1D(histname+"Endcap_Fakes","vertex #chi^{2} endcap", 100, -10., 10.);

    histname="maxDlClosestHitToVtxSig";
    h_maxDlClosestHitToVtxSig_[0][0] = dbe_->book1D(histname+"All","vertex #chi^{2} all", 100, -8., 8.);
    h_maxDlClosestHitToVtxSig_[0][1] = dbe_->book1D(histname+"Barrel","vertex #chi^{2} barrel", 100, -8., 8.);
    h_maxDlClosestHitToVtxSig_[0][2] = dbe_->book1D(histname+"Endcap","vertex #chi^{2} endcap", 100, -8., 8.);
    h_maxDlClosestHitToVtxSig_[1][0] = dbe_->book1D(histname+"All_Ass","vertex #chi^{2} all", 100, -8., 8.);
    h_maxDlClosestHitToVtxSig_[1][1] = dbe_->book1D(histname+"Barrel_Ass","vertex #chi^{2} barrel", 100, -8., 8.);
    h_maxDlClosestHitToVtxSig_[1][2] = dbe_->book1D(histname+"Endcap_Ass","vertex #chi^{2} endcap", 100, -8., 8.);
    h_maxDlClosestHitToVtxSig_[2][0] = dbe_->book1D(histname+"All_Fakes","vertex #chi^{2} all", 100, -8., 8.);
    h_maxDlClosestHitToVtxSig_[2][1] = dbe_->book1D(histname+"Barrel_Fakes","vertex #chi^{2} barrel", 100, -8., 8.);
    h_maxDlClosestHitToVtxSig_[2][2] = dbe_->book1D(histname+"Endcap_Fakes","vertex #chi^{2} endcap", 100, -8., 8.);

    histname="deltaExpectedHitsInner";
    h_deltaExpectedHitsInner_[0][0] = dbe_->book1D(histname+"All","vertex #chi^{2} all", 31, -15.5, 15.5);
    h_deltaExpectedHitsInner_[0][1] = dbe_->book1D(histname+"Barrel","vertex #chi^{2} barrel", 31, -15.5, 15.5);
    h_deltaExpectedHitsInner_[0][2] = dbe_->book1D(histname+"Endcap","vertex #chi^{2} endcap", 31, -15.5, 15.5);
    h_deltaExpectedHitsInner_[1][0] = dbe_->book1D(histname+"All_Ass","vertex #chi^{2} all", 31, -15.5, 15.5);
    h_deltaExpectedHitsInner_[1][1] = dbe_->book1D(histname+"Barrel_Ass","vertex #chi^{2} barrel", 31, -15.5, 15.5);
    h_deltaExpectedHitsInner_[1][2] = dbe_->book1D(histname+"Endcap_Ass","vertex #chi^{2} endcap", 31, -15.5, 15.5);
    h_deltaExpectedHitsInner_[2][0] = dbe_->book1D(histname+"All_Fakes","vertex #chi^{2} all", 31, -15.5, 15.5);
    h_deltaExpectedHitsInner_[2][1] = dbe_->book1D(histname+"Barrel_Fakes","vertex #chi^{2} barrel", 31, -15.5, 15.5);
    h_deltaExpectedHitsInner_[2][2] = dbe_->book1D(histname+"Endcap_Fakes","vertex #chi^{2} endcap", 31, -15.5, 15.5);

    histname="leadExpectedHitsInner";
    h_leadExpectedHitsInner_[0][0] = dbe_->book1D(histname+"All","vertex #chi^{2} all", 16, -0.5, 15.5);
    h_leadExpectedHitsInner_[0][1] = dbe_->book1D(histname+"Barrel","vertex #chi^{2} barrel", 16, -0.5, 15.5);
    h_leadExpectedHitsInner_[0][2] = dbe_->book1D(histname+"Endcap","vertex #chi^{2} endcap", 16, -0.5, 15.5);
    h_leadExpectedHitsInner_[1][0] = dbe_->book1D(histname+"All_Ass","vertex #chi^{2} all", 16, -0.5, 15.5);
    h_leadExpectedHitsInner_[1][1] = dbe_->book1D(histname+"Barrel_Ass","vertex #chi^{2} barrel", 16, -0.5, 15.5);
    h_leadExpectedHitsInner_[1][2] = dbe_->book1D(histname+"Endcap_Ass","vertex #chi^{2} endcap", 16, -0.5, 15.5);
    h_leadExpectedHitsInner_[2][0] = dbe_->book1D(histname+"All_Fakes","vertex #chi^{2} all", 16, -0.5, 15.5);
    h_leadExpectedHitsInner_[2][1] = dbe_->book1D(histname+"Barrel_Fakes","vertex #chi^{2} barrel", 16, -0.5, 15.5);
    h_leadExpectedHitsInner_[2][2] = dbe_->book1D(histname+"Endcap_Fakes","vertex #chi^{2} endcap", 16, -0.5, 15.5);

    histname="nSharedHits";
    h_nSharedHits_[0][0] = dbe_->book1D(histname+"All","vertex #chi^{2} all", 16, -0.5, 15.5);
    h_nSharedHits_[0][1] = dbe_->book1D(histname+"Barrel","vertex #chi^{2} barrel", 16, -0.5, 15.5);
    h_nSharedHits_[0][2] = dbe_->book1D(histname+"Endcap","vertex #chi^{2} endcap", 16, -0.5, 15.5);
    h_nSharedHits_[1][0] = dbe_->book1D(histname+"All_Ass","vertex #chi^{2} all", 16, -0.5, 15.5);
    h_nSharedHits_[1][1] = dbe_->book1D(histname+"Barrel_Ass","vertex #chi^{2} barrel", 16, -0.5, 15.5);
    h_nSharedHits_[1][2] = dbe_->book1D(histname+"Endcap_Ass","vertex #chi^{2} endcap", 16, -0.5, 15.5);
    h_nSharedHits_[2][0] = dbe_->book1D(histname+"All_Fakes","vertex #chi^{2} all", 16, -0.5, 15.5);
    h_nSharedHits_[2][1] = dbe_->book1D(histname+"Barrel_Fakes","vertex #chi^{2} barrel", 16, -0.5, 15.5);
    h_nSharedHits_[2][2] = dbe_->book1D(histname+"Endcap_Fakes","vertex #chi^{2} endcap", 16, -0.5, 15.5);

    //////////////////// plots per track
    histname="nHits";
    nHits_[0] =  dbe_->book2D(histname+"AllTracks","Photons:Tracks from conversions: # of hits all tracks",etaBin,etaMin, etaMax,30,0., 30.);
    nHits_[1] =  dbe_->book2D(histname+"AllTracks_Ass","Photons:Tracks from conversions: # of hits  all tracks ass",etaBin,etaMin, etaMax,30,0., 30.);
    nHits_[2] =  dbe_->book2D(histname+"AllTracks_Fakes","Photons:Tracks from conversions: # of hits all tracks fakes",etaBin,etaMin, etaMax,30,0., 30.);


    histname="nHitsVsEta";
    nHitsVsEta_[0] =  dbe_->book2D(histname+"AllTracks","Photons:Tracks from conversions: # of hits vs #eta all tracks",etaBin,etaMin, etaMax,30,0., 30.);
    nHitsVsEta_[1] =  dbe_->book2D(histname+"AllTracks_Ass","Photons:Tracks from conversions: # of hits vs #eta all tracks",etaBin,etaMin, etaMax,30,0., 30.);
    nHitsVsEta_[2] =  dbe_->book2D(histname+"AllTracks_Fakes","Photons:Tracks from conversions: # of hits vs #eta all tracks",etaBin,etaMin, etaMax,30,0., 30.);
    histname="h_nHitsVsEta";
    p_nHitsVsEta_[0] =  dbe_->bookProfile(histname+"AllTracks","Photons:Tracks from conversions: # of hits vs #eta all tracks",etaBin,etaMin, etaMax, 30,-0.5, 29.5,"");
    p_nHitsVsEta_[1] =  dbe_->bookProfile(histname+"AllTracks_Ass","Photons:Tracks from conversions: # of hits vs #eta all tracks",etaBin,etaMin, etaMax, 30,-0.5, 29.5,"");
    p_nHitsVsEta_[2] =  dbe_->bookProfile(histname+"AllTracks_Fakes","Photons:Tracks from conversions: # of hits vs #eta all tracks",etaBin,etaMin, etaMax, 30,-0.5, 29.5,"");


    histname="nHitsVsR";
    nHitsVsR_[0] =  dbe_->book2D(histname+"AllTracks","Photons:Tracks from conversions: # of hits vs radius all tracks" ,rBin,rMin, rMax,30,0., 30.);
    nHitsVsR_[1] =  dbe_->book2D(histname+"AllTracks_Ass","Photons:Tracks from conversions: # of hits vs radius all tracks" ,rBin,rMin, rMax,30,0., 30.);
    nHitsVsR_[2] =  dbe_->book2D(histname+"AllTracks_Fakes","Photons:Tracks from conversions: # of hits vs radius all tracks" ,rBin,rMin, rMax,30,0., 30.);

    histname="h_nHitsVsR";
    p_nHitsVsR_[0] =  dbe_->bookProfile(histname+"AllTracks","Photons:Tracks from conversions: # of hits vs radius all tracks",rBin,rMin, rMax, 30,-0.5, 29.5,"");
    p_nHitsVsR_[1] =  dbe_->bookProfile(histname+"AllTracks_Ass","Photons:Tracks from conversions: # of hits vs radius all tracks",rBin,rMin, rMax, 30,-0.5, 29.5,"");
    p_nHitsVsR_[2] =  dbe_->bookProfile(histname+"AllTracks_Fakes","Photons:Tracks from conversions: # of hits vs radius all tracks",rBin,rMin, rMax, 30,-0.5, 29.5,"");

    histname="tkChi2";
    h_tkChi2_[0] = dbe_->book1D(histname+"AllTracks","Photons:Tracks from conversions: #chi^{2} of all tracks", 100, chi2Min, chi2Max);
    h_tkChi2_[1] = dbe_->book1D(histname+"AllTracks_Ass","Photons:Tracks from conversions: #chi^{2} of all tracks", 100, chi2Min, chi2Max);
    h_tkChi2_[2] = dbe_->book1D(histname+"AllTracks_Fakes","Photons:Tracks from conversions: #chi^{2} of all tracks", 100, chi2Min, chi2Max);

    histname="tkChi2Large";
    h_tkChi2Large_[0] = dbe_->book1D(histname+"AllTracks","Photons:Tracks from conversions: #chi^{2} of all tracks", 1000, 0., 5000.0);
    h_tkChi2Large_[1] = dbe_->book1D(histname+"AllTracks_Ass","Photons:Tracks from conversions: #chi^{2} of all tracks", 1000, 0., 5000.0);
    h_tkChi2Large_[2] = dbe_->book1D(histname+"AllTracks_Fakes","Photons:Tracks from conversions: #chi^{2} of all tracks", 1000, 0., 5000.0);


    histname="h2Chi2VsEta";
    h2_Chi2VsEta_[0]=dbe_->book2D(histname+"All"," Reco Track  #chi^{2} vs #eta: All ",etaBin2,etaMin, etaMax,100, chi2Min, chi2Max);
    h2_Chi2VsEta_[1]=dbe_->book2D(histname+"All_Ass"," Reco Track  #chi^{2} vs #eta: All ",etaBin2,etaMin, etaMax,100, chi2Min, chi2Max);
    h2_Chi2VsEta_[2]=dbe_->book2D(histname+"All_Fakes"," Reco Track  #chi^{2} vs #eta: All ",etaBin2,etaMin, etaMax,100, chi2Min, chi2Max);
    histname="pChi2VsEta";
    p_Chi2VsEta_[0]=dbe_->bookProfile(histname+"All"," Reco Track #chi^{2} vs #eta : All ",etaBin2,etaMin, etaMax, 100, chi2Min, chi2Max,"");
    p_Chi2VsEta_[1]=dbe_->bookProfile(histname+"All_Ass"," Reco Track #chi^{2} vs #eta : All ",etaBin2,etaMin, etaMax, 100, chi2Min, chi2Max,"");
    p_Chi2VsEta_[2]=dbe_->bookProfile(histname+"All_Fakes"," Reco Track #chi^{2} vs #eta : All ",etaBin2,etaMin, etaMax, 100, chi2Min, chi2Max,"");

    histname="h2Chi2VsR";
    h2_Chi2VsR_[0]=dbe_->book2D(histname+"All"," Reco Track  #chi^{2} vs R: All ",rBin,rMin, rMax,100,chi2Min, chi2Max);
    h2_Chi2VsR_[1]=dbe_->book2D(histname+"All_Ass"," Reco Track  #chi^{2} vs R: All ",rBin,rMin, rMax,100,chi2Min, chi2Max);
    h2_Chi2VsR_[2]=dbe_->book2D(histname+"All_Fakes"," Reco Track  #chi^{2} vs R: All ",rBin,rMin, rMax,100,chi2Min, chi2Max);
    histname="pChi2VsR";
    p_Chi2VsR_[0]=dbe_->bookProfile(histname+"All"," Reco Track #chi^{2} vas R : All ",rBin,rMin,rMax, 100,chi2Min, chi2Max,"");
    p_Chi2VsR_[1]=dbe_->bookProfile(histname+"All_Ass"," Reco Track #chi^{2} vas R : All ",rBin,rMin,rMax, 100,chi2Min, chi2Max,"");
    p_Chi2VsR_[2]=dbe_->bookProfile(histname+"All_Fakes"," Reco Track #chi^{2} vas R : All ",rBin,rMin,rMax, 100,chi2Min, chi2Max,"");

    histname="hTkD0";
    h_TkD0_[0]=dbe_->book1D(histname+"All"," Reco Track D0*q: All ",200,-0.1,60);
    h_TkD0_[1]=dbe_->book1D(histname+"All_Ass"," Reco Track D0*q: Barrel ",200,-0.1,60);
    h_TkD0_[2]=dbe_->book1D(histname+"All_Fakes"," Reco Track D0*q: Endcap ",200,-0.1,60);



    histname="hTkPtPull";
    h_TkPtPull_[0]=dbe_->book1D(histname+"All"," Reco Track Pt pull: All ",100, -20., 10.);
    histname="hTkPtPull";
    h_TkPtPull_[1]=dbe_->book1D(histname+"Barrel"," Reco Track Pt pull: Barrel ",100, -20., 10.);
    histname="hTkPtPull";
    h_TkPtPull_[2]=dbe_->book1D(histname+"Endcap"," Reco Track Pt pull: Endcap ",100, -20., 10.);

    histname="h2TkPtPullEta";
    h2_TkPtPull_[0]=dbe_->book2D(histname+"All"," Reco Track Pt pull: All ",etaBin2,etaMin, etaMax,100, -20., 10.);
    histname="pTkPtPullEta";
    p_TkPtPull_[0]=dbe_->bookProfile(histname+"All"," Reco Track Pt pull: All ",etaBin2,etaMin, etaMax, 100, -20., 10., " ");


    histname="PtRecVsPtSim";
    h2_PtRecVsPtSim_[0]=dbe_->book2D(histname+"All", "Pt Rec vs Pt sim: All ", etBin,etMin,etMax,etBin,etMin, etMax);
    h2_PtRecVsPtSim_[1]=dbe_->book2D(histname+"Barrel", "Pt Rec vs Pt sim: Barrel ", etBin,etMin,etMax,etBin,etMin, etMax);
    h2_PtRecVsPtSim_[2]=dbe_->book2D(histname+"Endcap", "Pt Rec vs Pt sim: Endcap ", etBin,etMin,etMax,etBin,etMin, etMax);

    histname="photonPtRecVsPtSim";
    h2_photonPtRecVsPtSim_=dbe_->book2D(histname+"All", "Pt Rec vs Pt sim: All ", etBin,etMin,etMax,etBin,etMin, etMax);

    histname="nHitsBeforeVtx";
    h_nHitsBeforeVtx_[0]=dbe_->book1D(histname+"All", "Pt Rec vs Pt sim: All ", 16, -0.5, 15.5);
    h_nHitsBeforeVtx_[1]=dbe_->book1D(histname+"Barrel", "Pt Rec vs Pt sim: Barrel ", 16, -0.5, 15.5);
    h_nHitsBeforeVtx_[2]=dbe_->book1D(histname+"Endcap", "Pt Rec vs Pt sim: Endcap ", 16, -0.5, 15.5);

    histname="dlClosestHitToVtx";
    h_dlClosestHitToVtx_[0]=dbe_->book1D(histname+"All", "Pt Rec vs Pt sim: All ", 100, -10., 10.);
    h_dlClosestHitToVtx_[1]=dbe_->book1D(histname+"Barrel", "Pt Rec vs Pt sim: Barrel ", 100, -10., 10.);
    h_dlClosestHitToVtx_[2]=dbe_->book1D(histname+"Endcap", "Pt Rec vs Pt sim: Endcap ", 100, -10., 10.);

    histname="dlClosestHitToVtxSig";
    h_dlClosestHitToVtxSig_[0]=dbe_->book1D(histname+"All", "Pt Rec vs Pt sim: All ", 100, -8., 8.);
    h_dlClosestHitToVtxSig_[1]=dbe_->book1D(histname+"Barrel", "Pt Rec vs Pt sim: Barrel ", 100, -8., 8.);
    h_dlClosestHitToVtxSig_[2]=dbe_->book1D(histname+"Endcap", "Pt Rec vs Pt sim: Endcap ", 100, -8., 8.);

    h_match_= dbe_->book1D("h_match"," ", 3, -0.5,2.5);


  } // if DQM



}



 void  TkConvValidator::beginRun (edm::Run const & r, edm::EventSetup const & theEventSetup) {

   //get magnetic field
  edm::LogInfo("ConvertedPhotonProducer") << " get magnetic field" << "\n";
  theEventSetup.get<IdealMagneticFieldRecord>().get(theMF_);


  edm::ESHandle<TrackAssociatorBase> theHitsAssociator;
  theEventSetup.get<TrackAssociatorRecord>().get("trackAssociatorByHitsForConversionValidation",theHitsAssociator);
  theTrackAssociator_ = (TrackAssociatorBase *) theHitsAssociator.product();




  thePhotonMCTruthFinder_ = new PhotonMCTruthFinder();

}

void  TkConvValidator::endRun (edm::Run& r, edm::EventSetup const & theEventSetup) {

  delete thePhotonMCTruthFinder_;

}



void TkConvValidator::analyze( const edm::Event& e, const edm::EventSetup& esup ) {

  using namespace edm;
  //  const float etaPhiDistance=0.01;
  // Fiducial region
  // const float TRK_BARL =0.9;
  const float BARL = 1.4442; // DAQ TDR p.290
  //  const float END_LO = 1.566; // unused
  const float END_HI = 2.5;
  // Electron mass
  //  const Float_t mElec= 0.000511; // unused


  nEvt_++;
  LogInfo("TkConvValidator") << "TkConvValidator Analyzing event number: " << e.id() << " Global Counter " << nEvt_ <<"\n";
  //  std::cout << "TkConvValidator Analyzing event number: "  << e.id() << " Global Counter " << nEvt_ <<"\n";


  // get the geometry from the event setup:
  esup.get<CaloGeometryRecord>().get(theCaloGeom_);


  // Transform Track into TransientTrack (needed by the Vertex fitter)
  edm::ESHandle<TransientTrackBuilder> theTTB;
  esup.get<TransientTrackRecord>().get("TransientTrackBuilder",theTTB);


  ///// Get the recontructed  conversions
  Handle<reco::ConversionCollection> convHandle;
  e.getByLabel(conversionCollectionProducer_, conversionCollection_ , convHandle);
  const reco::ConversionCollection convCollection = *(convHandle.product());
  if (!convHandle.isValid()) {
    edm::LogError("ConversionsProducer") << "Error! Can't get the  collection "<< std::endl;
    return;
  }

  ///// Get the recontructed  photons
  Handle<reco::PhotonCollection> photonHandle;
  e.getByLabel(photonCollectionProducer_, photonCollection_ , photonHandle);
  const reco::PhotonCollection photonCollection = *(photonHandle.product());
  if (!photonHandle.isValid()) {
    edm::LogError("PhotonProducer") << "Error! Can't get the Photon collection "<< std::endl;
    return;
  }


  // offline  Primary vertex
  edm::Handle<reco::VertexCollection> vertexHandle;
  reco::VertexCollection vertexCollection;
  e.getByLabel("offlinePrimaryVertices", vertexHandle);
  if (!vertexHandle.isValid()) {
      edm::LogError("TrackerOnlyConversionProducer") << "Error! Can't get the product primary Vertex Collection "<< "\n";
  } else {
      vertexCollection = *(vertexHandle.product());
  }
  reco::Vertex the_pvtx;
  bool valid_pvtx = false;
  if (!vertexCollection.empty()){
      the_pvtx = *(vertexCollection.begin());
      //asking for one good vertex
      if (the_pvtx.isValid() && fabs(the_pvtx.position().z())<=15 && the_pvtx.position().Rho()<=2){
          valid_pvtx = true;
      }
  }

  edm::Handle<reco::BeamSpot> bsHandle;
  e.getByLabel("offlineBeamSpot", bsHandle);
  if (!bsHandle.isValid()) {
      edm::LogError("TrackerOnlyConversionProducer") << "Error! Can't get the product primary Vertex Collection "<< "\n";
      return;
  }
  const reco::BeamSpot &thebs = *bsHandle.product();

 //get tracker geometry for hits positions
  edm::ESHandle<TrackerGeometry> tracker;
  esup.get<TrackerDigiGeometryRecord>().get(tracker);
  const TrackerGeometry* trackerGeom = tracker.product();



  //////////////////// Get the MC truth
  //get simtrack info
  std::vector<SimTrack> theSimTracks;
  std::vector<SimVertex> theSimVertices;

  edm::Handle<SimTrackContainer> SimTk;
  edm::Handle<SimVertexContainer> SimVtx;
  e.getByLabel("g4SimHits",SimTk);
  e.getByLabel("g4SimHits",SimVtx);

  bool useTP= parameters_.getParameter<bool>("useTP");
  TrackingParticleCollection tpForEfficiency;
  TrackingParticleCollection tpForFakeRate;
  edm::Handle<TrackingParticleCollection> TPHandleForEff;
  edm::Handle<TrackingParticleCollection> TPHandleForFakeRate;
  if ( useTP) {
    e.getByLabel("tpSelecForEfficiency",TPHandleForEff);
    tpForEfficiency = *(TPHandleForEff.product());
    e.getByLabel("tpSelecForFakeRate",TPHandleForFakeRate);
    tpForFakeRate = *(TPHandleForFakeRate.product());
  }



  theSimTracks.insert(theSimTracks.end(),SimTk->begin(),SimTk->end());
  theSimVertices.insert(theSimVertices.end(),SimVtx->begin(),SimVtx->end());
  std::vector<PhotonMCTruth> mcPhotons=thePhotonMCTruthFinder_->find (theSimTracks,  theSimVertices);

  edm::Handle<edm::HepMCProduct> hepMC;
  e.getByLabel("generator",hepMC);
  //  const HepMC::GenEvent *myGenEvent = hepMC->GetEvent(); // unused


  // get generated jets
  edm::Handle<reco::GenJetCollection> GenJetsHandle ;
  e.getByLabel("iterativeCone5GenJets","",GenJetsHandle);
  reco::GenJetCollection genJetCollection = *(GenJetsHandle.product());

  ConversionHitChecker hitChecker;

  // ################  SIM to RECO ######################### //
  std::map<const reco::Track*,TrackingParticleRef> myAss;
  std::map<const reco::Track*,TrackingParticleRef>::const_iterator itAss;

  for ( std::vector<PhotonMCTruth>::const_iterator mcPho=mcPhotons.begin(); mcPho !=mcPhotons.end(); mcPho++) {

    mcConvPt_= (*mcPho).fourMomentum().et();
    float mcPhi= (*mcPho).fourMomentum().phi();
    mcPhi_= phiNormalization(mcPhi);
    mcEta_= (*mcPho).fourMomentum().pseudoRapidity();
    mcEta_ = etaTransformation(mcEta_, (*mcPho).primaryVertex().z() );
    mcConvR_= (*mcPho).vertex().perp();
    mcConvX_= (*mcPho).vertex().x();
    mcConvY_= (*mcPho).vertex().y();
    mcConvZ_= (*mcPho).vertex().z();
    mcConvEta_= (*mcPho).vertex().eta();
    mcConvPhi_= (*mcPho).vertex().phi();

    if ( fabs(mcEta_) > END_HI ) continue;

    if (mcConvPt_<minPhoPtForEffic) continue;
    if (fabs(mcEta_)>maxPhoEtaForEffic) continue;
    if (fabs(mcConvZ_)>maxPhoZForEffic) continue;
    if (mcConvR_>maxPhoRForEffic) continue;
    ////////////////////////////////// extract info about simulated conversions

    bool goodSimConversion=false;
    bool visibleConversion=false;
    bool visibleConversionsWithTwoSimTracks=false;
    if (  (*mcPho).isAConversion() == 1 ) {
      nSimConv_[0]++;
      h_AllSimConv_[0]->Fill( mcEta_ ) ;
      h_AllSimConv_[1]->Fill( mcPhi_ );
      h_AllSimConv_[2]->Fill( mcConvR_ );
      h_AllSimConv_[3]->Fill( mcConvZ_ );
      h_AllSimConv_[4]->Fill(  (*mcPho).fourMomentum().et());

      if ( mcConvR_ <15) h_SimConvEtaPix_[0]->Fill( mcEta_ ) ;

      if ( ( fabs(mcEta_) <= BARL && mcConvR_ <85 )  ||
	   ( fabs(mcEta_) > BARL && fabs(mcEta_) <=END_HI && fabs( (*mcPho).vertex().z() ) < 210 )  ) visibleConversion=true;

      theConvTP_.clear();
      //      std::cout << " TkConvValidator TrackingParticles   TrackingParticleCollection size "<<  trackingParticles.size() <<  "\n";
      //duplicated TP collections for two associations
      for(size_t i = 0; i < tpForEfficiency.size(); ++i){
	TrackingParticleRef tp (TPHandleForEff,i);
	if ( fabs( tp->vx() - (*mcPho).vertex().x() ) < 0.0001   &&
	     fabs( tp->vy() - (*mcPho).vertex().y() ) < 0.0001   &&
	     fabs( tp->vz() - (*mcPho).vertex().z() ) < 0.0001) {
	  theConvTP_.push_back( tp );
	}
      }
      //std::cout << " TkConvValidator  theConvTP_ size " <<   theConvTP_.size() << std::endl;

      if ( theConvTP_.size() == 2 )   visibleConversionsWithTwoSimTracks=true;
      goodSimConversion=false;

      if (   visibleConversion && visibleConversionsWithTwoSimTracks )  goodSimConversion=true;
      if ( goodSimConversion ) {
	nSimConv_[1]++;
	h_VisSimConv_[0]->Fill( mcEta_ ) ;
	h_VisSimConv_[1]->Fill( mcPhi_ );
	h_VisSimConv_[2]->Fill( mcConvR_ );
	h_VisSimConv_[3]->Fill( mcConvZ_ );
	h_VisSimConv_[4]->Fill(  (*mcPho).fourMomentum().et());

      }

      for ( edm::RefVector<TrackingParticleCollection>::iterator iTrk=theConvTP_.begin(); iTrk!=theConvTP_.end(); ++iTrk) {
	h_simTkPt_ -> Fill ( (*iTrk)->pt() );
	h_simTkEta_ -> Fill ( (*iTrk)->eta() );
      }


    }  ////////////// End of info from sim conversions //////////////////////////////////////////////////

    if ( ! (visibleConversion &&  visibleConversionsWithTwoSimTracks ) ) continue;

      h_simConvVtxRvsZ_[0] ->Fill ( fabs (mcConvZ_), mcConvR_  ) ;
      if ( fabs(mcEta_) <=1.) {
	h_simConvVtxRvsZ_[1] ->Fill ( fabs (mcConvZ_), mcConvR_  ) ;
	h_simConvVtxYvsX_ ->Fill ( mcConvX_, mcConvY_  ) ;
      }
      else
	h_simConvVtxRvsZ_[2] ->Fill ( fabs (mcConvZ_), mcConvR_  ) ;

      //std::cout << " TkConvValidator  theConvTP_ size " <<   theConvTP_.size() << std::endl;
      for ( edm::RefVector<TrackingParticleCollection>::iterator iTP= theConvTP_.begin(); iTP!=theConvTP_.end(); iTP++)
	{
	  //  std::cout << " SIM to RECO TP vertex " << (*iTP)->vx() << " " <<  (*iTP)->vy() << " " << (*iTP)->vz() << " pt " << (*iTP)->pt() << std::endl;
	}

     bool recomatch = false;
     float chi2Prob = 0.;
      //////////////////Measure reco efficiencies
     // cout << " size of conversions " << convHandle->size() << endl;
     for (reco::ConversionCollection::const_iterator conv = convHandle->begin();conv!=convHandle->end();++conv) {

	const reco::Conversion aConv = (*conv);
        if ( arbitratedMerged_ && !aConv.quality(reco::Conversion::arbitratedMerged)  ) continue;
	if ( generalTracksOnly_ && !aConv.quality(reco::Conversion::generalTracksOnly) ) continue;
        if ( arbitratedEcalSeeded_ && !aConv.quality(reco::Conversion::arbitratedEcalSeeded)  ) continue;


        if ( highPurity_ && !aConv.quality(reco::Conversion::highPurity) ) continue;

	//problematic?
	std::vector<edm::RefToBase<reco::Track> > tracks = aConv.tracks();


	const reco::Vertex& vtx = aConv.conversionVertex();
	//requires two tracks and a valid vertex
	if (tracks.size() !=2 || !(vtx.isValid())) continue;


        if (ChiSquaredProbability( aConv.conversionVertex().chi2(),  aConv.conversionVertex().ndof() ) <= minProb_) continue;
        if (aConv.nHitsBeforeVtx().size()>1 && max(aConv.nHitsBeforeVtx().at(0),aConv.nHitsBeforeVtx().at(1)) > maxHitsBeforeVtx_ ) continue;


        //compute transverse decay length with respect to beamspot
        math::XYZVectorF  themom = aConv.refittedPairMomentum();
        double dbsx = aConv.conversionVertex().x() - thebs.x0();
        double dbsy = aConv.conversionVertex().y() - thebs.y0();
        double lxy = (themom.x()*dbsx + themom.y()*dbsy)/themom.rho();

        if (lxy<minLxy_) continue;

	//	bool  phoIsInBarrel=false; // unused
	//	bool  phoIsInEndcap=false; // unused
	RefToBase<reco::Track> tfrb1 = aConv.tracks().front();
	RefToBase<reco::Track> tfrb2 = aConv.tracks().back();

        if ( ecalalgotracks_ && ( !(tfrb1->algo()==15 || tfrb1->algo()==16) || !(tfrb2->algo()==15 || tfrb2->algo()==16)  )  ) continue;


	//reco::TrackRef tk1 = aConv.tracks().front();
	//reco::TrackRef tk2 = aConv.tracks().back();
	//std::cout << "SIM to RECO  conversion track pt " << tk1->pt() << " " << tk2->pt() << endl;
	//
	//Use two RefToBaseVector and do two association actions to avoid that if two tracks from different collection
	RefToBaseVector<reco::Track> tc1, tc2;
	tc1.push_back(tfrb1);
	tc2.push_back(tfrb2);
	bool isAssociated = false;
	reco::SimToRecoCollection q1 = theTrackAssociator_->associateSimToReco(tc1,theConvTP_,&e);
	reco::SimToRecoCollection q2 = theTrackAssociator_->associateSimToReco(tc2,theConvTP_,&e);
	//try {
	  std::vector<std::pair<RefToBase<reco::Track>, double> > trackV1, trackV2;

	  int tp_1 = 0, tp_2 = 1;//the index of associated tp in theConvTP_ for two tracks
	  if (q1.find(theConvTP_[0])!=q1.end()){
	      trackV1 = (std::vector<std::pair<RefToBase<reco::Track>, double> >) q1[theConvTP_[0]];
	  } else if (q1.find(theConvTP_[1])!=q1.end()){
	      trackV1 = (std::vector<std::pair<RefToBase<reco::Track>, double> >) q1[theConvTP_[1]];
	      tp_1 = 1;
	  }
	  if (q2.find(theConvTP_[1])!=q2.end()){
	      trackV2 = (std::vector<std::pair<RefToBase<reco::Track>, double> >) q2[theConvTP_[1]];
	  } else if (q2.find(theConvTP_[0])!=q2.end()){
	      trackV2 = (std::vector<std::pair<RefToBase<reco::Track>, double> >) q2[theConvTP_[0]];
	      tp_2 = 0;
	  }
	  if (!(trackV1.size()&&trackV2.size()))
	      continue;
	  if (tp_1 == tp_2) continue;

	  edm::RefToBase<reco::Track> tr1 = trackV1.front().first;
	  edm::RefToBase<reco::Track> tr2 = trackV2.front().first;
	  //std::cout << "associated tp1 " <<theConvTP_[0]->pt() << " to track with  pT=" << tr1->pt() << " " << (tr1.get())->pt() << endl;
	  //std::cout << "associated tp2 " <<theConvTP_[1]->pt() << " to track with  pT=" << tr2->pt() << " " << (tr2.get())->pt() << endl;
	  myAss.insert( std::make_pair (tr1.get(),theConvTP_[tp_1] ) );
	  myAss.insert( std::make_pair (tr2.get(),theConvTP_[tp_2]) );

	//} catch (Exception event) {
	  //cout << "continue: " << event.what()  << endl;
	//  continue;
	//}


	isAssociated = true;
        recomatch = true;
        chi2Prob = ChiSquaredProbability( aConv.conversionVertex().chi2(),  aConv.conversionVertex().ndof() );

        if (isAssociated) {
          h_SimRecConvTwoMTracks_[0]->Fill( mcEta_ ) ;
          h_SimRecConvTwoMTracks_[1]->Fill( mcPhi_ );
          h_SimRecConvTwoMTracks_[2]->Fill( mcConvR_ );
          h_SimRecConvTwoMTracks_[3]->Fill( mcConvZ_ );
          h_SimRecConvTwoMTracks_[4]->Fill(  (*mcPho).fourMomentum().et());
        }

       // break;
      } // loop over reco conversions
      if (recomatch) {
        ////////// Numerators for conversion efficiencies, both tracks are associated
        h_SimConvTwoMTracks_[0]->Fill( mcEta_ ) ;
        h_SimConvTwoMTracks_[1]->Fill( mcPhi_ );
        h_SimConvTwoMTracks_[2]->Fill( mcConvR_ );
        h_SimConvTwoMTracks_[3]->Fill( mcConvZ_ );
        h_SimConvTwoMTracks_[4]->Fill(  (*mcPho).fourMomentum().et());


        if (   chi2Prob > 0) {
          h_SimConvTwoMTracksAndVtxPGT0_[0]->Fill( mcEta_ ) ;
          h_SimConvTwoMTracksAndVtxPGT0_[1]->Fill( mcPhi_ );
          h_SimConvTwoMTracksAndVtxPGT0_[2]->Fill( mcConvR_ );
          h_SimConvTwoMTracksAndVtxPGT0_[3]->Fill( mcConvZ_ );
          h_SimConvTwoMTracksAndVtxPGT0_[4]->Fill(  (*mcPho).fourMomentum().et());
        }
        if (   chi2Prob > 0.0005) {
          h_SimConvTwoMTracksAndVtxPGT0005_[0]->Fill( mcEta_ ) ;
          h_SimConvTwoMTracksAndVtxPGT0005_[1]->Fill( mcPhi_ );
          h_SimConvTwoMTracksAndVtxPGT0005_[2]->Fill( mcConvR_ );
          h_SimConvTwoMTracksAndVtxPGT0005_[3]->Fill( mcConvZ_ );
          h_SimConvTwoMTracksAndVtxPGT0005_[4]->Fill(  (*mcPho).fourMomentum().et());

        }
      }

  } //End loop over simulated conversions


  // ########################### RECO to SIM ############################## //

  for (reco::ConversionCollection::const_iterator conv = convHandle->begin();conv!=convHandle->end();++conv) {
    const reco::Conversion aConv = (*conv);
    if ( arbitratedMerged_ && !aConv.quality(reco::Conversion::arbitratedMerged)  ) continue;
    if ( generalTracksOnly_ && !aConv.quality(reco::Conversion::generalTracksOnly) ) continue;
    if ( arbitratedEcalSeeded_ && !aConv.quality(reco::Conversion::arbitratedEcalSeeded)  ) continue;


    if ( highPurity_ && !aConv.quality(reco::Conversion::highPurity) ) continue;

    //problematic?
    std::vector<edm::RefToBase<reco::Track> > tracks = aConv.tracks();

    const reco::Vertex& vtx = aConv.conversionVertex();
    //requires two tracks and a valid vertex
    if (tracks.size() !=2 || !(vtx.isValid())) continue;
    //if (tracks.size() !=2) continue;


    if (ChiSquaredProbability( aConv.conversionVertex().chi2(),  aConv.conversionVertex().ndof() ) <= minProb_) continue;
    if (aConv.nHitsBeforeVtx().size()>1 && max(aConv.nHitsBeforeVtx().at(0),aConv.nHitsBeforeVtx().at(1)) > maxHitsBeforeVtx_ ) continue;

    //compute transverse decay length with respect to beamspot
    math::XYZVectorF  themom = aConv.refittedPairMomentum();
    double dbsx = aConv.conversionVertex().x() - thebs.x0();
    double dbsy = aConv.conversionVertex().y() - thebs.y0();
    double lxy = (themom.x()*dbsx + themom.y()*dbsy)/themom.rho();

    if (lxy<minLxy_) continue;

    bool  phoIsInBarrel=false;
    bool  phoIsInEndcap=false;
    RefToBase<reco::Track> tk1 = aConv.tracks().front();
    RefToBase<reco::Track> tk2 = aConv.tracks().back();
    RefToBaseVector<reco::Track> tc1, tc2;
    tc1.push_back(tk1);
    tc2.push_back(tk2);

   if ( ecalalgotracks_ && ( !(tk1->algo()==15 || tk1->algo()==16) || !(tk2->algo()==15 || tk2->algo()==16)  )  ) continue;


    //std::cout << " RECO to SIM conversion track pt " << tk1->pt() << " " << tk2->pt() << endl;
    const reco::Track refTk1 = aConv.conversionVertex().refittedTracks().front();
    const reco::Track refTk2 = aConv.conversionVertex().refittedTracks().back();

    //TODO replace it with phi at vertex
    float  dPhiTracksAtVtx =  aConv.dPhiTracksAtVtx();
    // override with the phi calculated at the vertex
    math::XYZVector p1AtVtx= recalculateMomentumAtFittedVertex (  (*theMF_), *trackerGeom, tk1,  aConv.conversionVertex() );
    math::XYZVector p2AtVtx= recalculateMomentumAtFittedVertex (  (*theMF_), *trackerGeom, tk2,  aConv.conversionVertex() );
    if (  sqrt(p1AtVtx.perp2())  >  sqrt(p2AtVtx.perp2())  )
      dPhiTracksAtVtx = p1AtVtx.phi() - p2AtVtx.phi();
    else
      dPhiTracksAtVtx = p2AtVtx.phi() - p1AtVtx.phi();


    math::XYZVectorF refittedMom =  aConv.refittedPairMomentum();


    if (fabs(refittedMom.eta())< 1.479 ) {
      phoIsInBarrel=true;
    } else {
      phoIsInEndcap=true;
    }

    nRecConv_++;

    // check matching with reco photon 
    double Mindeltaeta = 999999;
    double Mindeltaphi = 999999;
    bool matchConvSC=false;
    reco::PhotonCollection::const_iterator  iMatchingSC;
    for( reco::PhotonCollection::const_iterator  iPho = photonCollection.begin(); iPho != photonCollection.end(); iPho++) {
      reco::Photon aPho = reco::Photon(*iPho);
      const double deltaphi= reco::deltaPhi( aConv.refittedPairMomentum().phi(), aPho.superCluster()->position().phi());
      double ConvEta = etaTransformation(aConv.refittedPairMomentum().eta(),aConv.zOfPrimaryVertexFromTracks());
      double deltaeta = abs( aPho.superCluster()->position().eta() -ConvEta);
      if (abs(deltaeta)<abs(Mindeltaeta) && abs(deltaphi)<abs(Mindeltaphi)) {
	Mindeltaphi=abs(deltaphi);
	Mindeltaeta=abs(deltaeta);
	iMatchingSC = iPho ;
      }
    }
    if (abs(Mindeltaeta)<0.1 && abs(Mindeltaphi)<0.1) {
      matchConvSC=true;
    }
  

    ///////////  Quantities per conversion
    int match =0;
    float invM=aConv.pairInvariantMass();
    float chi2Prob = ChiSquaredProbability( aConv.conversionVertex().chi2(),  aConv.conversionVertex().ndof() );
    uint maxNHitsBeforeVtx = aConv.nHitsBeforeVtx().size()>1 ? max(aConv.nHitsBeforeVtx().at(0),aConv.nHitsBeforeVtx().at(1)) : 0;
    uint sumNHitsBeforeVtx = aConv.nHitsBeforeVtx().size()>1 ? aConv.nHitsBeforeVtx().at(0) + aConv.nHitsBeforeVtx().at(1) : 0;
    float maxDlClosestHitToVtx = aConv.dlClosestHitToVtx().size()>1 ? max(aConv.dlClosestHitToVtx().at(0).value(),aConv.dlClosestHitToVtx().at(1).value()) : 0;
    float maxDlClosestHitToVtxSig = aConv.dlClosestHitToVtx().size()>1 ? max(aConv.dlClosestHitToVtx().at(0).value()/aConv.dlClosestHitToVtx().at(0).error(),aConv.dlClosestHitToVtx().at(1).value()/aConv.dlClosestHitToVtx().at(1).error()) : 0;

    int ilead = 0, itrail = 1;
    if (tk2->pt() > tk1->pt()) {
      ilead = 1;
      itrail = 0;
    }
    RefToBase<reco::Track> tklead = aConv.tracks().at(ilead);
    RefToBase<reco::Track> tktrail = aConv.tracks().at(itrail);

    int deltaExpectedHitsInner = tklead->trackerExpectedHitsInner().numberOfHits() - tktrail->trackerExpectedHitsInner().numberOfHits();
    int leadExpectedHitsInner = tklead->trackerExpectedHitsInner().numberOfHits();
    uint leadNHitsBeforeVtx = aConv.nHitsBeforeVtx().size()>1 ? aConv.nHitsBeforeVtx().at(ilead) : 0;
    uint trailNHitsBeforeVtx = aConv.nHitsBeforeVtx().size()>1 ? aConv.nHitsBeforeVtx().at(itrail) : 0;


    h_convEta_[match][0]->Fill( refittedMom.eta() );
    h_convEta2_[match][0]->Fill( refittedMom.eta() );

    h_convPhi_[match][0]->Fill( refittedMom.phi() );
    h_convR_[match][0]->Fill( sqrt(aConv.conversionVertex().position().perp2()) );
    h_convRplot_->Fill( sqrt(aConv.conversionVertex().position().perp2()) );
    h_convZ_[match][0]->Fill( aConv.conversionVertex().position().z() );
    h_convZplot_->Fill( aConv.conversionVertex().position().z() );
    h_convPt_[match][0]->Fill(  sqrt(refittedMom.perp2()) );
    h_invMass_[match][0] ->Fill( invM);
    h_vtxChi2Prob_[match][0] ->Fill (chi2Prob);
    h_lxybs_[match][0] ->Fill (lxy);
    h_maxNHitsBeforeVtx_[match][0] ->Fill (maxNHitsBeforeVtx);
    h_leadNHitsBeforeVtx_[match][0] ->Fill (leadNHitsBeforeVtx);
    h_trailNHitsBeforeVtx_[match][0] ->Fill (trailNHitsBeforeVtx);
    h_sumNHitsBeforeVtx_[match][0] ->Fill (sumNHitsBeforeVtx);
    h_deltaExpectedHitsInner_[match][0] ->Fill (deltaExpectedHitsInner);
    h_leadExpectedHitsInner_[match][0] ->Fill (leadExpectedHitsInner);
    h_maxDlClosestHitToVtx_[match][0] ->Fill (maxDlClosestHitToVtx);
    h_maxDlClosestHitToVtxSig_[match][0] ->Fill (maxDlClosestHitToVtxSig);
    h_nSharedHits_[match][0] ->Fill (aConv.nSharedHits());


    if (  matchConvSC ) {
      h_convEtaMatchSC_[match][0]->Fill( refittedMom.eta() );
      h_EoverPTracks_[match][0] ->Fill (iMatchingSC->superCluster()->energy()/sqrt(refittedMom.mag2()));
      h_convSCdPhi_[match][0]->Fill( iMatchingSC->superCluster()->position().phi() - refittedMom.phi() );
      double ConvEta = etaTransformation(aConv.refittedPairMomentum().eta(),aConv.zOfPrimaryVertexFromTracks());
      h_convSCdEta_[match][0]->Fill( iMatchingSC->superCluster()->position().eta() - ConvEta );
    }

    h_distMinAppTracks_[match][0] ->Fill (aConv.distOfMinimumApproach());
    h_DPhiTracksAtVtx_[match][0]->Fill( dPhiTracksAtVtx);
    h2_DPhiTracksAtVtxVsEta_->Fill( mcEta_, dPhiTracksAtVtx);
    h2_DPhiTracksAtVtxVsR_->Fill( mcConvR_, dPhiTracksAtVtx);
    p_DPhiTracksAtVtxVsEta_->Fill( mcEta_, dPhiTracksAtVtx);
    p_DPhiTracksAtVtxVsR_->Fill( mcConvR_, dPhiTracksAtVtx);

    h_DCotTracks_[match][0] ->Fill ( aConv.pairCotThetaSeparation() );
    h2_DCotTracksVsEta_->Fill( mcEta_, aConv.pairCotThetaSeparation() );
    h2_DCotTracksVsR_->Fill( mcConvR_, aConv.pairCotThetaSeparation() );
    p_DCotTracksVsEta_->Fill( mcEta_, aConv.pairCotThetaSeparation() );
    p_DCotTracksVsR_->Fill( mcConvR_, aConv.pairCotThetaSeparation() );

    if ( phoIsInBarrel ) {
      h_invMass_[match][1] ->Fill(invM);
      h_vtxChi2Prob_[match][1] ->Fill (chi2Prob);
      h_distMinAppTracks_[match][1] ->Fill (aConv.distOfMinimumApproach());
      h_DPhiTracksAtVtx_[match][1]->Fill( dPhiTracksAtVtx);
      h_DCotTracks_[match][1] ->Fill ( aConv.pairCotThetaSeparation() );
      h_lxybs_[match][1] ->Fill (lxy);
      h_maxNHitsBeforeVtx_[match][1] ->Fill (maxNHitsBeforeVtx);
      h_leadNHitsBeforeVtx_[match][1] ->Fill (leadNHitsBeforeVtx);
      h_trailNHitsBeforeVtx_[match][1] ->Fill (trailNHitsBeforeVtx);
      h_sumNHitsBeforeVtx_[match][1] ->Fill (sumNHitsBeforeVtx);
      h_deltaExpectedHitsInner_[match][1] ->Fill (deltaExpectedHitsInner);
      h_leadExpectedHitsInner_[match][1] ->Fill (leadExpectedHitsInner);
      h_maxDlClosestHitToVtx_[match][1] ->Fill (maxDlClosestHitToVtx);
      h_maxDlClosestHitToVtxSig_[match][1] ->Fill (maxDlClosestHitToVtxSig);
      h_nSharedHits_[match][1] ->Fill (aConv.nSharedHits());

      /*
      if ( aConv.caloCluster().size() ) { 
	h_convSCdPhi_[match][1]->Fill( 	aConv.caloCluster()[0]->phi() - refittedMom.phi() );
	double ConvEta = etaTransformation(aConv.refittedPairMomentum().eta(),aConv.zOfPrimaryVertexFromTracks());  
	h_convSCdEta_[match][1]->Fill( aConv.caloCluster()[0]->eta() - ConvEta );
      }
      */

      if (  matchConvSC ) {
	h_EoverPTracks_[match][1] -> Fill(iMatchingSC->superCluster()->energy()/sqrt(refittedMom.mag2()));
	h_convSCdPhi_[match][1]->Fill( iMatchingSC->superCluster()->position().phi() - refittedMom.phi() );
	double ConvEta = etaTransformation(aConv.refittedPairMomentum().eta(),aConv.zOfPrimaryVertexFromTracks());
	h_convSCdEta_[match][1]->Fill( iMatchingSC->superCluster()->position().eta() - ConvEta );
      }
    }


    if ( phoIsInEndcap ) {
      h_invMass_[match][2] ->Fill(invM);
      h_vtxChi2Prob_[match][2] ->Fill (chi2Prob);
      h_distMinAppTracks_[match][2] ->Fill (aConv.distOfMinimumApproach());
      h_DPhiTracksAtVtx_[match][2]->Fill( dPhiTracksAtVtx);
      h_DCotTracks_[match][2] ->Fill ( aConv.pairCotThetaSeparation() );
      h_lxybs_[match][2] ->Fill (lxy);
      h_maxNHitsBeforeVtx_[match][2] ->Fill (maxNHitsBeforeVtx);
      h_leadNHitsBeforeVtx_[match][2] ->Fill (leadNHitsBeforeVtx);
      h_trailNHitsBeforeVtx_[match][2] ->Fill (trailNHitsBeforeVtx);
      h_sumNHitsBeforeVtx_[match][2] ->Fill (sumNHitsBeforeVtx);
      h_deltaExpectedHitsInner_[match][2] ->Fill (deltaExpectedHitsInner);
      h_leadExpectedHitsInner_[match][2] ->Fill (leadExpectedHitsInner);
      h_maxDlClosestHitToVtx_[match][2] ->Fill (maxDlClosestHitToVtx);
      h_maxDlClosestHitToVtxSig_[match][2] ->Fill (maxDlClosestHitToVtxSig);
      h_nSharedHits_[match][2] ->Fill (aConv.nSharedHits());
      if (  matchConvSC ) {
	h_EoverPTracks_[match][2] ->Fill (iMatchingSC->superCluster()->energy()/sqrt(refittedMom.mag2()));
	h_convSCdPhi_[match][2]->Fill( iMatchingSC->superCluster()->position().phi() - refittedMom.phi() );
	double ConvEta = etaTransformation(aConv.refittedPairMomentum().eta(),aConv.zOfPrimaryVertexFromTracks());
	h_convSCdEta_[match][2]->Fill( iMatchingSC->superCluster()->position().eta() - ConvEta );

      }
    }

    h_convVtxRvsZ_[0] ->Fill ( fabs (aConv.conversionVertex().position().z() ),  sqrt(aConv.conversionVertex().position().perp2())  ) ;
    h_convVtxYvsX_ ->Fill (  aConv.conversionVertex().position().x(), aConv.conversionVertex().position().y() );
    h_convVtxYvsX_zoom_[0] ->Fill (  aConv.conversionVertex().position().x(), aConv.conversionVertex().position().y() );
    h_convVtxYvsX_zoom_[1] ->Fill (  aConv.conversionVertex().position().x(), aConv.conversionVertex().position().y() );


    // quantities per track: all conversions
    for (unsigned int i=0; i<tracks.size(); i++) {
      double d0;
      if (valid_pvtx){
	d0 = - tracks[i]->dxy(the_pvtx.position());
      } else {
	d0 = tracks[i]->d0();
      }
      h_TkD0_[match]->Fill ( d0* tracks[i]->charge() );
      h_nHitsBeforeVtx_[match]->Fill ( aConv.nHitsBeforeVtx().size()>1 ? aConv.nHitsBeforeVtx().at(i) : 0 );
      h_dlClosestHitToVtx_[match]->Fill ( aConv.dlClosestHitToVtx().size()>1 ? aConv.dlClosestHitToVtx().at(i).value() : 0 );
      h_dlClosestHitToVtxSig_[match]->Fill ( aConv.dlClosestHitToVtx().size()>1 ? aConv.dlClosestHitToVtx().at(i).value()/aConv.dlClosestHitToVtx().at(i).error() : 0 );

      nHitsVsEta_[match] ->Fill (mcEta_,   float(tracks[i]->numberOfValidHits()) );
      nHitsVsR_[match] ->Fill (mcConvR_,   float(tracks[i]->numberOfValidHits()) );
      p_nHitsVsEta_[match] ->Fill (mcEta_,   float(tracks[i]->numberOfValidHits()) -0.0001);
      p_nHitsVsR_[match] ->Fill (mcConvR_,   float(tracks[i]->numberOfValidHits()) -0.0001);
      h_tkChi2_[match] ->Fill (tracks[i]->normalizedChi2() );
      h_tkChi2Large_[match] ->Fill (tracks[i]->normalizedChi2() );
      h2_Chi2VsEta_[match] ->Fill(  mcEta_, tracks[i]->normalizedChi2() );
      h2_Chi2VsR_[match] ->Fill(  mcConvR_, tracks[i]->normalizedChi2() );
      p_Chi2VsEta_[match] ->Fill(  mcEta_, tracks[i]->normalizedChi2() );
      p_Chi2VsR_[match] ->Fill(  mcConvR_, tracks[i]->normalizedChi2() );

    }

    bool associated = false;
    float mcConvPt_= -99999999;
    //    float mcPhi= 0; // unused
    float simPV_Z=0;
    for ( std::vector<PhotonMCTruth>::const_iterator mcPho=mcPhotons.begin(); mcPho !=mcPhotons.end(); mcPho++) {
      mcConvPt_= (*mcPho).fourMomentum().et();
      float mcPhi= (*mcPho).fourMomentum().phi();
      simPV_Z = (*mcPho).primaryVertex().z();
      mcPhi_= phiNormalization(mcPhi);
      mcEta_= (*mcPho).fourMomentum().pseudoRapidity();
      mcEta_ = etaTransformation(mcEta_, (*mcPho).primaryVertex().z() );
      mcConvR_= (*mcPho).vertex().perp();
      mcConvX_= (*mcPho).vertex().x();
      mcConvY_= (*mcPho).vertex().y();
      mcConvZ_= (*mcPho).vertex().z();
      mcConvEta_= (*mcPho).vertex().eta();
      mcConvPhi_= (*mcPho).vertex().phi();
      if ( fabs(mcEta_) > END_HI ) continue;
      if (mcConvPt_<minPhoPtForPurity) continue;
      if (fabs(mcEta_)>maxPhoEtaForPurity) continue;
      if (fabs(mcConvZ_)>maxPhoZForPurity) continue;
      if (mcConvR_>maxPhoRForEffic) continue;

      if (  (*mcPho).isAConversion() != 1 ) continue;
      if (!( ( fabs(mcEta_) <= BARL && mcConvR_ <85 )  ||
	     ( fabs(mcEta_) > BARL && fabs(mcEta_) <=END_HI && fabs( (*mcPho).vertex().z() ) < 210 )  ) )
	continue;


      theConvTP_.clear();
      for(size_t i = 0; i < tpForFakeRate.size(); ++i){
	TrackingParticleRef tp (TPHandleForFakeRate,i);
	if ( fabs( tp->vx() - (*mcPho).vertex().x() ) < 0.0001   &&
	     fabs( tp->vy() - (*mcPho).vertex().y() ) < 0.0001   &&
	     fabs( tp->vz() - (*mcPho).vertex().z() ) < 0.0001) {
	  theConvTP_.push_back( tp );


	}
      }

      if ( theConvTP_.size() < 2 )   continue;

      //associated = false;
      reco::RecoToSimCollection p1 =  theTrackAssociator_->associateRecoToSim(tc1,theConvTP_,&e);
      reco::RecoToSimCollection p2 =  theTrackAssociator_->associateRecoToSim(tc2,theConvTP_,&e);
      try{
	std::vector<std::pair<TrackingParticleRef, double> > tp1 = p1[tk1];
	std::vector<std::pair<TrackingParticleRef, double> > tp2 = p2[tk2];
	if (!(tp1.size()&&tp2.size())){
	    tp1 = p1[tk2];
	    tp2 = p2[tk1];
	}
	if (tp1.size()&&tp2.size()) {
	  TrackingParticleRef tpr1 = tp1.front().first;
	  TrackingParticleRef tpr2 = tp2.front().first;
	  if (abs(tpr1->pdgId())==11&&abs(tpr2->pdgId())==11&& tpr1->pdgId()*tpr2->pdgId()<0) {
	    if ( (tpr1->parentVertex()->sourceTracks_end()-tpr1->parentVertex()->sourceTracks_begin()==1) &&
		 (tpr2->parentVertex()->sourceTracks_end()-tpr2->parentVertex()->sourceTracks_begin()==1)) {
	      if (tpr1->parentVertex().key()==tpr2->parentVertex().key() && ((*tpr1->parentVertex()->sourceTracks_begin())->pdgId()==22)) {
		mcConvR_ = sqrt(tpr1->parentVertex()->position().Perp2());
		mcConvZ_ = tpr1->parentVertex()->position().z();
		mcConvX_ = tpr1->parentVertex()->position().x();
		mcConvY_ = tpr1->parentVertex()->position().y();
		mcConvEta_ = tpr1->parentVertex()->position().eta();
		mcConvPhi_ = tpr1->parentVertex()->position().phi();
		mcConvPt_ = sqrt((*tpr1->parentVertex()->sourceTracks_begin())->momentum().Perp2());
		//std::cout << " Reco to Sim mcconvpt " << mcConvPt_ << std::endl;
		//cout << "associated track1 to " << tpr1->pdgId() << " with p=" << tpr1->p4() << " with pT=" << tpr1->pt() << endl;
		//cout << "associated track2 to " << tpr2->pdgId() << " with p=" << tpr2->p4() << " with pT=" << tpr2->pt() << endl;
		associated = true;
                break;
	      }
	    }
	  }
	}
      } catch (Exception event) {
	//cout << "do not continue: " << event.what()  << endl;
	//continue;
      }

    }// end loop on sim photons


    if (0) {
        theConvTP_.clear();
        for(size_t i = 0; i < tpForFakeRate.size(); ++i){
          TrackingParticleRef tp (TPHandleForFakeRate,i);
            theConvTP_.push_back( tp );
        }
        reco::RecoToSimCollection p1incl =  theTrackAssociator_->associateRecoToSim(tc1,theConvTP_,&e);
        reco::RecoToSimCollection p2incl =  theTrackAssociator_->associateRecoToSim(tc2,theConvTP_,&e);


      for ( std::vector<PhotonMCTruth>::const_iterator mcPho=mcPhotons.begin(); mcPho !=mcPhotons.end(); mcPho++) {
        mcConvPt_= (*mcPho).fourMomentum().et();
        float mcPhi= (*mcPho).fourMomentum().phi();
        simPV_Z = (*mcPho).primaryVertex().z();
        mcPhi_= phiNormalization(mcPhi);
        mcEta_= (*mcPho).fourMomentum().pseudoRapidity();
        mcEta_ = etaTransformation(mcEta_, (*mcPho).primaryVertex().z() );
        mcConvR_= (*mcPho).vertex().perp();
        mcConvX_= (*mcPho).vertex().x();
        mcConvY_= (*mcPho).vertex().y();
        mcConvZ_= (*mcPho).vertex().z();
        mcConvEta_= (*mcPho).vertex().eta();
        mcConvPhi_= (*mcPho).vertex().phi();
        if ( fabs(mcEta_) > END_HI ) continue;
        if (mcConvPt_<minPhoPtForPurity) continue;
        if (fabs(mcEta_)>maxPhoEtaForPurity) continue;
        if (fabs(mcConvZ_)>maxPhoZForPurity) continue;
        if (mcConvR_>maxPhoRForEffic) continue;

        if (  (*mcPho).isAConversion() != 1 ) continue;
        if (!( ( fabs(mcEta_) <= BARL && mcConvR_ <85 )  ||
              ( fabs(mcEta_) > BARL && fabs(mcEta_) <=END_HI && fabs( (*mcPho).vertex().z() ) < 210 )  ) )
          continue;


        theConvTP_.clear();
        for(size_t i = 0; i < tpForFakeRate.size(); ++i){
          TrackingParticleRef tp (TPHandleForFakeRate,i);
          if ( fabs( tp->vx() - (*mcPho).vertex().x() ) < 0.0001   &&
              fabs( tp->vy() - (*mcPho).vertex().y() ) < 0.0001   &&
              fabs( tp->vz() - (*mcPho).vertex().z() ) < 0.0001) {
            theConvTP_.push_back( tp );


          }
        }

        if ( theConvTP_.size() < 2 )   continue;

        //associated = false;
        reco::RecoToSimCollection p1 =  theTrackAssociator_->associateRecoToSim(tc1,theConvTP_,&e);
        reco::RecoToSimCollection p2 =  theTrackAssociator_->associateRecoToSim(tc2,theConvTP_,&e);





          if ( (p1incl.size() && p2incl.size()) && (p1.size() || p2.size()) ) { // associated = true;
            try{
              std::vector<std::pair<TrackingParticleRef, double> > tp1 = p1incl[tk1];
              std::vector<std::pair<TrackingParticleRef, double> > tp2 = p2incl[tk2];
              if (!(tp1.size()&&tp2.size())){
                  tp1 = p1[tk2];
                  tp2 = p2[tk1];
              }
              if (tp1.size()&&tp2.size()) {
                TrackingParticleRef tpr1 = tp1.front().first;
                TrackingParticleRef tpr2 = tp2.front().first;
                if (abs(tpr1->pdgId())==11&&abs(tpr2->pdgId())==11 && tpr1->pdgId()*tpr2->pdgId()<0) {
                  if ( ((tpr1->parentVertex()->sourceTracks_end()-tpr1->parentVertex()->sourceTracks_begin()>=1) && (*tpr1->parentVertex()->sourceTracks_begin())->pdgId()==22) &&
                      ((tpr2->parentVertex()->sourceTracks_end()-tpr2->parentVertex()->sourceTracks_begin()>=1) && (*tpr2->parentVertex()->sourceTracks_begin())->pdgId()==22) ) {

                   // if ( fabs(tpr1->vx() - tpr2->vx()) < 0.1 && fabs(tpr1->vy() - tpr2->vy()) < 0.1 && fabs(tpr1->vz() - tpr2->vz()) < 0.1) {
                    //if (((*tpr1->parentVertex()->sourceTracks_begin())->pdgId()==22) || ((*tpr2->parentVertex()->sourceTracks_begin())->pdgId()==22)) {
//                       mcConvR_ = sqrt(tpr1->parentVertex()->position().Perp2());
//                       mcConvZ_ = tpr1->parentVertex()->position().z();
//                       mcConvX_ = tpr1->parentVertex()->position().x();
//                       mcConvY_ = tpr1->parentVertex()->position().y();
//                       mcConvEta_ = tpr1->parentVertex()->position().eta();
//                       mcConvPhi_ = tpr1->parentVertex()->position().phi();
//                       mcConvPt_ = sqrt((*tpr1->parentVertex()->sourceTracks_begin())->momentum().Perp2());
                      //std::cout << " Reco to Sim mcconvpt " << mcConvPt_ << std::endl;
                      //cout << "associated track1 to " << tpr1->pdgId() << " with p=" << tpr1->p4() << " with pT=" << tpr1->pt() << endl;
                      //cout << "associated track2 to " << tpr2->pdgId() << " with p=" << tpr2->p4() << " with pT=" << tpr2->pt() << endl;
                      associated = true;
                      break;
                    //}
                    //}
                  }
                }
              }
            } catch (Exception event) {
              //cout << "do not continue: " << event.what()  << endl;
              //continue;
            }

          }

        }
      }

      if ( associated ) match=1;
      else
	match=2;

      h_match_->Fill(float(match));
      //////// here reco is matched to sim or is fake
      if ( match == 1) nRecConvAss_++;
      h_convEta_[match][0]->Fill( refittedMom.eta() );
      h_convEta_[match][1]->Fill( refittedMom.eta() );
      if (matchConvSC) h_convEtaMatchSC_[match][0]->Fill( refittedMom.eta() );
      h_convPhi_[match][0]->Fill( refittedMom.phi() );
      h_convR_[match][0]->Fill( sqrt(aConv.conversionVertex().position().perp2()) );
      h_convZ_[match][0]->Fill( aConv.conversionVertex().position().z() );
      h_convPt_[match][0]->Fill(  sqrt(refittedMom.perp2()) );
      h_invMass_[match][0] ->Fill( invM);
      h_vtxChi2Prob_[match][0] ->Fill (chi2Prob);
      h_DPhiTracksAtVtx_[match][0]->Fill( dPhiTracksAtVtx);
      h_DCotTracks_[match][0] ->Fill ( aConv.pairCotThetaSeparation() );
      h_distMinAppTracks_[match][0] ->Fill (aConv.distOfMinimumApproach());
      h_lxybs_[match][0] ->Fill (lxy);
      h_maxNHitsBeforeVtx_[match][0] ->Fill (maxNHitsBeforeVtx);
      h_leadNHitsBeforeVtx_[match][0] ->Fill (leadNHitsBeforeVtx);
      h_trailNHitsBeforeVtx_[match][0] ->Fill (trailNHitsBeforeVtx);
      h_sumNHitsBeforeVtx_[match][0] ->Fill (sumNHitsBeforeVtx);
      h_deltaExpectedHitsInner_[match][0] ->Fill (deltaExpectedHitsInner);
      h_leadExpectedHitsInner_[match][0] ->Fill (leadExpectedHitsInner);
      h_maxDlClosestHitToVtx_[match][0] ->Fill (maxDlClosestHitToVtx);
      h_maxDlClosestHitToVtxSig_[match][0] ->Fill (maxDlClosestHitToVtxSig);
      h_nSharedHits_[match][0] ->Fill (aConv.nSharedHits());
      if (  matchConvSC ) {
	//h_EoverPTracks_[match][0] ->Fill (aConv.EoverPrefittedTracks());
	h_EoverPTracks_[match][0] ->Fill (iMatchingSC->superCluster()->energy()/sqrt(refittedMom.mag2()));
	h_convSCdPhi_[match][0]->Fill( iMatchingSC->superCluster()->position().phi() - refittedMom.phi() );
	double ConvEta = etaTransformation(aConv.refittedPairMomentum().eta(),aConv.zOfPrimaryVertexFromTracks());
	h_convSCdEta_[match][0]->Fill( iMatchingSC->superCluster()->position().eta() - ConvEta );
	
      }
      if ( match==1) {
	h2_photonPtRecVsPtSim_->Fill ( mcConvPt_, sqrt(refittedMom.perp2()) );
	h_convPtRes_[0]->Fill (  sqrt(refittedMom.perp2())/mcConvPt_);
      }

    if ( phoIsInBarrel ) {
      h_invMass_[match][1] ->Fill(invM);
      h_vtxChi2Prob_[match][1] ->Fill (chi2Prob);
      h_DPhiTracksAtVtx_[match][1]->Fill( dPhiTracksAtVtx);
      h_DCotTracks_[match][1] ->Fill ( aConv.pairCotThetaSeparation() );
      h_distMinAppTracks_[match][1] ->Fill (aConv.distOfMinimumApproach());
      h_lxybs_[match][1] ->Fill (lxy);
      h_maxNHitsBeforeVtx_[match][1] ->Fill (maxNHitsBeforeVtx);
      h_leadNHitsBeforeVtx_[match][1] ->Fill (leadNHitsBeforeVtx);
      h_trailNHitsBeforeVtx_[match][1] ->Fill (trailNHitsBeforeVtx);
      h_sumNHitsBeforeVtx_[match][1] ->Fill (sumNHitsBeforeVtx);
      h_deltaExpectedHitsInner_[match][1] ->Fill (deltaExpectedHitsInner);
      h_leadExpectedHitsInner_[match][1] ->Fill (leadExpectedHitsInner);
      h_maxDlClosestHitToVtx_[match][1] ->Fill (maxDlClosestHitToVtx);
      h_maxDlClosestHitToVtxSig_[match][1] ->Fill (maxDlClosestHitToVtxSig);
      h_nSharedHits_[match][1] ->Fill (aConv.nSharedHits());
      if (  matchConvSC ) {
	//	h_EoverPTracks_[match][1] ->Fill (aConv.EoverPrefittedTracks());
	h_EoverPTracks_[match][1] ->Fill (iMatchingSC->superCluster()->energy()/sqrt(refittedMom.mag2()));
	h_convSCdPhi_[match][1]->Fill( iMatchingSC->superCluster()->position().phi() - refittedMom.phi() );
	double ConvEta = etaTransformation(aConv.refittedPairMomentum().eta(),aConv.zOfPrimaryVertexFromTracks());
	h_convSCdEta_[match][1]->Fill( iMatchingSC->superCluster()->position().eta() - ConvEta );

      }
      if ( match==1) h_convPtRes_[1]->Fill (  sqrt(refittedMom.perp2())/mcConvPt_);
    }


    if ( phoIsInEndcap ) {
      h_invMass_[match][2] ->Fill(invM);
      h_vtxChi2Prob_[match][2] ->Fill (chi2Prob);
      h_DPhiTracksAtVtx_[match][2]->Fill( dPhiTracksAtVtx);
      h_DCotTracks_[match][2] ->Fill ( aConv.pairCotThetaSeparation() );
      h_distMinAppTracks_[match][2] ->Fill (aConv.distOfMinimumApproach());
      h_lxybs_[match][2] ->Fill (lxy);
      h_maxNHitsBeforeVtx_[match][2] ->Fill (maxNHitsBeforeVtx);
      h_leadNHitsBeforeVtx_[match][2] ->Fill (leadNHitsBeforeVtx);
      h_trailNHitsBeforeVtx_[match][2] ->Fill (trailNHitsBeforeVtx);
      h_sumNHitsBeforeVtx_[match][2] ->Fill (sumNHitsBeforeVtx);
      h_deltaExpectedHitsInner_[match][2] ->Fill (deltaExpectedHitsInner);
      h_leadExpectedHitsInner_[match][2] ->Fill (leadExpectedHitsInner);
      h_maxDlClosestHitToVtx_[match][2] ->Fill (maxDlClosestHitToVtx);
      h_maxDlClosestHitToVtxSig_[match][2] ->Fill (maxDlClosestHitToVtxSig);
      h_nSharedHits_[match][2] ->Fill (aConv.nSharedHits());
      if (  matchConvSC ) {
	//	h_EoverPTracks_[match][2] ->Fill (aConv.EoverPrefittedTracks());
	h_EoverPTracks_[match][2] ->Fill (iMatchingSC->superCluster()->energy()/sqrt(refittedMom.mag2()));
	h_convSCdPhi_[match][2]->Fill( iMatchingSC->superCluster()->position().phi() - refittedMom.phi() );
	double ConvEta = etaTransformation(aConv.refittedPairMomentum().eta(),aConv.zOfPrimaryVertexFromTracks());
	h_convSCdEta_[match][2]->Fill( iMatchingSC->superCluster()->position().eta() - ConvEta );
      }
      if ( match==1) h_convPtRes_[2]->Fill (  sqrt(refittedMom.perp2())/mcConvPt_);
    }


    if ( match == 1 ) {
      h_convVtxdX_ ->Fill ( aConv.conversionVertex().position().x() - mcConvX_);
      h_convVtxdY_ ->Fill ( aConv.conversionVertex().position().y() - mcConvY_);
      h_convVtxdZ_ ->Fill ( aConv.conversionVertex().position().z() - mcConvZ_);
      h_convVtxdR_ ->Fill ( sqrt(aConv.conversionVertex().position().perp2()) - mcConvR_);
      h_convVtxdPhi_ ->Fill ( aConv.conversionVertex().position().phi() - mcConvPhi_);
      h_convVtxdEta_ ->Fill ( aConv.conversionVertex().position().eta() - mcConvEta_);
      h2_convVtxdRVsR_ ->Fill (mcConvR_, sqrt(aConv.conversionVertex().position().perp2()) - mcConvR_ );
      h2_convVtxdRVsEta_ ->Fill (mcEta_, sqrt(aConv.conversionVertex().position().perp2()) - mcConvR_ );
      p_convVtxdRVsR_ ->Fill (mcConvR_, sqrt(aConv.conversionVertex().position().perp2()) - mcConvR_ );
      p_convVtxdRVsEta_ ->Fill (mcEta_, sqrt(aConv.conversionVertex().position().perp2()) - mcConvR_ );
      p_convVtxdXVsX_ ->Fill (mcConvX_, aConv.conversionVertex().position().x() - mcConvX_ );
      p_convVtxdYVsY_ ->Fill (mcConvY_, aConv.conversionVertex().position().y() - mcConvY_ );
      p_convVtxdZVsZ_ ->Fill (mcConvZ_, aConv.conversionVertex().position().z() - mcConvZ_ );
      p_convVtxdZVsR_ ->Fill (mcConvR_, aConv.conversionVertex().position().z() - mcConvZ_ );

      float dR=sqrt(aConv.conversionVertex().position().perp2()) - mcConvR_;
      float dZ=aConv.conversionVertex().position().z() - mcConvZ_;
      p2_convVtxdRVsRZ_ ->Fill (mcConvZ_,mcConvR_, dR );
      p2_convVtxdZVsRZ_ ->Fill (mcConvZ_,mcConvR_, dZ );




      h2_convVtxRrecVsTrue_ -> Fill (mcConvR_, sqrt(aConv.conversionVertex().position().perp2()) );


      h_zPVFromTracks_[match]->Fill ( aConv.zOfPrimaryVertexFromTracks() );
      h_dzPVFromTracks_[match]->Fill ( aConv.zOfPrimaryVertexFromTracks() - simPV_Z );
      h2_dzPVVsR_ ->Fill(mcConvR_, aConv.zOfPrimaryVertexFromTracks() - simPV_Z );
      p_dzPVVsR_ ->Fill(mcConvR_, aConv.zOfPrimaryVertexFromTracks() - simPV_Z );

      if ( phoIsInBarrel ) {
	h_convVtxdX_barrel_ ->Fill ( aConv.conversionVertex().position().x() - mcConvX_);
	h_convVtxdY_barrel_ ->Fill ( aConv.conversionVertex().position().y() - mcConvY_);
	h_convVtxdZ_barrel_ ->Fill ( aConv.conversionVertex().position().z() - mcConvZ_);
	h_convVtxdR_barrel_ ->Fill ( sqrt(aConv.conversionVertex().position().perp2()) - mcConvR_);

      }
      if ( phoIsInEndcap ) {
	h_convVtxdX_endcap_ ->Fill ( aConv.conversionVertex().position().x() - mcConvX_);
	h_convVtxdY_endcap_ ->Fill ( aConv.conversionVertex().position().y() - mcConvY_);
	h_convVtxdZ_endcap_ ->Fill ( aConv.conversionVertex().position().z() - mcConvZ_);
	h_convVtxdR_endcap_ ->Fill ( sqrt(aConv.conversionVertex().position().perp2()) - mcConvR_);

      }


    }

    ///////////  Quantities per track when tracks are associated
    for (unsigned int i=0; i<tracks.size(); i++) {
      //std::cout << " Loop over tracks  pt " << tracks[i]->pt() << std::endl;
      RefToBase<reco::Track> tfrb(aConv.tracks()[i] );
      itAss= myAss.find( tfrb.get() );

      nHitsVsEta_[match] ->Fill (mcEta_,   float(tracks[i]->numberOfValidHits()) );
      nHitsVsR_[match] ->Fill (mcConvR_,   float(tracks[i]->numberOfValidHits()) );
      p_nHitsVsEta_[match] ->Fill (mcEta_,   float(tracks[i]->numberOfValidHits()) -0.0001);
      p_nHitsVsR_[match] ->Fill (mcConvR_,   float(tracks[i]->numberOfValidHits()) -0.0001);
      h_tkChi2_[match] ->Fill (tracks[i]->normalizedChi2() );
      h_tkChi2Large_[match] ->Fill (tracks[i]->normalizedChi2() );
      h2_Chi2VsEta_[match] ->Fill(  mcEta_, tracks[i]->normalizedChi2() );
      h2_Chi2VsR_[match] ->Fill(  mcConvR_, tracks[i]->normalizedChi2() );
      p_Chi2VsEta_[match] ->Fill(  mcEta_, tracks[i]->normalizedChi2() );
      p_Chi2VsR_[match] ->Fill(  mcConvR_, tracks[i]->normalizedChi2() );
      double d0;
      if (valid_pvtx){
	d0 = - tracks[i]->dxy(the_pvtx.position());
      } else {
	d0 = tracks[i]->d0();
      }
      h_TkD0_[match]->Fill (d0* tracks[i]->charge() );
      h_nHitsBeforeVtx_[match]->Fill ( aConv.nHitsBeforeVtx().size()>1 ? aConv.nHitsBeforeVtx().at(i) : 0 );
      h_dlClosestHitToVtx_[match]->Fill ( aConv.dlClosestHitToVtx().size()>1 ? aConv.dlClosestHitToVtx().at(i).value() : 0 );
      h_dlClosestHitToVtxSig_[match]->Fill ( aConv.dlClosestHitToVtx().size()>1 ? aConv.dlClosestHitToVtx().at(i).value()/aConv.dlClosestHitToVtx().at(i).error() : 0 );


      if ( itAss == myAss.end()  ) continue;
      reco::Track refTrack= aConv.conversionVertex().refittedTracks()[i];

      float simPt = sqrt( ((*itAss).second)->momentum().perp2() );
      float recPt = refTrack.pt();
      float ptres= recPt - simPt ;
      //float pterror = aConv.tracks()[i]->ptError();
      float pterror = aConv.conversionVertex().refittedTracks()[i].ptError();
      h2_PtRecVsPtSim_[0]->Fill ( simPt, recPt);
      h_TkPtPull_[0] ->Fill(ptres/pterror);
      h2_TkPtPull_[0] ->Fill(mcEta_, ptres/pterror);

      if ( phoIsInBarrel ) {
	h_TkPtPull_[1] ->Fill(ptres/pterror);
	h2_PtRecVsPtSim_[1]->Fill ( simPt, recPt);
      }
      if ( phoIsInEndcap ) {
	h_TkPtPull_[2] ->Fill(ptres/pterror);
	h2_PtRecVsPtSim_[2]->Fill ( simPt, recPt);
      }
    } // end loop over track



  } // loop over reco conversions


  h_nConv_[0][0]->Fill (float(nRecConv_));
  h_nConv_[1][0]->Fill (float(nRecConvAss_));



}





void TkConvValidator::endJob() {


  std::string outputFileName = parameters_.getParameter<std::string>("OutputFileName");
  if ( ! isRunCentrally_ ) {
    dbe_->save(outputFileName);
  }

  edm::LogInfo("TkConvValidator") << "Analyzed " << nEvt_  << "\n";
  // std::cout  << "::endJob Analyzed " << nEvt_ << " events " << " with total " << nPho_ << " Photons " << "\n";
  //  std::cout  << "TkConvValidator::endJob Analyzed " << nEvt_ << " events " << "\n";

  return ;
}


math::XYZVector TkConvValidator::recalculateMomentumAtFittedVertex ( const MagneticField& mf, const TrackerGeometry& trackerGeom, const  edm::RefToBase<reco::Track>&   tk, const reco::Vertex& vtx) {

  math::XYZVector result;
  Surface::RotationType rot;
  auto scp = new SimpleCylinderBounds(  sqrt(vtx.position().perp2())-0.001f,
					sqrt(vtx.position().perp2())+0.001f,
					-fabs(vtx.position().z()),
					 fabs(vtx.position().z())
                                     );
  ReferenceCountingPointer<Cylinder>  theBarrel_(new Cylinder(Cylinder::computeRadius(*scp), Surface::PositionType(0,0,0), rot,scp));

  ReferenceCountingPointer<Disk>      theDisk_(new Disk( Surface::PositionType( 0, 0, vtx.position().z()), rot,
								   new SimpleDiskBounds( 0,  sqrt(vtx.position().perp2()), -0.001, 0.001) )
                                              );

  
  const TrajectoryStateOnSurface myTSOS = trajectoryStateTransform::innerStateOnSurface(*tk, trackerGeom, &mf);
  PropagatorWithMaterial propag( anyDirection, 0.000511, &mf );
  TrajectoryStateOnSurface  stateAtVtx;
  stateAtVtx = propag.propagate(myTSOS, *theBarrel_);
  if (!stateAtVtx.isValid() ) {
    stateAtVtx = propag.propagate(myTSOS, *theDisk_);
  }
  if (stateAtVtx.isValid()){
    return  math::XYZVector ( double(stateAtVtx.globalMomentum().x()), double(stateAtVtx.globalMomentum().y()), double(stateAtVtx.globalMomentum().z()));
  } else {
    return  math::XYZVector(0.,0.,0.);
  }



}


float TkConvValidator::phiNormalization(float & phi)
{
  //---Definitions
  const float PI    = 3.1415927;
  const float TWOPI = 2.0*PI;


  if(phi >  PI) {phi = phi - TWOPI;}
  if(phi < -PI) {phi = phi + TWOPI;}

  //  cout << " Float_t PHInormalization out " << PHI << endl;
  return phi;

}


float TkConvValidator::etaTransformation(  float EtaParticle , float Zvertex)  {

  //---Definitions
  const float PI    = 3.1415927;

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


