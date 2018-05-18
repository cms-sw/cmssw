// system include files
#include <memory>
#include <utility>


// user include files
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"


#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "Geometry/CaloTopology/interface/EcalTrigTowerConstituentsMap.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"


#include "RecoEcal/EgammaCoreTools/interface/EcalClusterFunctionBaseClass.h" 
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterFunctionFactory.h" 
#include "RecoEcal/EgammaCoreTools/plugins/EcalClusterCrackCorrection.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"
#include "CalibCalorimetry/EcalTPGTools/interface/EcalTPGScale.h"

#include "FastSimulation/Particle/interface/RawParticle.h"
#include "FastSimulation/BaseParticlePropagator/interface/BaseParticlePropagator.h"
#include "FastSimulation/Particle/interface/ParticleTable.h"
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"

#include "RecoEgamma/EgammaMCTools/interface/PhotonMCTruth.h"


#include "EcalEBTrigPrimAnalyzer.h"


using namespace edm;
class CaloSubdetectorGeometry;

EcalEBTrigPrimAnalyzer::EcalEBTrigPrimAnalyzer(const edm::ParameterSet&  iConfig)

{
  ecal_parts_.push_back("Barrel");
 
  outputFileName_ = iConfig.getParameter<std::string>("outFileName");
  recoContent_= iConfig.getParameter<bool>("RecoContentAvailable");
  recHits_= iConfig.getParameter<bool>("AnalyzeRecHits");
  analyzeElectrons_= iConfig.getParameter<bool>("AnalyzeElectrons");
  etCluTPThreshold_ = iConfig.getParameter<double>("etCluTPThreshold");
  debug_= iConfig.getParameter<bool>("Debug");
  rechits_labelEB_=consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("inputRecHitsEB"));
  primToken_=consumes<EcalEBTrigPrimDigiCollection>(iConfig.getParameter<edm::InputTag>("inputTP"));
  primCluToken_=consumes<EcalEBClusterTrigPrimDigiCollection>(iConfig.getParameter<edm::InputTag>("inputClusterTP"));
  tokenEBdigi_=consumes<EBDigiCollection>(iConfig.getParameter<edm::InputTag>("barrelEcalDigis"));
  pileupSummaryToken_=consumes<std::vector<PileupSummaryInfo> >(iConfig.getParameter<edm::InputTag>("bxInfos"));
  genPartToken_ = consumes<reco::GenParticleCollection>(iConfig.getParameter<edm::InputTag>("genParticles"));
  gsfElectrons_ = consumes<reco::GsfElectronCollection>(iConfig.getParameter<edm::InputTag>("eleCollection"));
  gedPhotons_ = consumes<reco::PhotonCollection>(iConfig.getParameter<edm::InputTag>("phoCollection"));
 
  g4_simTk_Token_  = consumes<edm::SimTrackContainer>(edm::InputTag("g4SimHits"));
  g4_simVtx_Token_ = consumes<edm::SimVertexContainer>(edm::InputTag("g4SimHits"));


  histfile_=new TFile(outputFileName_.c_str(),"RECREATE");
  //
  tree_ = new TTree("TPGtree","TPGtree");
  tree_->Branch("tpIphi",&tpIphi_,"tpIphi/I");
  tree_->Branch("tpIeta",&tpIeta_,"tpIeta/I");
  tree_->Branch("rhIphi",&rhIphi_,"rhIphi/I");
  tree_->Branch("rhIeta",&rhIeta_,"rhIeta/I");
  tree_->Branch("eRec",&eRec_,"eRec/F");
  tree_->Branch("tpgADC",&tpgADC_,"tpgADC/I");
  tree_->Branch("tpgGeV",&tpgGeV_,"tpgGeV/F");
  tree_->Branch("ttf",&ttf_,"ttf/I");
  tree_->Branch("fg",&fg_,"fg/I");

  treeCl_ = new TTree("TPGClusterTree", "TPGClusterTree");
  treeCl_->Branch("nCl",&nCl_,"nCl_/I");
  treeCl_->Branch("nXtals",&nXtals_,"nXtals_/I");
  treeCl_->Branch("eCl",&eCl_,"eCl_/F");
  treeCl_->Branch("etClInADC",&etClInADC_,"etClInADC_/I");
  treeCl_->Branch("etClInGeV",&etClInGeV_,"etClInGeV_/F");
  treeCl_->Branch("etaCl",&etaCl_,"etaCl_/F");
  treeCl_->Branch("phiCl",&phiCl_,"phiCl_/F");
  treeCl_->Branch("s_eCl",&s_eCl_,"s_eCl_/F");
  treeCl_->Branch("s_etCl",&s_etCl_,"s_etCl_/F");
  treeCl_->Branch("s_etaCl",&s_etaCl_,"s_etaCl_/F");
  treeCl_->Branch("s_phiCl",&s_phiCl_,"s_phiCl_/F");
  treeCl_->Branch("etCluFromRH",&etCluFromRH_,"etCluFromRH_/F");

  for (unsigned int i=0;i<ecal_parts_.size();++i) {
    char title[30];
    sprintf(title,"%s_Et",ecal_parts_[i].c_str());
    ecal_et_[i]=new TH1I(title,"Et",255,0,255);
    sprintf(title,"%s_ttf",ecal_parts_[i].c_str());
    ecal_tt_[i]=new TH1I(title,"TTF",10,0,10);
    sprintf(title,"%s_fgvb",ecal_parts_[i].c_str());
    ecal_fgvb_[i]=new TH1I(title,"FGVB",10,0,10);
  }



  nEvents_=0;

  hTPvsTow_eta_= new TH2F("TP_vs_Tow_eta","TP vs Tow eta ; #eta(tow); #eta(tp)",50,-2.5,2.5,50,-2.5,2.5);
  hAllTPperEvt_ = new TH1F("AllTPperEvt","TP per Event; N_{TP};  ", 100, 0., 20000.);
  hTPperEvt_ = new TH1F("TPperEvt","N_{TP} per Event; N_{TP};  ", 100, 0., 20000.);
  hTP_iphiVsieta_= new TH2F("TP_iphiVsieta","TP i#phi vs i#eta ; i#eta(tp); i#phi(tp)",10,70,80,10,340,350);
  hTP_iphiVsieta_fullrange_= new TH2F("TP_iphiVsieta_fullrange","TP i#phi vs i#eta ; i#eta(tp); i#phi(tp)",200,-100,100,350,0,350);
  h_bxNumber_ = new TH1F ("bunchCrossing", "Bunch Crossing", 16, -12, 3);
  h_pu_ = new TH1F ("PU", "PU", 100 , 0, 300);
  //
  hCluTPperEvt_ = new TH1F("CluTPperEvt","N_{TP} per Event; N_{TP};  ", 10, 0., 10.);
  h_nXtals_ = new TH1F("nXtals","Number of xTals in the cluster; N_{xtals}; ",   20, -0.5, 19.5);
  h_etCluTP_ = new TH1F("etCluTP","E_{T}(tp);E_{T}(tp) (GeV);Count",100,0,50); 
  //
  /*
  h_nClu_[0] = new TH1F("nClu0","Number of Clusters with E_{T}>0 GeV; N_{Clu};N_{ev}", 20, 0., 20.);
  h_nClu_[1] = new TH1F("nClu05","Number of Cluster with E_{T}>0.5 GeV; N_{Clu};N_{ev}", 20, 0., 20.);
  h_nClu_[2] = new TH1F("nClu1","Number of Clusters with E_{T}>1 GeV; N_{Clu};N_{ev}", 20, 0., 20.);
  h_nClu_[3] = new TH1F("nClu2","Number of Clusters with E_{T}>2 GeV; N_{Clu};N_{ev}", 20, 0., 20.);
  h_nClu_[4] = new TH1F("nClu3","Number of Clusters with E_{T}>3 GeV; N_{Clu};N_{ev}", 20, 0., 20.);
  */

  
  h_nClu_[0] = new TH1F("nClu0","Number of Clusters with E_{T}>0 GeV; N_{Clu};N_{ev}",   100, 0., 1000.);
  h_nClu_[1] = new TH1F("nClu05","Number of Cluster with E_{T}>0.5 GeV; N_{Clu};N_{ev}", 100, 0., 300.);
  h_nClu_[2] = new TH1F("nClu1","Number of Clusters with E_{T}>1 GeV; N_{Clu};N_{ev}",   100, 0., 300.);
  h_nClu_[3] = new TH1F("nClu2","Number of Clusters with E_{T}>2 GeV; N_{Clu};N_{ev}",   100, 0., 300.);
  h_nClu_[4] = new TH1F("nClu3","Number of Clusters with E_{T}>3 GeV; N_{Clu};N_{ev}",   100, 0., 300.);
  
  /*
  h_nClu_[0] = new TH1F("nClu0","Number of Clusters with E_{T}>0 GeV; N_{Clu};N_{ev}",   10, -0.5, 9.5);
  h_nClu_[1] = new TH1F("nClu05","Number of Cluster with E_{T}>0.5 GeV; N_{Clu};N_{ev}", 10, -0.5, 9.5);
  h_nClu_[2] = new TH1F("nClu1","Number of Clusters with E_{T}>1 GeV; N_{Clu};N_{ev}",   10, -0.5, 9.5);
  h_nClu_[3] = new TH1F("nClu2","Number of Clusters with E_{T}>2 GeV; N_{Clu};N_{ev}",   10, -0.5, 9.5);
  h_nClu_[4] = new TH1F("nClu3","Number of Clusters with E_{T}>3 GeV; N_{Clu};N_{ev}",   10, -0.5, 9.5);
  */

  h2_recEle_vs_Gen_size_= new TH2F("recEle_vs_Gen_size","gsfEle size vs gen ele size; N_{gen}; N_{reco}",15,-0.5,14.5,15,-0.5,14.5); 
  h2_cluTP_vs_Gen_size_= new TH2F("cluTP_vs_Gen_size","cluTP size vs gen ele size; N_{gen}; N_{cluTP}",15,-0.5,14.5,15,-0.5,14.5); 
  h2_cluRH_vs_Gen_size_= new TH2F("cluRH_vs_Gen_size","cluRH size vs gen ele size; N_{gen}; N_{cluRH}",15,-0.5,14.5,15,-0.5,14.5); 



  // matching cluTP with gen electrons 
  h_deltaR_cluTPGen_= new TH1F("deltaR_cluTPGen","#DR(cluTP-gen); #DR(cluTP-gen);N_{ev}", 100, 0., 0.15);
  h_dEta_cluTP_gen_ = new TH1F("dEta_cluTP_gen","Entries; #D#eta(cluTP-gen);  ", 50,-0.05,0.05);
  h_dPhi_cluTP_gen_ = new TH1F("dPhi_cluTP_gen","Entries; #D#phi(cluTP-gen);  ", 50,-0.1,0.1);
  h_cluTPEtoverGenEt_ = new TH1F("CluTPEt_over_GenEt","Et(CluTP/gen); E_{T}(tp)/E_{T}(gen); Counts",200,-0.5,1.5);

  //matching cluTP with cluRH
  h_deltaR_cluTPcluRH_= new TH1F("deltaR_cluTPcluRH","#DR(cluTP-cluRH); #DR(cluTP-cluRH);N_{ev}", 200, 0., 0.01);
  h_dEta_cluTP_cluRH_ = new TH1F("dEta_cluTP_cluRH","Entries; #D#eta(cluTP-cluRH);  ", 50,-0.01,0.01);
  h_dPhi_cluTP_cluRH_ = new TH1F("dPhi_cluTP_cluRH","Entries; #D#phi(cluTP-cluRH);  ", 50,-0.01,0.01);
  hCluTPoverRechit_= new TH1F("CluTP_over_RecHit","Et(CluTP/rechit); E_{T}(tp)/E_{T}(rh); Counts",200,-0.5,1.5);
  hCluTPvsRechit_= new TH2F("CluTP_vs_RecHit","CluTP vs rechit Et;E_{T}(rh) (GeV);E_{T}(tp) (GeV)",100,-5,50,100,-5,50);
  h_fBrem_truth_ = new TH1F("fBrem_truth","fBrem; fBrem; Counts",100,0.,1.);


  // macthing cluRH with gen electrons
  h_deltaR_cluRHGen_= new TH1F("deltaR_cluRHGen","#DR(cluRH-gen); #DR(cluRH-gen);N_{ev}", 100, 0., 0.15);
  h_dEta_cluRH_gen_ = new TH1F("dEta_cluRH_gen","Entries; #D#eta(cluRH-gen);  ", 50,-0.05,0.05);
  h_dPhi_cluRH_gen_ = new TH1F("dPhi_cluRH_gen","Entries; #D#phi(cluRH-gen);  ", 50,-0.1,0.1);
  h_cluRHEt_ = new TH1F("CluRHEt","Et(CluRH); E_{T}(cluRH); Counts",100,0.,50.);
  h_cluRHEtoverGenEt_ = new TH1F("CluRHEt_over_GenEt","Et(CluRH/gen); E_{T}(rh)/E_{T}(gen); Counts",200,-0.5,1.5);

  
  // matching reco ele with gen ele
  h_elePtRecoOverPtTrue_ = new TH1F("elePtRecoOverPtTrue","Pt(gsfEle/gen); E_{T}(gsfEle)/E_{T}(gen); Counts",200,-0.5,1.5);
  h_corrEleEtRecoOverPtTrue_ = new TH1F("corrEleEtRecoOverPtTrue","corrEt(gsfEle/gen); E_{T}_{corr}(gsfEle)/E_{T}(gen); Counts",200,-0.5,1.5);
  h_uncorrEleEtRecoOverPtTrue_ = new TH1F("uncorrEleEtRecoOverPtTrue","uncorrEt(gsfEle/gen); E_{T}_{uncorr}(gsfEle)/E_{T}(gen); Counts",200,-0.5,1.5);
  h_dPhi_5x5SC_gen_    = new TH1F("dPhi_5by5SC_gen","Entries; #D#phi(5x5-gen);  ", 50,-0.1,0.1);
  h_5x5SCOverPtTrue_   = new TH1F("sc5by5OverPtTrue","Pt(5x5/gen); E_{T}(5x5)/E_{T}(gen); Counts",200,-0.5,1.5);
  h_3x3SCOverPtTrue_   = new TH1F("sc3by3OverPtTrue","Pt(3x3/gen); E_{T}(3x3)/E_{T}(gen); Counts",200,-0.5,1.5);
  h_dEta_gsfEle_gen_   = new TH1F("dEta_gsfEle_gen","Entries; #D#eta(gsfEle-gen);  ", 50,-0.1,0.1);
  h_dPhi_gsfEle_gen_   = new TH1F("dPhi_gsfEle_gen","Entries; #D#phi(gsfEle-gen);  ", 50,-0.1,0.1);
  h_deltaR_recoEleGen_ = new TH1F("deltaR_recoEleGen","#DR(ele-gen); #DR(ele-gen);N_{ev}", 200, 0., 0.1); 
  

  if (recHits_) {
    hTPvsTow_ieta_= new TH2F("TP_vs_Tow_ieta","TP vs Tow ieta ; i#eta(tow); i#eta(tp)",200,-100,100,200,-100,100);

    hTPvsRechit_= new TH2F("TP_vs_RecHit","TP vs rechit Et;E_{T}(rh) (GeV);E_{T}(tp) (GeV)",100,-5,50,100,-5,50);
    hDeltaEt_ = new TH1F("DeltaEt","[Et(rh)-Et(TP)]/Et(rh); [E_{T}(rh)-E_{T}(tp)]/E_{T}(rh); Counts",200,-1,1);
    hTPoverRechit_= new TH1F("TP_over_RecHit","Et(TP/rechit); E_{T}(tp)/E_{T}(rh); Counts",200,0,2);
    hRechitEt_= new TH1F("RecHitEt","E_{T};E_{T}(rh) (GeV);Counts",100,0,50);
    hTPEt_= new TH1F("TPEt","E_{T}{tp);E_{T}(rh) (GeV);Count",100,0,50);
    hRatioEt_ = new TH1F("RatioTPoverRH","Et",100,0,50);
    hAllRechitEt_= new TH1F("AllRecHit","Et",100,0,50);

    hRH_iphiVsieta_= new TH2F("RH_iphiVsieta","RH i#phi vs i#eta ; i#eta(rh); i#phi(rh)",10,70,80,10,340,350);
    hRH_iphiVsieta_fullrange_= new TH2F("RH_iphiVsieta_fullrange","RH i#phi vs i#eta ; i#eta(rh); i#phi(rh)",200,-100,100,350,0,350);

    
  }
}


EcalEBTrigPrimAnalyzer::~EcalEBTrigPrimAnalyzer()
{

   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

  histfile_->Write();
  histfile_->Close();
  thePhotonMCTruthFinder_.reset();
}

void EcalEBTrigPrimAnalyzer::beginRun(edm::Run const& run, edm::EventSetup const& es)
{


}


void EcalEBTrigPrimAnalyzer::init(const edm::EventSetup & iSetup) {
  iSetup.get<IdealGeometryRecord>().get(eTTmap_);
  nTotTP_=0;
  nTotCluTP_=0;
  nTotCluRH_=0;
  thePhotonMCTruthFinder_.reset(new PhotonMCTruthFinder() );   
}

//
// member functions
//

// ------------ method called to analyze the data  ------------
void
EcalEBTrigPrimAnalyzer::analyze(const edm::Event& iEvent, const  edm::EventSetup & iSetup)
{
  using namespace edm;
  using namespace std;
  nEvents_++;

  if ( nEvents_==1) this->init(iSetup);

 
  thePhotonMCTruthFinder_->clear();
 

  //get simtrack info
  std::vector<SimTrack> theSimTracks;
  std::vector<SimVertex> theSimVertices;
  edm::Handle<SimTrackContainer> SimTk;
  edm::Handle<SimVertexContainer> SimVtx;
  iEvent.getByToken(g4_simTk_Token_, SimTk);
  iEvent.getByToken(g4_simVtx_Token_, SimVtx);
  theSimTracks.insert(theSimTracks.end(),SimTk->begin(),SimTk->end());
  theSimVertices.insert(theSimVertices.end(),SimVtx->begin(),SimVtx->end());
  //
  std::vector<PhotonMCTruth> mcEle=thePhotonMCTruthFinder_->find (theSimTracks, theSimVertices);


  // Get input TP
  edm::Handle<EcalEBTrigPrimDigiCollection> tp;
  iEvent.getByToken(primToken_,tp);
  //
  edm::Handle<EcalEBClusterTrigPrimDigiCollection> tpClu;
  iEvent.getByToken(primCluToken_,tpClu);
  bool tpCluHandleIsValid=false;
  if ( tpClu.isValid() )
    tpCluHandleIsValid=true;


  for (unsigned int i=0;i<tp.product()->size();i++) {
    EcalEBTriggerPrimitiveDigi d=(*(tp.product()))[i];
    int subdet=0;
    if (subdet==0) {
      ecal_et_[subdet]->Fill(d.encodedEt());
    }
  }



  // Get GenParticles
  edm::Handle<reco::GenParticleCollection> gp;
  iEvent.getByToken(genPartToken_,gp);
  isGenParticleValid_=false;
  if ( gp.isValid() ) 
    isGenParticleValid_=true;
  if (isGenParticleValid_) std::cout << " Gen Particle size " << gp->size();

  //
  /*
  edm::Handle<EBDigiCollection> barrelDigiHandle;
  const EBDigiCollection *ebdigi=NULL;
  iEvent.getByToken(tokenEBdigi_,barrelDigiHandle);
  ebdigi=barrelDigiHandle.product();
  */
  // Get PUinfos
   edm::Handle<std::vector<PileupSummaryInfo> > puInfo;
   iEvent.getByToken(pileupSummaryToken_, puInfo);
   
  

   // get the  RecHits
  edm::Handle<EcalRecHitCollection> rechit_EB_col;
  if ( recHits_ ) {
    iEvent.getByToken(rechits_labelEB_,rechit_EB_col);
  }

  // get the geometry
  edm::ESHandle<CaloSubdetectorGeometry> theBarrelGeometry_handle;
  iSetup.get<EcalBarrelGeometryRecord>().get("EcalBarrel",theBarrelGeometry_handle);
  const CaloSubdetectorGeometry *theBarrelGeometry;
  theBarrelGeometry = &(*theBarrelGeometry_handle);
  
  const CaloTopology *topology = 0;
  if ( recoContent_ ) { 
    edm::ESHandle<CaloTopology> theCaloTopo;
    iSetup.get<CaloTopologyRecord>().get(theCaloTopo);
    topology = theCaloTopo.product();
  }


  if (debug_) { std::cout << " TP analyzer  =================> Treating event  "<<iEvent.id()  <<  " Number of TPs " <<  tp.product()->size() <<  std::endl;
    if ( recHits_ ) std::cout << " Number of EB rechits "<<  rechit_EB_col.product()->size() << std::endl;
  }


  edm::Handle<reco::GsfElectronCollection> gsfElectronsH;
  edm::Handle<reco::PhotonCollection> gedPhotonH; 
  if ( recoContent_) {
    // get the gsfElectrons
    iEvent.getByToken(gsfElectrons_,gsfElectronsH);
    std::cout << " gsfElectron size " << gsfElectronsH->size() ;
    // get the gedPhotons
    iEvent.getByToken(gedPhotons_,gedPhotonH);
    std::cout << " gedPhotons size " << gedPhotonH->size() ;
  }
  
  //std::cout << " PU info size " << puInfo->size() << std::endl;
  for (auto const& v : *puInfo) {
    int bx = v.getBunchCrossing();
    h_bxNumber_->Fill(float(bx));
    //std::cout << " bx " << bx << std::endl;
    if (bx == 0) 
      h_pu_->Fill(float(v.getPU_NumInteractions()));
  }


  hAllTPperEvt_->Fill(float(tp.product()->size())); 
  
  //if ( iEvent.id().event() != 648) return;

  //EcalEBTPGScale ecalScale ;
  EcalTPGScale ecalScale ;
  ecalScale.setEventSetup(iSetup) ;
  
  
  //  for(unsigned int iDigi = 0; iDigi < ebdigi->size() ; ++iDigi) {
  // EBDataFrame myFrame((*ebdigi)[iDigi]);  
  // const EBDetId & myId = myFrame.id();
     
  int nTP=0;  
  for (unsigned int i=0;i<tp.product()->size();i++) {
    EcalEBTriggerPrimitiveDigi d=(*(tp.product()))[i];
    const EBDetId TPid= d.id();
    // if ( myId != TPid ) continue;
    
    
    /*    
	  int index=getIndex(ebdigi,coarser);
	  std::cout << " Same xTal " << myId << " " << TPid << " coarser " << coarser << " index " << index << std::endl;
	  double Et = ecalScale.getTPGInGeV(d.encodedEt(), coarser) ; 
    */
    //this works if the energy is compressed into 8 bits float Et=d.compressedEt()/2.; // 2ADC counts/GeV
    float Et=d.encodedEt()/8.;    // 8 ADCcounts/GeV
    if ( Et<= 0 ) continue;
    nTP++;
    

    //    std::cout << " TP digi size " << d.size() << std::endl;
    for (int iBx=0;iBx<d.size();iBx++) {
      //std::cout << " TP samples " << d.sample(iBx) << std::endl; 

    }

    //      EcalTrigTowerDetId coarser=(*eTTmap_).towerOf(myId);
    // does not work float etaTow =  theBarrelGeometry->getGeometry(coarser)->getPosition().theta();
    // float etaTP =  theBarrelGeometry->getGeometry(TPid)->getPosition().eta();
    // does not work hTPvsTow_eta_->Fill ( etaTow,  etaTP );
    //      hTPvsTow_ieta_->Fill ( coarser.ieta(),  TPid.ieta() );
    
    
    tpIphi_ = TPid.iphi() ;
    tpIeta_ = TPid.ieta() ;
    tpgADC_ = d.encodedEt();
    tpgGeV_ = Et ;
    
    hTP_iphiVsieta_->Fill(TPid.ieta(), TPid.iphi(), Et);
    hTP_iphiVsieta_fullrange_->Fill(TPid.ieta(), TPid.iphi(), Et);
    
    
    if ( recHits_ ) {      
      for (unsigned int j=0;j<rechit_EB_col.product()->size();j++) {
	const EBDetId & myid1=(*rechit_EB_col.product())[j].id();
	float theta =  theBarrelGeometry->getGeometry(myid1)->getPosition().theta();
	float rhEt=((*rechit_EB_col.product())[j].energy())*sin(theta);
	if ( myid1 == TPid ) {
	  if (debug_) std::cout << " Analyzer same cristal " << myid1 << " " << TPid << std::endl;
	  //	  if ( rhEt < 1.5 && Et > 10 )  {
	  // std::cout << " TP analyzer  =================> Treating event  "<<iEvent.id()<< ", Number of EB rechits "<<  rechit_EB_col.product()->size() <<  " Number of TPs " <<  tp.product()->size() <<  std::endl;
	  //std::cout << " TP compressed et " << d.encodedEt()  << " Et in GeV  " <<  Et << " RH Et " << rhEt << " Et/rhEt " << Et/rhEt << std::endl;          
	  //} 
	  
	  //std::cout << " TP out " <<  d << std::endl;
	  
	  //	  for (int isam=0;isam< d.size();++isam) {
	  // std::cout << " d[isam].raw() "  <<  d[isam].raw() << std::endl;
	  //}
	  
	  rhIphi_ =  myid1.iphi() ;
	  rhIeta_ =  myid1.ieta() ;
	  hRH_iphiVsieta_->Fill(myid1.ieta(), myid1.iphi(), rhEt);
	  hRH_iphiVsieta_fullrange_->Fill(myid1.ieta(), myid1.iphi(), rhEt);
	  
	  hTPvsRechit_->Fill(rhEt,Et);
	  hTPoverRechit_->Fill(Et/rhEt);
	  hDeltaEt_ ->Fill ((rhEt-Et)/rhEt);
	  if (debug_) std::cout << " TP compressed et " << d.encodedEt()  << " Et in GeV  " <<  Et << " RH Et " << rhEt << " Et/rhEt " << Et/rhEt << std::endl;          
	  hRechitEt_->Fill(rhEt);
	  hTPEt_->Fill(Et);
	  if ( rhEt < 1000000) eRec_ = rhEt;
	  tree_->Fill() ;
	}
	
      }  // end loop of recHits
    }  // if recHits
    
  }   // end loop over TP collection
  
  //  } // end loop over digi collection

  hTPperEvt_->Fill(float(nTP));
  if ( nTP > 0 ) nTotTP_++;
  
  if ( recHits_) {  
    hRatioEt_->Divide( hTPEt_, hRechitEt_);
    for (unsigned int j=0;j<rechit_EB_col.product()->size();j++) {
      const EBDetId & myid1=(*rechit_EB_col.product())[j].id();
      float theta =  theBarrelGeometry->getGeometry(myid1)->getPosition().theta();
      float rhEt=((*rechit_EB_col.product())[j].energy())*sin(theta);
      if ( rhEt >0 ) 
	hAllRechitEt_ ->Fill(rhEt);
    }
  }



  //  float Etsum=0;
  std::vector<SimpleCluster> cluCollection;
  std::vector<SimpleCaloHit2> myHits;
  if ( recHits_ ) {    

    myHits.clear();
    for (unsigned int j=0;j<rechit_EB_col.product()->size();j++) {
      const EBDetId & rhID=(*rechit_EB_col.product())[j].detid();
      float eta =  theBarrelGeometry->getGeometry(rhID)->getPosition().eta();
      float energy= (*rechit_EB_col.product())[j].energy();
      float et= energy/cosh(eta);
      if (energy <=0) continue;
      if (debug_) std::cout << " my Hits energy " << energy << " et " << et << std::endl;
      SimpleCaloHit2 hit(et);
      hit.setPosition(GlobalVector( theBarrelGeometry->getGeometry(rhID)->getPosition().x(), 
				    theBarrelGeometry->getGeometry(rhID)->getPosition().y(), 
				    theBarrelGeometry->getGeometry(rhID)->getPosition().z()));



      hit.setId(rhID);
      myHits.push_back(hit);	  


    }
    if ( debug_) std::cout << " myHits collection size " << myHits.size() << std::endl;
    for (unsigned int iRH=0;iRH<myHits.size();++iRH) {
      if (debug_) std::cout << " RecHit Et " << myHits[iRH].et() << std::endl;
    }    


    /*
      for  (int ii=0;ii<nXtals_;ii++) {
      const EBDetId & xtalID =  d.crystalsInCluster()[ii];
      
      for (unsigned int j=0;j<rechit_EB_col.product()->size();j++) {
      // check that the recHit correspond to the crystal in the cluster TP
      const EBDetId & rhID=(*rechit_EB_col.product())[j].id();
      float theta =  theBarrelGeometry->getGeometry(rhID)->getPosition().theta();
      if ( rhID == xtalID ) {
      // std::cout << " RH id " << rhID << std::endl;
      Etsum+=((*rechit_EB_col.product())[j].energy())*sin(theta);
      }
      }
      }
    */
    
    // if rechits are available  
    cluCollection = makeCluster ( myHits, 3, 3  );
    
  }  // if (recHIts_)
  
  if (debug_) std::cout << " cluCollection from RH " << cluCollection.size()  << std::endl;  
  
  
  // loop over cluster TP
  int nClu=0;
  int nClu_Etgt_05=0;
  int nClu_Etgt_1=0;
  int nClu_Etgt_2=0;
  int nClu_Etgt_3=0;



  if (tpCluHandleIsValid ) {
 
    nCl_ = tpClu.product()->size();
    int nCluTPPerEvt=0;
    for (unsigned int i=0;i<tpClu.product()->size();i++) {
      EcalEBClusterTriggerPrimitiveDigi d=(*(tpClu.product()))[i];
      const EBDetId TPid= d.id();
      float etaCluTP =  theBarrelGeometry->getGeometry(TPid)->getPosition().eta();
      float phiCluTP =  theBarrelGeometry->getGeometry(TPid)->getPosition().phi();
      //std::cout << " cluTP eta " << d.eta() << " " << d.phi() << std::endl;
      nXtals_= d.crystalsInCluster().size();
      nCluTPPerEvt++;
      nTotCluTP_++;

      /*
      std::cout << " Number of xtals in clu " << nXtals_ << std::endl;
      std::cout << " List of IDs: " << std::endl;
      for  (int ii=0;ii<nXtals_;ii++) {
	std::cout <<  d.crystalsInCluster()[ii] << std::endl;
      }
      */


      etClInADC_ = d.encodedEt();
      etClInGeV_ = d.encodedEt()/8.;
      etaCl_ = d.eta();
      phiCl_ = d.phi();
      s_eCl_ = 0.;
      s_etCl_ = 0.;
      s_etaCl_ = etaCluTP;
      s_phiCl_ = phiCluTP;
 
      if (etClInGeV_>0)   nClu++;
      if (etClInGeV_>0.5) nClu_Etgt_05++;
      if (etClInGeV_>1) nClu_Etgt_1++;
      if (etClInGeV_>2) nClu_Etgt_2++;
      if (etClInGeV_>3) nClu_Etgt_3++;

      if (etClInGeV_< etCluTPThreshold_ ) continue;
      h_nXtals_->Fill(float(nXtals_));
      h_etCluTP_->Fill ( etClInGeV_);


      if (debug_) std::cout << " cluTP Et before matching " << etClInGeV_ << " eta " << etaCl_ << " phi " << phiCl_ << std::endl;  


      if (recHits_) {
	// match with the cluster made from recHits
        float dR=99999;
	
	int iMatch=-1;
	for (unsigned int iClu=0;iClu<cluCollection.size();iClu++) {
	  //	  std::cout << " Clu Et " << cluCollection[iClu].et() << " eta " <<  cluCollection[iClu].eta() << " phi " <<  cluCollection[iClu].phi() << std::endl;
	  
	  double dphi =  d.phi() -  cluCollection[iClu].phi();
	  if ( std::abs(dphi)>CLHEP::pi) { dphi = dphi<0? (CLHEP::twopi) + dphi : dphi - CLHEP::twopi;}
	  double  deltaR2  = ( d.eta()-  cluCollection[iClu].eta() ) *  ( d.eta() -  cluCollection[iClu].eta() ) + dphi*dphi;
	 
	  
	  if ( deltaR2 < dR) {
            dR=deltaR2;
	    iMatch=iClu;
	  }
       	}

	if ( iMatch >=0  && sqrt(dR) < 0.1 ) {
	  h_dEta_cluTP_cluRH_ -> Fill( etaCl_ -  cluCollection[iMatch].eta());
          float matchDPhi = d.phi() -  cluCollection[iMatch].phi();
	  if ( std::abs(matchDPhi)>CLHEP::pi) { matchDPhi = matchDPhi<0? (CLHEP::twopi) + matchDPhi : matchDPhi - CLHEP::twopi;}
	  h_dPhi_cluTP_cluRH_ -> Fill(matchDPhi);
          double matchDR = sqrt( ( d.eta()-  cluCollection[iMatch].eta() ) *  ( d.eta() -  cluCollection[iMatch].eta() ) + matchDPhi*matchDPhi );
	  h_deltaR_cluTPcluRH_->Fill ( matchDR );	    
	  hCluTPoverRechit_->Fill(etClInGeV_/ cluCollection[iMatch].et());
	  hCluTPvsRechit_->Fill( cluCollection[iMatch].et(),etClInGeV_);
	  etCluFromRH_=  cluCollection[iMatch].et();	
	}
	
	
      }


      // matching the cluTP with the genParticle
      reco::Candidate::PolarLorentzVector* cluTP_p4;
      reco::Candidate::PolarLorentzVector* trueEle_p4=0;
      reco::Candidate::PolarLorentzVector* matchEle_p4=0;
      reco::GenParticleCollection::const_iterator gpIter ;
      reco::GenParticleCollection::const_iterator gpIterMatch ;
      cluTP_p4 = new reco::Candidate::PolarLorentzVector;
      cluTP_p4->SetPt(etClInGeV_);
      cluTP_p4->SetEta(etaCl_);
      cluTP_p4->SetPhi(phiCl_);
      cluTP_p4->SetM(0.);

      double dR=9999.; 
      if ( isGenParticleValid_ )  {
	BaseParticlePropagator prop;

	for ( gpIter=gp->begin() ; gpIter!=gp->end() ; gpIter++ ) {
	  if ( gpIter->pdgId() == 22 ) {
	    if (abs(gpIter->eta()) < 1.5 && gpIter->pt()>10 ) {

	      
	      const reco::GenParticle* myGp=&(*gpIter);
              RawParticle trueEle(myGp->p4());
	      trueEle.setVertex(myGp->vertex().x(), myGp->vertex().y(), myGp->vertex().z(), 0.);
	      if (debug_) std::cout << " Before propagation gen ID " << gpIter->pdgId() << " p " << myGp->p() << " vertex " << myGp->vertex().x() << " " << myGp->vertex().y() << " " << myGp->vertex().z() << std::endl;
	
              float ch=gpIter->charge();
	      if (debug_) std::cout << " my charge " << ch << std::endl;
	      trueEle.setCharge(ch);
	      prop = BaseParticlePropagator(trueEle,0.,0.,4.);
	      BaseParticlePropagator start(prop);
	      prop.propagateToEcalEntrance();
              if ( prop.getSuccess() !=0) {
                trueEle_p4 = new reco::Candidate::PolarLorentzVector(prop.E()*sin(prop.vertex().theta()),  prop.vertex().eta(), prop.vertex().phi(), 0.);
                float pstart = sqrt ( start.momentum().x()*start.momentum().x() + start.momentum().y()*start.momentum().y() + start.momentum().z()*start.momentum().z()  );
		if (debug_) std::cout << " starting state   "  << " vertex " << start.vertex().x() << " " << start.vertex().y() << " " << start.vertex().z() << " momentum " << pstart << std::endl;
                float pprop = sqrt ( prop.momentum().x()*prop.momentum().x() + prop.momentum().y()*prop.momentum().y() + prop.momentum().z()*prop.momentum().z()  );
		 if (debug_)std::cout << " After propagation "  << " vertex " << prop.vertex().x() << " " << prop.vertex().y() << " " << prop.vertex().z() << " momentum " << pprop << std::endl;
		if (debug_) std::cout << " Momentum difference " << myGp->p()-pprop << std::endl;
	      } else {
		if (debug_) std::cout << " Prop fails " << std::endl;
                continue;
	      }


	      if ( reco::deltaR(*cluTP_p4, *trueEle_p4) < dR &&  (fabs(cluTP_p4->pt() - trueEle_p4->pt()) < 0.5*trueEle_p4->pt() ) ) {
		dR=  reco::deltaR(*cluTP_p4, *trueEle_p4);
		matchEle_p4 =  trueEle_p4;
                gpIterMatch=gpIter;
	      }
	    }
	  }
	}
	

	if (  matchEle_p4 !=0 && dR<0.1) {

	  	
	  //float matchDPhi =  reco::deltaPhi (cluTP_p4->phi(), matchEle_p4->phi());
	  float matchDPhi = normalizedPhi ( cluTP_p4->phi()- matchEle_p4->phi());
	  //	if ( std::abs(matchDPhi)>CLHEP::pi) { matchDPhi = matchDPhi<0? (CLHEP::twopi) + matchDPhi : matchDPhi - CLHEP::twopi;}
	  // float gpEta = etaTransformation( matchEle_p4->eta(), gpIterMatch->vz() );
     	
	  h_cluTPEtoverGenEt_->Fill ( cluTP_p4->pt()/ matchEle_p4->pt()); 
	  h_dEta_cluTP_gen_ -> Fill(  cluTP_p4->eta()  - matchEle_p4->eta());
	  h_dPhi_cluTP_gen_ -> Fill(matchDPhi);
	  h_deltaR_cluTPGen_ -> Fill (dR);

	}
	

      }  // if ( isGenParticleValid_ )

      /*
      if ( isGenParticleValid_ )  {

        float dR=9999;
	reco::GenParticleCollection::const_iterator gpIter ;
	float gpEta=-9999.;
	const reco::GenParticle* gpMatch=0; 
	for ( gpIter=gp->begin() ; gpIter!=gp->end() ; gpIter++ ) {
	  if ( gpIter->pdgId() == 11 ||  gpIter->pdgId() == -11 ) {
	    if (abs(gpIter->eta()) < 1.5 && gpIter->pt()>5 ) {
	      const reco::GenParticle* myGp=&(*gpIter); 
	      gpEta = etaTransformation(gpIter->eta(), gpIter->vz() );

              double dphi =  phiCl_ - gpIter->phi();
              if ( std::abs(dphi)>CLHEP::pi) { dphi = dphi<0? (CLHEP::twopi) + dphi : dphi - CLHEP::twopi;}
              double  deltaR2  = ( etaCl_ - gpEta ) *  ( etaCl_ - gpEta ) + dphi*dphi;

	      if ( deltaR2 < dR ) {
		dR = deltaR2;
		gpMatch=myGp;
	      } 
	    }
	  }
	}
	if (  gpMatch !=0 && sqrt(dR)<0.1) {
	  std::cout << "cluTP Et " << etClInGeV_ << " gen ele mom " << gpMatch->p() << " pt " << gpMatch->pt() << " charge " << gpMatch->charge() <<  " eta " << gpMatch->eta() << " phi " << gpMatch->phi() << std::endl;
	  h_cluTPEtoverGenEt_->Fill ( etClInGeV_/ gpMatch->pt()); 
          float matchDPhi = phiCl_ - gpMatch->phi();
	  if ( std::abs(matchDPhi)>CLHEP::pi) { matchDPhi = matchDPhi<0? (CLHEP::twopi) + matchDPhi : matchDPhi - CLHEP::twopi;}
	  float matchEta = etaTransformation(gpMatch->eta(), gpMatch->vz() );
          double matchDR = sqrt ( ( etaCl_ - matchEta ) *  ( etaCl_ - matchEta ) + matchDPhi*matchDPhi );
	  h_dEta_cluTP_gen_ -> Fill( etaCl_ - matchEta);
	  h_dPhi_cluTP_gen_ -> Fill(matchDPhi);
	  h_deltaR_cluTPGen_ -> Fill (matchDR);



	}     
      }  // if genParticleValid 

      */

      treeCl_->Fill();    

    }
    std::cout << " nCluTPPerEvt " << nCluTPPerEvt << std::endl;
    hCluTPperEvt_ ->Fill(float(nCluTPPerEvt));

  }

  
  if ( nClu>0)         h_nClu_[0] ->Fill(float(nClu));
  if ( nClu_Etgt_05>0) h_nClu_[1] ->Fill(float(nClu_Etgt_05));
  if ( nClu_Etgt_1>0)  h_nClu_[2] ->Fill(float(nClu_Etgt_1));
  if ( nClu_Etgt_2>0)  h_nClu_[3] ->Fill(float(nClu_Etgt_2));
  if ( nClu_Etgt_3>0)  h_nClu_[4] ->Fill(float(nClu_Etgt_3));
  
  
  // match the clusters made of RH with gen
  float nGenInEB=0;    
  if ( isGenParticleValid_ )  {
    reco::GenParticleCollection::const_iterator gpIter ;    
    reco::GenParticleCollection::const_iterator gpIterMatch ;    
    reco::Candidate::PolarLorentzVector* cluRH_p4;
    reco::Candidate::PolarLorentzVector* trueEle_p4=0;
    reco::Candidate::PolarLorentzVector* matchEle_p4=0;
    BaseParticlePropagator prop;
    
    float dR=9999;
    nGenInEB=0;    
    
    
    for ( std::vector<PhotonMCTruth>::const_iterator imcEle=mcEle.begin(); imcEle!=mcEle.end(); imcEle++ ) {
      
      if (   abs(imcEle->fourMomentum().pseudoRapidity()) >= 1.5 || imcEle->fourMomentum().et() < 10 ) continue;
      //      if ( imcEle->hasBrem() !=0) continue;
      
      if (debug_) std::cout << " imcEle pt " << imcEle->fourMomentum().et() << " E " << imcEle->fourMomentum().e() << std::endl;
      bool isTheSame=false;
      for ( gpIter=gp->begin() ; gpIter!=gp->end() ; gpIter++ ) {
	//	if ( abs(gpIter->pdgId()) != 11 ) continue;
	// float ch= gpIter->pdgId()/11;
	if ( abs(gpIter->pdgId()) != 22 ) continue;
	float ch= 0.;

        

	if (debug_) std::cout << "gpIter charge in the loop " << ch <<  " and Pt " << gpIter->pt() << std::endl;

	isTheSame=false;
	
	float myMCPhi=imcEle->fourMomentum().phi();
	myMCPhi= normalizedPhi(myMCPhi);

	float gpPhi=gpIter->phi();
	gpPhi= normalizedPhi(gpPhi);
	double dPhi = myMCPhi - gpPhi;
	std::cout << " myMCPhi " << myMCPhi << " gpPhi " << gpPhi << std::endl;
	double dEta = fabs(imcEle->fourMomentum().pseudoRapidity() - gpIter->eta());
	double dPt = fabs(imcEle->fourMomentum().et() - gpIter->pt() );
	
	if ( dEta <= 0.0001 && dPhi <= 0.0001 && dPt <= 0.0001)  {
	  isTheSame=true;
          gpIterMatch=gpIter;
	  break;
	}

      }

      if ( ! isTheSame ) continue;
      
      //  std::cout << " mctruth eloss  " << imcEle->eloss()[0] << " fBrem " << imcEle->eloss()[0]/imcEle->fourMomentum().e() << std::endl;     
      ///h_fBrem_truth_->Fill( imcEle->eloss()[0]/imcEle->fourMomentum().e() );
      nGenInEB++;	    
      reco::Candidate::LorentzVector eleP4(imcEle->fourMomentum().px(),imcEle->fourMomentum().py(), imcEle->fourMomentum().pz(), imcEle->fourMomentum().e()); 
      RawParticle trueEle(eleP4);
      trueEle.setVertex(imcEle->primaryVertex().x(), imcEle->primaryVertex().y(), imcEle->primaryVertex().z(), 0.);
      //float ch= gpIterMatch->pdgId()/11;
      float ch=0;
      std::cout << " my charge " << ch << std::endl;
      trueEle.setCharge(ch);
      prop = BaseParticlePropagator(trueEle,0.,0.,4.);
      prop.propagateToEcalEntrance();
      
      if ( prop.getSuccess() !=0) {
	trueEle_p4 = new reco::Candidate::PolarLorentzVector(prop.E()*sin(prop.vertex().theta()),  prop.vertex().eta(), prop.vertex().phi(), 0.);
	float pprop = sqrt ( prop.momentum().x()*prop.momentum().x() + prop.momentum().y()*prop.momentum().y() + prop.momentum().z()*prop.momentum().z()  );
	if (debug_) std::cout << " After propagation "  << " vertex " << prop.vertex().x() << " " << prop.vertex().y() << " " << prop.vertex().z() << " momentum " << pprop << std::endl;
      } else {
	if (debug_) std::cout << " Prop fails " << std::endl;
	continue;
      }
      
      
      for (unsigned int iClu=0;iClu<cluCollection.size();iClu++) {
	cluRH_p4 = new reco::Candidate::PolarLorentzVector;
	cluRH_p4->SetPt( cluCollection[iClu].et());
	cluRH_p4->SetEta(cluCollection[iClu].eta());
	cluRH_p4->SetPhi(cluCollection[iClu].phi());
	cluRH_p4->SetM(0.);
	
	
	if ( reco::deltaR(*cluRH_p4, *trueEle_p4) < dR &&  (fabs(cluRH_p4->pt() - trueEle_p4->pt()) < 0.5*trueEle_p4->pt() ) ) {
	  dR=  reco::deltaR(*cluRH_p4, *trueEle_p4);
	  matchEle_p4 =  cluRH_p4;
	}
	
      }	    
      if (  matchEle_p4 !=0 && dR<0.1) {
	h_cluRHEt_->Fill ( matchEle_p4->pt() );
	h_cluRHEtoverGenEt_->Fill ( matchEle_p4->pt() / trueEle_p4->pt()); 
	h_dEta_cluRH_gen_ -> Fill(  matchEle_p4->eta() - trueEle_p4->eta());
	float matchDPhi = normalizedPhi ( matchEle_p4->phi()- trueEle_p4->phi());
	h_dPhi_cluRH_gen_ -> Fill(matchDPhi);
	h_deltaR_cluRHGen_ -> Fill (dR);
      }	    
      
	  
    } // end loop over  mcEle
    
  
  }  // if genParticleValid 
  
  nTotCluRH_+= cluCollection.size(); 

  if (tpCluHandleIsValid &&  isGenParticleValid_ && recoContent_) {
  
    h2_recEle_vs_Gen_size_ ->Fill(float(nGenInEB) , gsfElectronsH->size());  
    h2_cluTP_vs_Gen_size_  ->Fill(float(nGenInEB) , tpClu.product()->size());
    h2_cluRH_vs_Gen_size_  ->Fill(float(nGenInEB) , cluCollection.size());
    
  }


  if ( isGenParticleValid_  && recoContent_ )  {

    if ( analyzeElectrons_ ) {

      // test on gsfElectrons
      // matching  the genParticle with reco

      float dR=9999;    
      reco::GenParticleCollection::const_iterator gpIter ;
      for ( gpIter=gp->begin() ; gpIter!=gp->end() ; gpIter++ ) {
	if ( gpIter->pdgId() == 11 ||  gpIter->pdgId() == -11 ) {
	  if (abs(gpIter->eta()) < 1.5 && gpIter->pt()>5 ) {
	    
	    reco::GsfElectronCollection::const_iterator gsfIter ;
	    const reco::GsfElectron* matchEle=0;
	    for (gsfIter=gsfElectronsH->begin(); gsfIter!=gsfElectronsH->end(); gsfIter++ ) { 
	      const reco::GsfElectron* myEle=&(*gsfIter);
	      
	      double dphi =  gsfIter->phi() - gpIter->phi();
	      if ( std::abs(dphi)>CLHEP::pi) { dphi = dphi<0? (CLHEP::twopi) + dphi : dphi - CLHEP::twopi;}
	      double  deltaR2  = ( gsfIter->eta() - gpIter->eta() ) *  ( gsfIter->eta() - gpIter->eta() ) + dphi*dphi;
	      
	      
	      if ( deltaR2 < dR ) {
		dR = deltaR2;
		matchEle=myEle;
	      }
	    }
	    
	    if ( matchEle!=0 && sqrt(dR)<0.1) { 
	      h_elePtRecoOverPtTrue_       ->Fill (  matchEle->pt()/gpIter->pt());
	      h_corrEleEtRecoOverPtTrue_   ->Fill ( ( matchEle->ecalEnergy()/cosh( matchEle->superCluster()->eta()))/gpIter->pt());
	      h_uncorrEleEtRecoOverPtTrue_ ->Fill ( ( matchEle->superCluster()->rawEnergy()/cosh( matchEle->superCluster()->eta()))/gpIter->pt());
	      //
	      float e5x5 =  EcalClusterTools::e5x5( *( matchEle->superCluster()->seed()), &(*rechit_EB_col), &(*topology));
	      float e3x3 =  EcalClusterTools::e3x3( *( matchEle->superCluster()->seed()), &(*rechit_EB_col), &(*topology));
	     	      
	      double dphi5x5=matchEle->superCluster()->phi() - gpIter->phi();
	      if ( std::abs(dphi5x5)>CLHEP::pi) { dphi5x5 = dphi5x5<0? (CLHEP::twopi) + dphi5x5 : dphi5x5 - CLHEP::twopi;}
	      h_dPhi_5x5SC_gen_ -> Fill(dphi5x5);
	      
	      h_5x5SCOverPtTrue_ ->Fill ( (e5x5/cosh( matchEle->superCluster()->eta()))/gpIter->pt());
	      h_3x3SCOverPtTrue_ ->Fill ( (e3x3/cosh( matchEle->superCluster()->eta()))/gpIter->pt());
	      double dPhi =  matchEle->phi() - gpIter->phi();
	      if ( std::abs(dPhi)>CLHEP::pi) { dPhi = dPhi<0? (CLHEP::twopi) + dPhi : dPhi - CLHEP::twopi;}
	      float dr2= ( matchEle->eta() - gpIter->eta() ) *  ( matchEle->eta() - gpIter->eta() ) + dPhi*dPhi;
	      h_deltaR_recoEleGen_-> Fill (sqrt(dr2));
	      h_dEta_gsfEle_gen_ -> Fill( matchEle->eta()-gpIter->eta());
	      h_dPhi_gsfEle_gen_ -> Fill(dPhi);
	    }
	  }
	}
      } // endl loop on gen ele
      
    } else {

      // test on gedPhotons
      // matching  the genParticle with reco

      float dR=9999;    
      reco::GenParticleCollection::const_iterator gpIter ;
      for ( gpIter=gp->begin() ; gpIter!=gp->end() ; gpIter++ ) {
	if ( gpIter->pdgId() == 22) {
	  //	  float gpEta = etaTransformation(gpIter->eta(), gpIter->vz() );
	  float gpEta = gpIter->eta();
	  if (abs(gpEta) < 1.5 && gpIter->pt()>10 ) {
	    
	    reco::PhotonCollection::const_iterator phoIter ;
	    const reco::Photon* matchPho=0;
	    for (phoIter=gedPhotonH->begin(); phoIter!=gedPhotonH->end(); phoIter++ ) { 
	      const reco::Photon* myPho=&(*phoIter);


	      
	      double dphi =  phoIter->phi() - gpIter->phi();
	      if ( std::abs(dphi)>CLHEP::pi) { dphi = dphi<0? (CLHEP::twopi) + dphi : dphi - CLHEP::twopi;}
	      double  deltaR2  = ( phoIter->eta() - gpEta ) *  ( phoIter->eta() - gpEta ) + dphi*dphi;
	      
	      
	      if ( deltaR2 < dR ) {
		dR = deltaR2;
		matchPho=myPho;
	      }
	    }
	    
	    if ( matchPho!=0 && sqrt(dR)<0.1) { 
	      h_elePtRecoOverPtTrue_       ->Fill (  matchPho->pt()/gpIter->pt());
	      h_corrEleEtRecoOverPtTrue_   ->Fill ( ( matchPho->getCorrectedEnergy(reco::Photon::regression1)/cosh( matchPho->superCluster()->eta()))/gpIter->pt());
	      h_uncorrEleEtRecoOverPtTrue_ ->Fill ( ( matchPho->superCluster()->rawEnergy()/cosh( matchPho->superCluster()->eta()))/gpIter->pt());
	      //
	      float e5x5 =  EcalClusterTools::e5x5( *( matchPho->superCluster()->seed()), &(*rechit_EB_col), &(*topology));
	      float e3x3 =  EcalClusterTools::e3x3( *( matchPho->superCluster()->seed()), &(*rechit_EB_col), &(*topology));
	      
	      
	      double dphi5x5=matchPho->superCluster()->phi() - gpIter->phi();
	      if ( std::abs(dphi5x5)>CLHEP::pi) { dphi5x5 = dphi5x5<0? (CLHEP::twopi) + dphi5x5 : dphi5x5 - CLHEP::twopi;}
	      h_dPhi_5x5SC_gen_ -> Fill(dphi5x5);
	      
	      h_5x5SCOverPtTrue_ ->Fill ( (e5x5/cosh( matchPho->superCluster()->eta()))/gpIter->pt());
	      h_3x3SCOverPtTrue_ ->Fill ( (e3x3/cosh( matchPho->superCluster()->eta()))/gpIter->pt());
	      double dPhi =  matchPho->phi() - gpIter->phi();
	      if ( std::abs(dPhi)>CLHEP::pi) { dPhi = dPhi<0? (CLHEP::twopi) + dPhi : dPhi - CLHEP::twopi;}
	      float dr2= ( matchPho->eta() - gpEta ) *  ( matchPho->eta() - gpEta ) + dPhi*dPhi;
	      h_deltaR_recoEleGen_-> Fill (sqrt(dr2));
	      h_dEta_gsfEle_gen_ -> Fill( matchPho->eta()-gpEta);
	      h_dPhi_gsfEle_gen_ -> Fill(dPhi);
	    }
	  }
	}
      } // endl loop on gen photons


    }  // if analyzeElectrons



  } // if genParticleValid and reco content available

  
  std::cout << " SUMMARY " << std::endl;
  std::cout << " Total cluTP " << nTotCluTP_ << " total cluRH " << nTotCluRH_ << std::endl;	
  
}

void
EcalEBTrigPrimAnalyzer::endJob(){
  for (unsigned int i=0;i<ecal_parts_.size();++i) {
    ecal_et_[i]->Write();
    ecal_tt_[i]->Write();
    ecal_fgvb_[i]->Write();
  }

  h_bxNumber_->Write();
  h_pu_->Write();
  hAllTPperEvt_->Write();
  hTPperEvt_->Write();
  //
  for (int i=0; i<5; i++) h_nClu_[i] ->Write();

  hCluTPperEvt_->Write();
  h_nXtals_ ->Write();
  h_etCluTP_ ->Write();

  h_dEta_cluTP_cluRH_->Write();
  h_dPhi_cluTP_cluRH_->Write();
  h_deltaR_cluTPcluRH_  -> Write();
  h_fBrem_truth_-> Write();

  if (isGenParticleValid_) {
    h_deltaR_cluTPGen_  -> Write();
    h_cluTPEtoverGenEt_->Write();
    h_dEta_cluTP_gen_->Write();
    h_dPhi_cluTP_gen_->Write();
    //
    h_dEta_cluRH_gen_->Write();
    h_dPhi_cluRH_gen_->Write();
    h_deltaR_cluRHGen_  -> Write();
    h_cluRHEt_->Write();
    h_cluRHEtoverGenEt_->Write();
    //
    h_elePtRecoOverPtTrue_       ->Write();
    h_corrEleEtRecoOverPtTrue_   ->Write();
    h_uncorrEleEtRecoOverPtTrue_ ->Write();
    h_dPhi_5x5SC_gen_->Write();
    h_5x5SCOverPtTrue_ ->Write();
    h_3x3SCOverPtTrue_ ->Write();
    h_dEta_gsfEle_gen_->Write();
    h_dPhi_gsfEle_gen_->Write();
    h_deltaR_recoEleGen_  -> Write();



  }

  std::cout << " Total number of xtal TP " << nTotTP_ << std::endl;
  if (recHits_) {
    hTPvsTow_ieta_->Write();
    hTPvsTow_eta_->Write();    
    hTPvsRechit_->Write();
    hTPoverRechit_->Write();
    hAllRechitEt_->Write();
    hRechitEt_->Write();
    hDeltaEt_ ->Write();
    hTPEt_->Write();
    hRatioEt_->Write();
    hTP_iphiVsieta_->Write();
    hRH_iphiVsieta_->Write();
    hTP_iphiVsieta_fullrange_->Write();
    hRH_iphiVsieta_fullrange_->Write();
    hCluTPoverRechit_->Write();
    hCluTPvsRechit_->Write();
  }

  h2_recEle_vs_Gen_size_->Write();
  h2_cluTP_vs_Gen_size_ ->Write();
  h2_cluRH_vs_Gen_size_ ->Write();




}



std::vector<SimpleCluster>  EcalEBTrigPrimAnalyzer::makeCluster (std::vector<SimpleCaloHit2> &  hitCollection, int dEta, int dPhi ) {


   if (debug_) std::cout << "  makeCluster  input collection size " << hitCollection.size() << std::endl;

   std::vector<SimpleCluster> clusters;
   while (true) {
     
     SimpleCaloHit2 centerhit(0);      
     for (unsigned int iRH=0;iRH<hitCollection.size();++iRH) {
       SimpleCaloHit2  hit = hitCollection[iRH];  

       float energy = hit.et()/sin(hit.position().theta());

	//if ( energy < 0.2 ) continue;
       	if ( energy < 0.080 ) continue;
       //if ( hit.et() < 0.500 ) continue;
	

	if ( !hit.stale && hit.et() > centerhit.et() ) {
	  centerhit = hit;  

	}      
	if (debug_) std::cout << "  makeCluster energy " << energy << " " << hit.energy() << " et " << hit.et() << " stale " << hit.stale << std::endl;

	
      } //looping over the pseudo-hits

     //if ( centerhit.et() <= 0.350 ) break;
      if ( centerhit.et() <= 1. ) break;
      centerhit.stale=true;
      if (debug_) {
	std::cout << "-------------------------------------" << std::endl;
	std::cout << "New cluster: center crystal pt = " << centerhit.et() << std::endl;
      }



      GlobalVector weightedPosition;
      float totalEnergy = 0.;
      std::vector<float> crystalEt;
      std::vector<EBDetId> crystalId;
    
      for (unsigned int iRH=0;iRH<hitCollection.size();++iRH) {
	SimpleCaloHit2 &hit(hitCollection[iRH]);  
	//float energy = hit.energy();
	//if ( energy < 0.2 ) continue;
	//	if ( energy < 0.08 ) continue;
	if ( hit.et() < 0.500 ) continue;


	if ( !hit.stale &&  (abs(hit.dieta(centerhit)) < dEta && abs(hit.diphi(centerhit)) <  dPhi ) ) {
	  
	  weightedPosition += hit.position()*hit.energy();
	  if (debug_) std::cout << " evolving  weightedPosition " << weightedPosition.eta() << " " << weightedPosition.phi() << std::endl;
	  totalEnergy += hit.energy();
	  hit.stale = true;
	  crystalEt.push_back(hit.et());
	  crystalId.push_back(hit.id());


          if ( debug_) {
	    if ( hit == centerhit )
	      std::cout << "      "; 
	    std::cout <<
	      "\tCrystal (" << hit.dieta(centerhit) << "," << hit.diphi(centerhit) <<
	      ", Et  =" << hit.et() <<
	      ", eta=" << hit.position().eta() <<
	      ", phi=" << hit.position().phi() << std::endl;
	  }

 
	  
	}
       
      }
      float totalEt = totalEnergy*sin(weightedPosition.theta());
      float etaClu= weightedPosition.eta() ;
      float phiClu= weightedPosition.phi();
      SimpleCluster myClu(totalEt);
      myClu.setEta(etaClu);
      myClu.setPhi(phiClu);

      if (debug_) std::cout << " Cluster total energy " << totalEnergy << " total Et " << totalEt  << " weighted eta " << etaClu << " weighted phi " <<  phiClu << std::endl;

      clusters.push_back(myClu);


    } // while true 
    
      
   // looping over the samples of the future TP
  
  
    if (debug_) {
      std::cout <<  " Clusters size " << clusters.size() << std::endl;  
      for (unsigned int iet=0; iet < clusters.size(); iet++) std::cout << " Et  " << clusters[iet].et() << " ";
      std::cout << " " << std::endl;
    }


    return clusters;

}




float EcalEBTrigPrimAnalyzer::normalizedPhi(float  phi)
{
  //---Definitions
  const float PI    = 3.1415927;
  const float TWOPI = 2.0*PI;


  if(phi >  PI) {phi = phi - TWOPI;}
  if(phi < -PI) {phi = phi + TWOPI;}

  return phi;

}
 
float EcalEBTrigPrimAnalyzer::etaTransformation(  float EtaParticle , float Zvertex)  {

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
