// Producer for validation histograms for Calo, JPT and PF jet objects
// F. Ratnikov, Sept. 7, 2006
// Modified by Chiyoung Jeong, Feb. 2, 2010
// Modified by J. Piedra, Sept. 11, 2013
// Modified by Raghav Kunnawalkam Elayavalli, Aug 18th 2014
//                                          , Oct 22nd 2014 to run in 73X

#include "JetTester_HeavyIons.h"

using namespace edm;
using namespace reco;
using namespace std;

JetTester_HeavyIons::JetTester_HeavyIons(const edm::ParameterSet& iConfig) :
  mInputCollection               (iConfig.getParameter<edm::InputTag>       ("src")),
  mInputGenCollection            (iConfig.getParameter<edm::InputTag>       ("srcGen")),
  mInputPFCandCollection         (iConfig.getParameter<edm::InputTag>       ("PFcands")),
//  rhoTag                         (iConfig.getParameter<edm::InputTag>       ("srcRho")),
  centrality                     (iConfig.getParameter<edm::InputTag>       ("centrality")),
  mOutputFile                    (iConfig.getUntrackedParameter<std::string>("OutputFile","")),
  JetType                        (iConfig.getUntrackedParameter<std::string>("JetType")),
  UEAlgo                         (iConfig.getUntrackedParameter<std::string>("UEAlgo")),
  Background                     (iConfig.getParameter<edm::InputTag>       ("Background")),
  mRecoJetPtThreshold            (iConfig.getParameter<double>              ("recoJetPtThreshold")),
  mMatchGenPtThreshold           (iConfig.getParameter<double>              ("matchGenPtThreshold")),
  mGenEnergyFractionThreshold    (iConfig.getParameter<double>              ("genEnergyFractionThreshold")),
  mReverseEnergyFractionThreshold(iConfig.getParameter<double>              ("reverseEnergyFractionThreshold")),
  mRThreshold                    (iConfig.getParameter<double>              ("RThreshold")),
  JetCorrectionService           (iConfig.getParameter<std::string>         ("JetCorrections"))
{
  std::string inputCollectionLabel(mInputCollection.label());

  // std::size_t foundCaloCollection = inputCollectionLabel.find("Calo");
  // std::size_t foundJPTCollection  = inputCollectionLabel.find("JetPlusTrack");
  // std::size_t foundPFCollection   = inputCollectionLabel.find("PF");

  isCaloJet = (std::string("calo")==JetType);
  isJPTJet  = (std::string("jpt") ==JetType);
  isPFJet   = (std::string("pf")  ==JetType);

  //consumes
  pvToken_ = consumes<std::vector<reco::Vertex> >(edm::InputTag("offlinePrimaryVertices"));
  caloTowersToken_ = consumes<CaloTowerCollection>(edm::InputTag("towerMaker"));
  if (isCaloJet) caloJetsToken_  = consumes<reco::CaloJetCollection>(mInputCollection);
  if (isJPTJet)  jptJetsToken_   = consumes<reco::JPTJetCollection>(mInputCollection);
  if (isPFJet)   {
    if(std::string("Pu")==UEAlgo) basicJetsToken_    = consumes<reco::BasicJetCollection>(mInputCollection);
    if(std::string("Vs")==UEAlgo) pfJetsToken_    = consumes<reco::PFJetCollection>(mInputCollection);
  }

  genJetsToken_ = consumes<reco::GenJetCollection>(edm::InputTag(mInputGenCollection));
  evtToken_ = consumes<edm::HepMCProduct>(edm::InputTag("generator"));
  pfCandToken_ = consumes<reco::PFCandidateCollection>(mInputPFCandCollection);
  pfCandViewToken_ = consumes<reco::CandidateView>(mInputPFCandCollection);
  //backgrounds_ = consumes<reco::VoronoiBackground>(Background);
  backgrounds_ = consumes<edm::ValueMap<reco::VoronoiBackground>>(Background);
  backgrounds_value_ = consumes<std::vector<float>>(Background);
  centralityToken_ = consumes<reco::Centrality>(centrality);
  
  // need to initialize the PF cand histograms : which are also event variables 

  mNPFpart = 0;
  mPFPt = 0;
  mPFEta = 0;
  mPFPhi = 0;
  mPFVsPt = 0;
  mPFVsPtInitial = 0;
  mPFVsPtEqualized = 0;
  mPFArea = 0;
  mSumpt = 0;
  mvn = 0;
  mpsin = 0;
  
  mSumPFVsPt = 0;
  mSumPFVsPtInitial = 0;
  mSumPFPt = 0;
  mSumPFVsPtInitial_eta = 0;
  mSumPFVsPt_eta = 0;
  mSumPFPt_eta = 0;

  // Events variables
  mNvtx         = 0;
  mHF           = 0;

  // added Jan 12th 2015
  mSumPFVsPtInitial_HF = 0;
  mSumPFVsPt_HF = 0;
  mSumPFPt_HF = 0;
  mPFVsPtInitial_eta_phi = 0;
  mPFVsPt_eta_phi = 0;
  mPFPt_eta_phi = 0;
  //mDeltapT_HF = 0;
  mDeltapT_eta = 0;
  //mDeltapT_phiMinusPsi2 = 0;
  mDeltapT_eta_phi = 0;
  
  // Jet parameters
  mEta          = 0;
  mPhi          = 0;
  mEnergy       = 0;
  mP            = 0;
  mPt           = 0;
  mMass         = 0;
  mConstituents = 0;
  mJetArea      = 0;
  mjetpileup    = 0;
  mNJets_40     = 0;
  
}
  //DQMStore* dbe = &*edm::Service<DQMStore>();
   
void JetTester_HeavyIons::bookHistograms(DQMStore::IBooker & ibooker, edm::Run const & iRun,edm::EventSetup const &) 
  {

    //if (dbe) {
    //dbe->setCurrentFolder("JetMET/JetValidation/"+mInputCollection.label());

    ibooker.setCurrentFolder("JetMET/JetValidation/"+mInputCollection.label());

    // double log10PtMin  = 0.50;
    // double log10PtMax  = 3.75;
    // int    log10PtBins = 26; 

    // double etaRange[91] = {
    //     -6.0, -5.8, -5.6, -5.4, -5.2, -5.0, -4.8, -4.6, -4.4, -4.2,
    // 		     -4.0, -3.8, -3.6, -3.4, -3.2, -3.0, -2.9, -2.8, -2.7, -2.6,
    // 		     -2.5, -2.4, -2.3, -2.2, -2.1, -2.0, -1.9, -1.8, -1.7, -1.6,
    // 		     -1.5, -1.4, -1.3, -1.2, -1.1, -1.0, -0.9, -0.8, -0.7, -0.6,
    // 		     -0.5, -0.4, -0.3, -0.2, -0.1,
    // 		     0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
    // 		     1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9,
    // 		     2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9,
    // 		     3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8,
    // 		     5.0, 5.2, 5.4, 5.6, 5.8, 6.0
    //    };

    //cout<<"inside the book histograms function"<<endl;

    // particle flow variables histograms 
    mNPFpart         = ibooker.book1D("NPFpart","No of particle flow candidates",1000,0,10000);
    mPFPt            = ibooker.book1D("PFPt","PF candidate p_{T}",100,0,1000);
    mPFEta           = ibooker.book1D("PFEta","PF candidate #eta",120,-6,6);
    mPFPhi           = ibooker.book1D("PFPhi","PF candidate #phi",70,-3.5,3.5);
    mPFVsPt          = ibooker.book1D("PFVsPt","Vs PF candidate p_{T}",100,0,1000);
    mPFVsPtInitial   = ibooker.book1D("PFVsPtInitial","Vs background subtracted PF candidate p_{T}",100,0,1000);
    mPFVsPtEqualized = ibooker.book1D("PFVsPtEqualized","Vs equalized subtracted PF candidate p_{T}",100,0,1000);
    mPFArea          = ibooker.book1D("PFArea","VS PF candidate area",100,0,4);
    mSumpt           = ibooker.book1D("SumpT","Sum p_{T} of all the PF candidates pre event",1000,0,10000);
    mvn              = ibooker.book1D("vn","vn",100,0,10);
    mpsin            = ibooker.book1D("mpsin","psin",100,0,10);
    mSumPFVsPt       = ibooker.book1D("SumPFVsPt","Sum of PF VS p_T ",10000,-10000,10000);
    mSumPFVsPtInitial= ibooker.book1D("SumPFVsPtInitial","Sum PF VS p_T initial",10000,-10000,10000);
    mSumPFPt         = ibooker.book1D("SumPFPt","",10000,-10000,10000);
    mSumPFVsPt_eta   = ibooker.book2D("SumPFVsPt_etaBins","",60,-6,+6,10000,-10000,10000);
    mSumPFVsPtInitial_eta   = ibooker.book2D("SumPFVsPtInitial_etaBins","",60,-6,+6,10000,-10000,10000);
    mSumPFPt_eta     = ibooker.book2D("SumPFPt_etaBins","",60,-6,+6,10000,-10000,10000);

    // Event variables
    mNvtx            = ibooker.book1D("Nvtx",           "number of vertices", 60, 0, 60);
    mHF              = ibooker.book1D("HF", "HF energy distribution",1000,0,10000);
    
    // Jet parameters
    mEta             = ibooker.book1D("Eta",          "Eta",          120,   -6,    6); 
    mPhi             = ibooker.book1D("Phi",          "Phi",           70, -3.5,  3.5); 
    mPt              = ibooker.book1D("Pt",           "Pt",           100,    0,  1000); 
    mP               = ibooker.book1D("P",            "P",            100,    0,  1000); 
    mEnergy          = ibooker.book1D("Energy",       "Energy",       100,    0,  1000); 
    mMass            = ibooker.book1D("Mass",         "Mass",         100,    0,  200); 
    mConstituents    = ibooker.book1D("Constituents", "Constituents", 100,    0,  100); 
    mJetArea         = ibooker.book1D("JetArea",      "JetArea",       100,   0, 4);
    mjetpileup       = ibooker.book1D("jetPileUp","jetPileUp",100,0,150);
    mNJets_40        = ibooker.book1D("NJets", "NJets 40<Pt",  50,    0,   50);

    // added Jan 12th 2015
    mSumPFVsPtInitial_HF    = ibooker.book2D("SumPFVsPtInitial_HF","",2000,-10000,10000,1000,0,10000);
    mSumPFVsPt_HF    = ibooker.book2D("SumPFVsPt_HF","",2000,-10000,10000,1000,0,10000);
    mSumPFPt_HF    = ibooker.book2D("SumPFPt_HF","",2000,-10000,10000,1000,0,10000);
    mPFVsPtInitial_eta_phi    = ibooker.book3D("SumPFVsPtInitial_eta_phi","",100,0,1000,120,-6,6,70,-3.5,3.5);
    mPFVsPt_eta_phi    = ibooker.book3D("SumPFVsPt_eta_phi","",100,0,1000,120,-6,6,70,-3.5,3.5);
    mPFPt_eta_phi    = ibooker.book3D("SumPFPt_eta_phi","",100,0,1000,120,-6,6,70,-3.5,3.5);
    //mDeltapT_HF  = ibooker.book2D("DeltapT_HF","",400,-200,200,1000,0,10000);
    mDeltapT_eta = ibooker.book2D("DeltapT_eta","",400,-200,200,70,-6,6);
    //mDeltapT_phiMinusPsi2 = ibooker.book2D("DeltapT_phiMinusPsi2","",400,-200,200,35,-1.75,1.75);
    mDeltapT_eta_phi = ibooker.book3D("DeltapT_eta_phi","",400,-200,200,120,-6,6,70,-3.5,3.5);
    
    if (mOutputFile.empty ()) 
      LogInfo("OutputInfo") << " Histograms will NOT be saved";
    else 
      LogInfo("OutputInfo") << " Histograms will be saved to file:" << mOutputFile;
  }



//------------------------------------------------------------------------------
// ~JetTester_HeavyIons
//------------------------------------------------------------------------------
JetTester_HeavyIons::~JetTester_HeavyIons() {}


//------------------------------------------------------------------------------
// beginJob
//------------------------------------------------------------------------------
void JetTester_HeavyIons::beginJob() {
  std::cout<<"inside the begin job function"<<endl;
}


//------------------------------------------------------------------------------
// endJob
//------------------------------------------------------------------------------
void JetTester_HeavyIons::endJob()
{
  if (!mOutputFile.empty() && &*edm::Service<DQMStore>())
    {
      edm::Service<DQMStore>()->save(mOutputFile);
    }
}


//------------------------------------------------------------------------------
// analyze
//------------------------------------------------------------------------------
void JetTester_HeavyIons::analyze(const edm::Event& mEvent, const edm::EventSetup& mSetup)
{
  //std::cout<<"in the analyze function"<<endl;
  // Get the primary vertices
  //----------------------------------------------------------------------------
  edm::Handle<vector<reco::Vertex> > pvHandle;
  mEvent.getByToken(pvToken_, pvHandle);

  int nGoodVertices = 0;

  if (pvHandle.isValid())
    {
      for (unsigned i=0; i<pvHandle->size(); i++)
	{
	  if ((*pvHandle)[i].ndof() > 4 &&
	      (fabs((*pvHandle)[i].z()) <= 24) &&
	      (fabs((*pvHandle)[i].position().rho()) <= 2))
	    nGoodVertices++;
	}
    }

  mNvtx->Fill(nGoodVertices);


  // Get the Jet collection
  //----------------------------------------------------------------------------
  //math::XYZTLorentzVector p4tmp[2];
  
  std::vector<Jet> recoJets;
  recoJets.clear();
  
  edm::Handle<CaloJetCollection>  caloJets;
  edm::Handle<JPTJetCollection>   jptJets;
  edm::Handle<PFJetCollection>    pfJets;
  edm::Handle<BasicJetCollection> basicJets;
  
  // Get the Particle flow candidates and the Voronoi variables 
  edm::Handle<reco::PFCandidateCollection> pfCandidates;
  edm::Handle<reco::CandidateView> candidates_;
  
  //const reco::PFCandidateCollection *pfCandidateColl = pfcandidates.product();
  edm::Handle<edm::ValueMap<VoronoiBackground>> VsBackgrounds;
  edm::Handle<std::vector<float>> vn_;

  edm::Handle<reco::Centrality> cent;
  
  // for example
  // edm::Handle<PFCandidate /*name of actual handle*/> pfcand;
  // mEvent.getByToken(PFCandToken_,pfcand);
  
  if (isCaloJet) mEvent.getByToken(caloJetsToken_, caloJets);
  if (isJPTJet)  mEvent.getByToken(jptJetsToken_, jptJets);
  if (isPFJet) {  
    if(std::string("Pu")==UEAlgo) mEvent.getByToken(basicJetsToken_, basicJets);
    if(std::string("Vs")==UEAlgo) mEvent.getByToken(pfJetsToken_, pfJets);
  }
  
  mEvent.getByToken(pfCandToken_, pfCandidates);
  mEvent.getByToken(pfCandViewToken_, candidates_);

  mEvent.getByToken(backgrounds_, VsBackgrounds);
  mEvent.getByToken(backgrounds_value_, vn_);

  // get the centrality 
  mEvent.getByToken(centralityToken_, cent);

  mHF->Fill(cent->EtHFtowerSum());
  Float_t HF_energy = cent->EtHFtowerSum();
  
  const reco::PFCandidateCollection *pfCandidateColl = pfCandidates.product();
  
  Float_t vsPt=0;
  Float_t vsPtInitial = 0;
  Float_t vsArea = 0;
  Int_t NPFpart = 0;
  Float_t pfPt = 0;
  Float_t pfEta = 0;
  Float_t pfPhi = 0;
  Float_t SumPt_value = 0;
  Float_t vn_value[200];
  Float_t psin_value[200];

  Float_t vn[fourierOrder_][etaBins_];
  Float_t psin[fourierOrder_][etaBins_];
  Float_t sumpT[etaBins_];

  double edge_pseudorapidity[etaBins_ +1] = {-5.191, -2.650, -2.043, -1.740, -1.479, -1.131, -0.783, -0.522, 0.522, 0.783, 1.131, 1.479, 1.740, 2.043, 2.650, 5.191 };

  UEParameters vnUE(vn_.product(),fourierOrder_,etaBins_);
  const std::vector<float>& vue = vnUE.get_raw();
  
  for(int ieta = 0;ieta<etaBins_;++ieta){
    sumpT[ieta] = vnUE.get_sum_pt(ieta);
    for(int ifour = 0;ifour<fourierOrder_; ++ifour){
      vn[ifour][ieta] = vnUE.get_vn(ifour,ieta);
      vn_value[ifour * etaBins_ + ieta]= vnUE.get_vn(ifour,ieta);
      mvn->Fill(vn_value[ifour * etaBins_ + ieta]);

      psin[ifour][ieta] = vnUE.get_psin(ifour,ieta);
      psin_value[ifour * etaBins_ + ieta] = vnUE.get_psin(ifour,ieta);
      mpsin->Fill(psin_value[ifour * etaBins_ + ieta]);

      if(0>1)std::cout<<vn[ifour][ieta]<<" "<<psin[ifour][ieta]<<" "<<sumpT[ieta]<<std::endl;

    }
  }

  /*
  //lets start making the necessary plots 
  Float_t Vs_0_x_minus = sumpT[0]*vn[0][0]*TMath::Cos(0*psin[0][0]);
  Float_t Vs_0_x_plus = sumpT[14]*vn[0][14]*TMath::Cos(0*psin[0][14]);
  Float_t Vs_0_y_minus = sumpT[0]*vn[0][0]*TMath::Sin(0*psin[0][0]);
  Float_t Vs_0_y_plus = sumpT[14]*vn[0][14]*TMath::Sin(0*psin[0][14]);
  Float_t Vs_0_x = Vs_0_x_minus + Vs_0_x_plus;
  Float_t Vs_0_y = Vs_0_y_minus + Vs_0_y_plus;

  Float_t Vs_1_x_minus = sumpT[0]*vn[1][0]*TMath::Cos(1*psin[1][0]);
  Float_t Vs_1_x_plus = sumpT[14]*vn[1][14]*TMath::Cos(1*psin[1][14]);
  Float_t Vs_1_y_minus = sumpT[0]*vn[1][0]*TMath::Sin(1*psin[1][0]);
  Float_t Vs_1_y_plus = sumpT[14]*vn[1][14]*TMath::Sin(1*psin[1][14]);
  Float_t Vs_1_x = Vs_1_x_minus + Vs_0_x_plus;
  Float_t Vs_1_y = Vs_1_y_minus + Vs_0_y_plus;

  Float_t Vs_2_x_minus = sumpT[0]*vn[2][0]*TMath::Cos(2*psin[2][0]);
  Float_t Vs_2_x_plus = sumpT[14]*vn[2][14]*TMath::Cos(2*psin[2][14]);
  Float_t Vs_2_y_minus = sumpT[0]*vn[2][0]*TMath::Sin(2*psin[2][0]);
  Float_t Vs_2_y_plus = sumpT[14]*vn[2][14]*TMath::Sin(2*psin[2][14]);
  Float_t Vs_2_x = Vs_2_x_minus + Vs_2_x_plus;
  Float_t Vs_2_y = Vs_2_y_minus + Vs_2_y_plus;

  Float_t Vs_3_x_minus = sumpT[0]*vn[3][0]*TMath::Cos(3*psin[3][0]);
  Float_t Vs_3_x_plus = sumpT[14]*vn[3][14]*TMath::Cos(3*psin[3][14]);
  Float_t Vs_3_y_minus = sumpT[0]*vn[3][0]*TMath::Sin(3*psin[3][0]);
  Float_t Vs_3_y_plus = sumpT[14]*vn[3][14]*TMath::Sin(3*psin[3][14]);
  Float_t Vs_3_x = Vs_3_x_minus + Vs_3_x_plus;
  Float_t Vs_3_y = Vs_3_y_minus + Vs_3_y_plus;

  Float_t Vs_4_x_minus = sumpT[0]*vn[4][0]*TMath::Cos(4*psin[4][0]);
  Float_t Vs_4_x_plus = sumpT[14]*vn[4][14]*TMath::Cos(4*psin[4][14]);
  Float_t Vs_4_y_minus = sumpT[0]*vn[4][0]*TMath::Sin(4*psin[4][0]);
  Float_t Vs_4_y_plus = sumpT[14]*vn[4][14]*TMath::Sin(4*psin[4][14]);
  Float_t Vs_4_x = Vs_4_x_minus + Vs_4_x_plus;
  Float_t Vs_4_y = Vs_4_y_minus + Vs_4_y_plus;  
  */

  Float_t SumPFVsPtInitial[etaBins_];
  Float_t SumPFVsPt[etaBins_];
  Float_t SumPFPt[etaBins_];

  Float_t DeltapT = 0;

  for(unsigned icand=0;icand<pfCandidateColl->size(); icand++){
    
    const reco::PFCandidate pfCandidate = pfCandidateColl->at(icand);
    reco::CandidateViewRef ref(candidates_,icand);
    
    if(std::string("Vs")==UEAlgo) {
      
      const reco::VoronoiBackground& voronoi = (*VsBackgrounds)[ref];
      vsPt = voronoi.pt();
      vsPtInitial = voronoi.pt_subtracted();
      vsArea = voronoi.area();
      
      //std::cout<<"vsPt = "<<vsPt<<"; vsPtInitial = "<<vsPtInitial<<"; vsArea = "<<vsArea<<std::endl;
    }
    
    NPFpart++;
    pfPt = pfCandidate.pt();
    pfEta = pfCandidate.eta();
    pfPhi = pfCandidate.phi();

    mPFVsPtInitial_eta_phi->Fill(vsPtInitial,pfEta,pfPhi);
    mPFVsPt_eta_phi->Fill(vsPt,pfEta,pfPhi);
    mPFPt_eta_phi->Fill(pfPt,pfEta,pfPhi);

    DeltapT = pfPt - vsPtInitial;

    mDeltapT_eta->Fill(DeltapT,pfEta);
    mDeltapT_eta_phi->Fill(DeltapT,pfEta,pfPhi);

    //std::cout<<pfPt<<" "<<pfEta<<" "<<pfPhi<<" "<<std::endl;

    for(size_t k = 0;k<nedge_pseudorapidity-1; k++){
      if(pfEta >= edge_pseudorapidity[k] && pfEta < edge_pseudorapidity[k+1]){
	SumPFVsPtInitial[k] = SumPFVsPtInitial[k] + vsPtInitial;
	SumPFVsPt[k] = SumPFVsPt[k] + vsPt;
	SumPFPt[k] = SumPFPt[k] + pfPt;
      }// eta selection statement 
      
    }// eta bin loop
    
    
    SumPt_value = SumPt_value + pfPt;
    
    mPFPt->Fill(pfPt);
    mPFEta->Fill(pfEta);
    mPFPhi->Fill(pfPhi);
    mPFVsPt->Fill(vsPt);
    mPFVsPtInitial->Fill(vsPtInitial);
    //mPFVsPtEqualized
    mPFArea->Fill(vsArea);
    
  }

  Float_t Evt_SumPFVsPt = 0;
  Float_t Evt_SumPFVsPtInitial = 0;
  Float_t Evt_SumPFPt = 0;

  for(size_t  k = 0;k<nedge_pseudorapidity-1;k++){
    
    mSumPFVsPtInitial->Fill(SumPFVsPtInitial[k]);
    Evt_SumPFVsPtInitial = Evt_SumPFVsPtInitial + SumPFVsPtInitial[k];
    mSumPFVsPt->Fill(SumPFVsPt[k]);
    Evt_SumPFVsPt = Evt_SumPFVsPt + SumPFVsPt[k];
    mSumPFPt->Fill(SumPFPt[k]);
    Evt_SumPFPt = Evt_SumPFPt + SumPFPt[k];

    mSumPFVsPtInitial_eta->Fill(edge_pseudorapidity[k],SumPFVsPtInitial[k]);
    mSumPFVsPt_eta->Fill(edge_pseudorapidity[k],SumPFVsPt[k]);
    mSumPFPt_eta->Fill(edge_pseudorapidity[k],SumPFPt[k]);
    
  }// eta bin loop  

  mSumPFVsPtInitial_HF->Fill(Evt_SumPFVsPtInitial,HF_energy);
  mSumPFVsPt_HF->Fill(Evt_SumPFVsPt,HF_energy);
  mSumPFPt_HF->Fill(Evt_SumPFPt,HF_energy);
  
  
  mNPFpart->Fill(NPFpart);
  mSumpt->Fill(SumPt_value);
  
  //std::cout<<"finished loading the pfcandidates"<<std::endl;
  
  if (isCaloJet)
    {
      //std::cout<<caloJets->size()<<endl;
      for (unsigned ijet=0; ijet<caloJets->size(); ijet++) recoJets.push_back((*caloJets)[ijet]);
    }
  
  if (isJPTJet)
    {
      //std::cout<<jptJets->size()<<endl;
      for (unsigned ijet=0; ijet<jptJets->size(); ijet++) recoJets.push_back((*jptJets)[ijet]);
    }
  
  if (isPFJet) {
    if(std::string("Pu")==UEAlgo){
      //std::cout<<basicJets->size()<<endl;
      for (unsigned ijet=0; ijet<basicJets->size();ijet++) recoJets.push_back((*basicJets)[ijet]);
    }
    if(std::string("Vs")==UEAlgo){
      //std::cout<<pfJets->size()<<endl;
      for (unsigned ijet=0; ijet<pfJets->size(); ijet++) recoJets.push_back((*pfJets)[ijet]);
    }
  }
  
  /*
    std::cout<<mInputCollection.label()<<endl;
    std::cout<<"jet type = "<<JetType<<endl;
    std::cout<<"UE algorithm = "<<UEAlgo<<endl;
    std::cout<<"size of jets = "<<recoJets.size()<<endl;
    if(isCaloJet)
    std::cout<<"isValid = "<<caloJets.isValid()<<endl; 
    if(isJPTJet)
    std::cout<<"isValid = "<<jptJets.isValid()<<endl; 
    if(isPFJet)
    std::cout<<"isValid = "<<pfJets.isValid()<<endl; 
  */
    
  if (isCaloJet && !caloJets.isValid()) return;
  if (isJPTJet  && !jptJets.isValid())  return;
  if (isPFJet){
    if(std::string("Pu")==UEAlgo){if(!basicJets.isValid())   return;}
    if(std::string("Vs")==UEAlgo){if(!pfJets.isValid())   return;}
  }
  
  
  //std::cout<<"after the trip point"<<endl;
  //std::cout<<mInputCollection.label()<<endl;
  //std::cout<<"jet type = "<<JetType<<endl;
  //std::cout<<"size of jets = "<<recoJets.size()<<endl;
  
  // int nJet      = 0;
  // int nJet_E_20_40 = 0;
  // int nJet_B_20_40 = 0;
  // int nJet_E_40 = 0;
  // int nJet_B_40 = 0;
  int nJet_40 = 0;
  
  for (unsigned ijet=0; ijet<recoJets.size(); ijet++) {

    //std::cout<<"pt = "<<recoJets[ijet].pt()<<endl;	  
    
    if (recoJets[ijet].pt() > mRecoJetPtThreshold) {
      //counting forward and barrel jets
      //cout<<"inside jet pt > 10 condition"<<endl;
      
      // get an idea of no of jets with pT>40 GeV 
      if(recoJets[ijet].pt() > 40)
	nJet_40++;
      
      if (mEta) mEta->Fill(recoJets[ijet].eta());
      if (mjetpileup) mjetpileup->Fill(recoJets[ijet].pileup());
      if (mJetArea)      mJetArea     ->Fill(recoJets[ijet].jetArea());
      if (mPhi)          mPhi         ->Fill(recoJets[ijet].phi());
      if (mEnergy)       mEnergy      ->Fill(recoJets[ijet].energy());
      if (mP)            mP           ->Fill(recoJets[ijet].p());
      if (mPt)           mPt          ->Fill(recoJets[ijet].pt());
      if (mMass)         mMass        ->Fill(recoJets[ijet].mass());
      if (mConstituents) mConstituents->Fill(recoJets[ijet].nConstituents());
      
    }
  }
  
  if (mNJets_40) mNJets_40->Fill(nJet_40); 
  
}


//------------------------------------------------------------------------------
// fillMatchHists
//------------------------------------------------------------------------------
void JetTester_HeavyIons::fillMatchHists(const double GenEta,
			       const double GenPhi,
			       const double GenPt,
			       const double RecoEta,
			       const double RecoPhi,
			       const double RecoPt) 
{
  //nothing for now. 
}
