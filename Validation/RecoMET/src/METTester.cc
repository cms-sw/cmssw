#include "Validation/RecoMET/interface/METTester.h"
// author: Mike Schmitt, University of Florida
// first version 8/24/2006
// modification: Bobby Scurlock
// date:  03.11.2006
// note:  added RMS(METx) vs SumET capability
// modification: Rick Cavanaugh
// date:  05.11.2006
// note:  cleaned up constructor and beginJob, removed int conv. warning
//        added configuration params

METTester::METTester(const edm::ParameterSet& iConfig)
{
  // DQM ROOT output
  outputFile_ = iConfig.getUntrackedParameter<string>("outputFile", "");

  if ( outputFile_.size() != 0 ) {
    LogInfo("OutputInfo") << " MET Task histograms will be saved to '" << outputFile_.c_str() << "'";
  } else {
    LogInfo("OutputInfo") << " MET Task histograms will NOT be saved";
  }
  InputGenMETLabel_           = iConfig.getParameter<string>("InputGenMETLabel");
  InputCaloMETLabel_          = iConfig.getParameter<string>("InputCaloMETLabel");
  CaloMExResFitMin_           = iConfig.getParameter<double>("CaloMExResFitMin");
  CaloMExResFitMax_           = iConfig.getParameter<double>("CaloMExResFitMax");
  CaloMExResHistoSumETNumBin_ = iConfig.getParameter<int>("CaloMExResHistoSumETNumBin");
  CaloMExResHistoLowSumET_    = iConfig.getParameter<double>("CaloMExResHistoLowSumET");
  CaloMExResHistoHighSumET_   = iConfig.getParameter<double>("CaloMExResHistoHighSumET");
  CaloMExResHistoMExNumBin_   = iConfig.getParameter<int>("CaloMExResHistoMExNumBin");
  CaloMExResHistoLowMEx_      = iConfig.getParameter<double>("CaloMExResHistoLowMEx");
  CaloMExResHistoHighMEx_     = iConfig.getParameter<double>("CaloMExResHistoHighMEx");
}
   
METTester::~METTester() { }

void METTester::endJob() 
{
  FillMETResHisto();
  cout << " outputFile_.size() =  " << outputFile_.size() << endl;
  cout << " dbe_ = " << dbe_ << endl; 
  if ( outputFile_.size() != 0 && dbe_ ) dbe_->save(outputFile_);
}

void METTester::beginJob(const edm::EventSetup& c)
{

  // get hold of back-end interface
  dbe_ = Service<DaqMonitorBEInterface>().operator->();
  
  Char_t histo[64];
  
  if ( dbe_ ) 
    {
      dbe_->setCurrentFolder("METTask");
      
      // CaloMET Histograms
      sprintf(histo,"METTask_CaloMEx");
      meCaloMEx = dbe_->book1D(histo,histo,51,0.,500);
      
      sprintf(histo,"METTask_CaloMEy");
      meCaloMEy = dbe_->book1D(histo,histo,51,0.,500);
      
      sprintf(histo,"METTask_CaloEz");
      meCaloEz = dbe_->book1D(histo,histo,51,0.,500);
      
      sprintf(histo,"METTask_CaloMETSig");
      meCaloMETSig = dbe_->book1D(histo,histo,51,0.,50);
      
      sprintf(histo,"METTask_CaloMET");
      meCaloMET = dbe_->book1D(histo,histo,101,0.,1000);
      
      sprintf(histo,"METTask_CaloMETPhi");
      meCaloMETPhi = dbe_->book1D(histo,histo,51,-M_PI,M_PI);
      
      sprintf(histo,"METTask_CaloSumET");
      meCaloSumET = dbe_->book1D(histo,histo,51,0.,1500);
      
      sprintf(histo,"METTask_CaloMaxEtInEmTowers");
      meCaloMaxEtInEmTowers = dbe_->book1D(histo,histo,51,0.,800);
      
      sprintf(histo,"METTask_CaloMaxEtInHadTowers");
      meCaloMaxEtInHadTowers = dbe_->book1D(histo,histo,51,0.,800);
      
      sprintf(histo,"METTask_CaloEtFractionHadronic");
      meCaloEtFractionHadronic = dbe_->book1D(histo,histo,51,0.,1);
      
      sprintf(histo,"METTask_CaloEmEtFraction");
      meCaloEmEtFraction = dbe_->book1D(histo,histo,51,0.,1);
      
      sprintf(histo,"METTask_CaloHadEtInHB");
      meCaloHadEtInHB = dbe_->book1D(histo,histo,51,0.,800);
      
      sprintf(histo,"METTask_CaloHadEtInHO");
      meCaloHadEtInHO = dbe_->book1D(histo,histo,51,0.,300);
      
      sprintf(histo,"METTask_CaloHadEtInHE");
      meCaloHadEtInHE = dbe_->book1D(histo,histo,51,0.,1000);
      
      sprintf(histo,"METTask_CaloHadEtInHF");
      meCaloHadEtInHF = dbe_->book1D(histo,histo,51,0.,1000);
      
      sprintf(histo,"METTask_CaloEmEtInEB");
      meCaloEmEtInEB = dbe_->book1D(histo,histo,51,0.,800);
      
      sprintf(histo,"METTask_CaloEmEtInEE");
      meCaloEmEtInEE = dbe_->book1D(histo,histo,51,0.,1000);
      
      sprintf(histo,"METTask_CaloEmEtInHF");
      meCaloEmEtInHF = dbe_->book1D(histo,histo,51,0.,1000);
      
      
      // GenMET Histograms
      sprintf(histo,"METTask_GenMEx");
      meGenMEx = dbe_->book1D(histo,histo,51,0.,500);
      
      sprintf(histo,"METTask_GenMEy");
      meGenMEy = dbe_->book1D(histo,histo,51,0.,500);
      
      sprintf(histo,"METTask_GenEz");
      meGenEz = dbe_->book1D(histo,histo,51,0.,500);
      
      sprintf(histo,"METTask_GenMETSig");
      meGenMETSig = dbe_->book1D(histo,histo,51,0.,50);
      
      sprintf(histo,"METTask_GenMET");
      meGenMET = dbe_->book1D(histo,histo,101,0.,1000);
      
      sprintf(histo,"METTask_GenMETPhi");
      meGenMETPhi = dbe_->book1D(histo,histo,51,-M_PI,M_PI);
      
      sprintf(histo,"METTask_GenSumET");
      meGenSumET = dbe_->book1D(histo,histo,51,0.,1500);
      
      sprintf(histo,"METTask_GenEmEnergy");
      meGenEmEnergy = dbe_->book1D(histo,histo,51,0.,1000);
      
      sprintf(histo,"METTask_GenHadEnergy");
      meGenHadEnergy = dbe_->book1D(histo,histo,51,0.,1000);
      
      sprintf(histo,"METTask_GenInvisibleEnergy");
      meGenInvisibleEnergy = dbe_->book1D(histo,histo,51,0.,500);
      
      sprintf(histo,"METTask_GenAuxiliaryEnergy");
      meGenAuxiliaryEnergy = dbe_->book1D(histo,histo,51,0.,500);
      
      // Combined Histograms
      sprintf(histo,"METTask_METSigmaVsGenSumET");
      meMETSigmaVsGenSumET = dbe_->bookProfile(histo,histo,26,0.,1500,21,0.,100);
      
      sprintf(histo,"METTask_meCaloMETvsCaloSumET");
      meCaloMETvsCaloSumET = dbe_->book2D(histo,histo,100,0.,5000, 100, 0,5000);
 
      sprintf(histo,"METTask_meCaloMExvsCaloSumET");
      meCaloMExvsCaloSumET = dbe_->book2D(histo,histo,
					  CaloMExResHistoSumETNumBin_,
					  CaloMExResHistoLowSumET_,
					  CaloMExResHistoHighSumET_, 
					  CaloMExResHistoMExNumBin_, 
					  CaloMExResHistoLowMEx_,
					  CaloMExResHistoHighMEx_);
      h_CaloMExvsCaloSumET = new TH2F(histo, histo, 
				      CaloMExResHistoSumETNumBin_,
				      CaloMExResHistoLowSumET_,
				      CaloMExResHistoHighSumET_, 
				      CaloMExResHistoMExNumBin_, 
				      CaloMExResHistoLowMEx_,
				      CaloMExResHistoHighMEx_);
    }
  
  
}

void METTester::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;

  // Get RecoMET objects from event
  Handle<CaloMETCollection> calo;
  iEvent.getByLabel(InputCaloMETLabel_, calo);
  const CaloMETCollection *calometcol = calo.product();
  const CaloMET calomet = calometcol->front();

  Handle<GenMETCollection> gen;
  iEvent.getByLabel(InputGenMETLabel_, gen);
  const GenMETCollection *genmetcol = gen.product();
  const GenMET genmet = genmetcol->front();

  // Get Generated MET
  double genSumET = genmet.sumEt();
  double genMETSig = genmet.mEtSig();
  double genEz = genmet.e_longitudinal();
  double genMET = genmet.pt();
  double genMEx = genmet.px();
  double genMEy = genmet.py();
  double genMETPhi = genmet.phi();
  double genEmEnergy = genmet.emEnergy();
  double genHadEnergy = genmet.hadEnergy();
  double genInvisibleEnergy= genmet.invisibleEnergy();
  double genAuxiliaryEnergy= genmet.auxiliaryEnergy();
  
  //  std::cout << " MET value = " << calomet.sumEt() << std::endl;

  // Get Reconstructed MET
  double caloSumET = calomet.sumEt();
  double caloMETSig = calomet.mEtSig();
  double caloEz = calomet.e_longitudinal();
  double caloMET = calomet.pt();
  double caloMEx = calomet.px();
  double caloMEy = calomet.py();
  double caloMETPhi = calomet.phi();
  double caloMaxEtInEMTowers = calomet.maxEtInEmTowers();
  double caloMaxEtInHadTowers = calomet.maxEtInHadTowers();
  double caloEtFractionHadronic = calomet.etFractionHadronic();
  double caloEmEtFraction = calomet.emEtFraction();
  double caloHadEtInHB = calomet.hadEtInHB();
  double caloHadEtInHO = calomet.hadEtInHO();
  double caloHadEtInHE = calomet.hadEtInHE();
  double caloHadEtInHF = calomet.hadEtInHF();
  double caloEmEtInEB = calomet.emEtInEB();
  double caloEmEtInEE = calomet.emEtInEE();
  double caloEmEtInHF = calomet.emEtInHF();

  // define "sigma" = abs(MET_calo - MET)
   double sigma = fabs(caloMET - genMET);

  // Fill Gen Histograms
  meGenMEx->Fill(genMEx);
  meGenMEy->Fill(genMEy);
  meGenMET->Fill(genMET);
  meGenMETPhi->Fill(genMETPhi);
  meGenSumET->Fill(genSumET);
  meGenMETSig->Fill(genMETSig);
  meGenEz->Fill(genEz);
  meGenEmEnergy->Fill(genEmEnergy);
  meGenHadEnergy->Fill(genHadEnergy);
  meGenInvisibleEnergy->Fill(genInvisibleEnergy);
  meGenAuxiliaryEnergy->Fill(genAuxiliaryEnergy);
  
  // Fill Calo Histograms
  meCaloMEx->Fill(caloMEx);
  meCaloMEy->Fill(caloMEy);
  meCaloMET->Fill(caloMET);
  meCaloMETPhi->Fill(caloMETPhi);
  meCaloSumET->Fill(caloSumET);
  meCaloMETSig->Fill(caloMETSig);
  meCaloEz->Fill(caloEz);
  meCaloMaxEtInEmTowers->Fill(caloMaxEtInEMTowers);
  meCaloMaxEtInHadTowers->Fill(caloMaxEtInHadTowers);
  meCaloEtFractionHadronic->Fill(caloEtFractionHadronic);
  meCaloEmEtFraction->Fill(caloEmEtFraction);
  meCaloHadEtInHB->Fill(caloHadEtInHB);
  meCaloHadEtInHO->Fill(caloHadEtInHO);
  meCaloHadEtInHE->Fill(caloHadEtInHE);
  meCaloHadEtInHF->Fill(caloHadEtInHF);
  meCaloEmEtInEB->Fill(caloEmEtInEB);
  meCaloEmEtInEE->Fill(caloEmEtInEE);
  meCaloEmEtInHF->Fill(caloEmEtInHF);

  meMETSigmaVsGenSumET->Fill(genSumET,sigma);
  meCaloMETvsCaloSumET->Fill( caloSumET, caloMET);
  meCaloMExvsCaloSumET->Fill( caloSumET, caloMEx);
  h_CaloMExvsCaloSumET->Fill( caloSumET, caloMEx);
  
 
}

  
void METTester::FillMETResHisto()
{
   //----Sigma(RecoMETx) vs RecoSumET--------
  gStyle->SetOptFit(111);
  TCanvas cGeneric("cGeneric","", 400, 900);
  // Histogram filled with RMS of MEx slices
  TH1F h_SigMETxvSumET("hSigMETvSumET", "#sigma(E_{x}^{Reco}) vs. #SigmaE_{T}^{Reco}", 
		       CaloMExResHistoSumETNumBin_,
		       CaloMExResHistoLowSumET_,
		       CaloMExResHistoHighSumET_);
  TH1F h_SigMETx2vSumET("hSigMETx2vSumET", "#sigma^{2}(E_{x}^{Reco}) vs. #SigmaE_{T}^{Reco}", 
		       CaloMExResHistoSumETNumBin_,
		       CaloMExResHistoLowSumET_,
		       CaloMExResHistoHighSumET_);
  h_SigMETxvSumET.GetYaxis()->SetTitle("#sigma(E_{x}) (GeV)"); 
  h_SigMETxvSumET.GetXaxis()->SetTitle("#SigmaE_{T} (GeV)");
  h_SigMETx2vSumET.GetYaxis()->SetTitle("#sigma^{2}(E_{x}) (GeV)"); 
  h_SigMETx2vSumET.GetXaxis()->SetTitle("#SigmaE_{T} (GeV)");
  for (int i=1;i<=h_CaloMExvsCaloSumET->GetNbinsX();i++)
    {
      // Get the RMS for Y-slice 
      float RMS = h_CaloMExvsCaloSumET->ProfileY("",i,i)->GetRMS();
      // Get the error on the RMS for Y-slice 
      float dRMS = h_CaloMExvsCaloSumET->ProfileY("",i,i)->GetRMSError();
      // Get the error on RMS^2 for Y-slice 
      float dRMS_2 = 2*RMS*dRMS;
      int Entries = (int) h_CaloMExvsCaloSumET->GetEntries();
      if ( Entries>0)
	{
	  h_SigMETxvSumET.SetBinContent(i, RMS );
	  h_SigMETxvSumET.SetBinError(i, dRMS );
	  h_SigMETx2vSumET.SetBinContent(i, pow( RMS, 2) );
	  h_SigMETx2vSumET.SetBinError(i, dRMS_2 );
	}
    } // loop over bins in SumET
  
  cGeneric.Divide(1,3);
  cGeneric.cd(1);
  cGeneric.GetPad(1)->SetLogz(1);
  gStyle->SetPalette(1);
  h_CaloMExvsCaloSumET->SetAxisRange(CaloMExResFitMin_, CaloMExResFitMax_);
  h_CaloMExvsCaloSumET->Draw("COLZ");

  cGeneric.cd(2);
  // Function to fit METx vs SumET: sigma^2 = p0^2 + (p1*sqrt(SumET))^2 + (p2*SumET)^2
  TF1 METxRes2Fit("METxRes2Fit","pow([0],2)+pow([1]*sqrt(x),2)+pow([2]*x,2)", CaloMExResFitMin_, CaloMExResFitMax_);
  METxRes2Fit.SetLineColor(kRed);
  // Fit over full range of data
  h_SigMETx2vSumET.SetAxisRange(CaloMExResFitMin_, CaloMExResFitMax_);
  h_SigMETx2vSumET.Fit("METxRes2Fit","V","", CaloMExResFitMin_, CaloMExResFitMax_);
  // Dump histogram to output file

  cGeneric.cd(3);
  TF1 METxResFit("METxResFit","sqrt(METxRes2Fit)", CaloMExResFitMin_, CaloMExResFitMax_);
  METxResFit.SetLineColor(kRed);
  h_SigMETxvSumET.SetAxisRange(CaloMExResFitMin_, CaloMExResFitMax_);
  h_SigMETxvSumET.Draw();
  METxResFit.Draw("LSAME");
  cGeneric.Print("CaloMExResvsCaloSumET.eps");
  //----------------------------------------
}
