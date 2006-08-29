#include "Validation/RecoMET/interface/METTester.h"
// author: Mike Schmitt, University of Florida
// first version 8/24/2006

METTester::METTester(const edm::ParameterSet& iConfig)
{
  // DQM ROOT output
  outputFile_ = iConfig.getUntrackedParameter<string>("outputFile", "");

  if ( outputFile_.size() != 0 ) {
    LogInfo("OutputInfo") << " MET Task histograms will be saved to '" << outputFile_.c_str() << "'";
  } else {
    LogInfo("OutputInfo") << " MET Task histograms will NOT be saved";
  }
  
  // get hold of back-end interface
  dbe_ = Service<DaqMonitorBEInterface>().operator->();
 
  Char_t histo[64];

  if ( dbe_ ) {
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
    
  }

}
   
METTester::~METTester()
{
  endJob();

/*  cout << " outputFile_.size() =  " << outputFile_.size() << endl;
  cout << " dbe_ = " << dbe_ << endl; 
 if ( outputFile_.size() != 0 && dbe_ ) dbe_->save(outputFile_);
*/
}

void METTester::endJob() {
 cout << " outputFile_.size() =  " << outputFile_.size() << endl;
  cout << " dbe_ = " << dbe_ << endl; 
 if ( outputFile_.size() != 0 && dbe_ ) dbe_->save(outputFile_);
}

void METTester::beginJob(const edm::EventSetup& c){

}


void METTester::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;

  // Get RecoMET objects from event
  Handle<CaloMETCollection> calo;
  iEvent.getByLabel("met", calo);
  const CaloMETCollection *calometcol = calo.product();
  const CaloMET calomet = calometcol->front();

  Handle<GenMETCollection> gen;
  iEvent.getByLabel("genMet", gen);
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


}


DEFINE_SEAL_MODULE ();
DEFINE_ANOTHER_FWK_MODULE (METTester) ;
