#include "TopQuarkAnalysis/TopEventProducers/interface/StEvtSolutionMaker.h"

//
// constructors and destructor
//
StEvtSolutionMaker::StEvtSolutionMaker(const edm::ParameterSet& iConfig)
{
   electronSrc_     = iConfig.getParameter<edm::InputTag>("electronSource");
   muonSrc_         = iConfig.getParameter<edm::InputTag>("muonSource");
   metSrc_          = iConfig.getParameter<edm::InputTag>("metSource");
   lJetSrc_         = iConfig.getParameter<edm::InputTag>("lJetSource");
   bJetSrc_         = iConfig.getParameter<edm::InputTag>("bJetSource");
   leptonFlavour_   = iConfig.getParameter< string > 	  ("leptonFlavour");
   //   jetInput_        = iConfig.getParameter< string > 	  ("jetInput");
   doKinFit_        = iConfig.getParameter< bool >        ("doKinFit");
   addLRJetComb_    = iConfig.getParameter< bool >        ("addLRJetComb");
   maxNrIter_       = iConfig.getParameter< int >         ("maxNrIter");
   maxDeltaS_       = iConfig.getParameter< double >      ("maxDeltaS");
   maxF_            = iConfig.getParameter< double >      ("maxF");
   param_           = iConfig.getParameter< int >         ("parametrisation");
   constraints_     = iConfig.getParameter< vector<int> > ("constraints");
   matchToGenEvt_   = iConfig.getParameter< bool > 	  ("matchToGenEvt");

   produces<vector<StEvtSolution> >();
   
   // define kinfitter
   if(doKinFit_){
     if(param_ == 1) myKinFitterEtEtaPhi   = new StKinFitterEtEtaPhi(maxNrIter_, maxDeltaS_, maxF_, constraints_);
     if(param_ == 2) myKinFitterEtThetaPhi = new StKinFitterEtThetaPhi(maxNrIter_, maxDeltaS_, maxF_, constraints_);
     if(param_ == 3) myKinFitterEMom       = new StKinFitterEMom(maxNrIter_, maxDeltaS_, maxF_, constraints_);
   }
}


StEvtSolutionMaker::~StEvtSolutionMaker()
{
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void StEvtSolutionMaker::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  //
  //  TopObject Selection
  //

   // select lepton (the TtLepton vectors are, for the moment, sorted on pT)
   bool leptonFound = false;
   TopMuon selMuon;
   if(leptonFlavour_ == "muon"){
     edm::Handle<std::vector<TopMuon> > muons;
     iEvent.getByLabel(muonSrc_, muons);
     if( muons->size() > 0 ){
       //select highest pT muon
       selMuon = (*muons)[0];
       leptonFound = true;
     }
   }
   TopElectron selElectron;
   if(leptonFlavour_ == "electron"){
     edm::Handle<std::vector<TopElectron> > electrons;
     iEvent.getByLabel(electronSrc_, electrons);
     if( electrons->size() > 0 ){
       //select highest pT electron
       selElectron = (*electrons)[0];
       leptonFound = true;
     }       
   }  

   // select MET (TopMET vector is sorted on ET)
   bool metFound = false;
   TopMET selMET;
   edm::Handle<std::vector<TopMET> > mets;
   iEvent.getByLabel(metSrc_, mets);
   if( mets->size() > 0 ){
     //select highest Et MET
     selMET = (*mets)[0];
     metFound = true;
   }

   // select Jets (TopJet vector is sorted on recET, so four first elements in both the lJets and bJets vector are the same )
   bool jetsFound = false;
   edm::Handle<std::vector<TopJet> > lJets;
   iEvent.getByLabel(lJetSrc_, lJets);
   edm::Handle<std::vector<TopJet> > bJets;
   iEvent.getByLabel(bJetSrc_, bJets);
   unsigned int maxJets=2;//this has to become a custom-defined parameter (we may want 2 or 3 jets)
   if (lJets->size() >= 2) jetsFound = true;
   
   
   
   vector<StEvtSolution> *evtsols = new vector<StEvtSolution>();
   if(leptonFound && metFound && jetsFound){
     cout<<"constructing solutions"<<endl;
     for (unsigned int b=0; b<maxJets; b++) {
       for (unsigned int l=0; l<maxJets; l++) {
	 if(b!=l){  // to avoid double counting
	   StEvtSolution asol;
	   if(leptonFlavour_ == "muon")     asol.setMuon(selMuon);
	   if(leptonFlavour_ == "electron") asol.setElectron(selElectron);
	   asol.setMET(selMET);
	   asol.setBottom((*bJets)[b]);
	   asol.setLight((*lJets)[l]);
	   if(doKinFit_){
	     if(param_ == 1) asol = myKinFitterEtEtaPhi->addKinFitInfo(&asol);
	     if(param_ == 2) asol = myKinFitterEtThetaPhi->addKinFitInfo(&asol);
	     if(param_ == 3) asol = myKinFitterEMom->addKinFitInfo(&asol);
	   }
     /* to be adapted to ST (Andrea)

	   if(addLRJetComb_){
	   asol.setPtrueCombExist(jetCombProbs[m].getPTrueCombExist(&afitsol));
	   asol.setPtrueBJetSel(jetCombProbs[m].getPTrueBJetSel(&afitsol));
	   asol.setPtrueBhadrSel(jetCombProbs[m].getPTrueBhadrSel(&afitsol));
	   asol.setPtrueJetComb(afitsol.getPtrueCombExist()*afitsol.getPtrueBJetSel()*afitsol.getPtrueBhadrSel());
	   }
       */
	   evtsols->push_back(asol);
	 }
       }
     }
     
     // if asked for, match the event solutions to the gen Event
     if(matchToGenEvt_){
       edm::Handle<StGenEvent> genEvt;
       iEvent.getByLabel ("genEvt",genEvt);
       double bestSolDR = 9999.;
       int bestSol = 0;
       for(size_t s=0; s<evtsols->size(); s++) {
         (*evtsols)[s].setGenEvt(genEvt->particles());
         vector<double> bm = BestMatch((*evtsols)[s], false); //false to use DR, true SpaceAngles
         (*evtsols)[s].setSumDeltaRjp(bm[0]); // dRBB + dRLL
         (*evtsols)[s].setChangeBL((int) bm[1]); // has swapped or not
         (*evtsols)[s].setDeltaRB(bm[2]);
         (*evtsols)[s].setDeltaRL(bm[3]);
	 if(bm[0]<bestSolDR) { bestSolDR =  bm[0]; bestSol = s; }
       }
       (*evtsols)[bestSol].setBestSol(true);
     }
     
     //store the vector of solutions to the event     
     auto_ptr<vector<StEvtSolution> > pOut(evtsols);
     iEvent.put(pOut);
   }
   else
   {

     cout<<"@@@ No calibrated solutions built, because:  " << endl;;
     if(lJets->size()<maxJets)   				  cout<<"@ nr light jets = " << lJets->size() << " < " << maxJets <<endl;
     if(bJets->size()<maxJets)   				  cout<<"@ nr b jets = " << bJets->size() << " < " << maxJets <<endl;
     if(leptonFlavour_ == "muon" && !leptonFound)     	          cout<<"@ no good muon candidate"<<endl;
     if(leptonFlavour_ == "electron" && !leptonFound)             cout<<"@ no good electron candidate"<<endl;
     if(mets->size() == 0)    					  cout<<"@ no MET reconstruction"<<endl;

     StEvtSolution asol;
     evtsols->push_back(asol);
     auto_ptr<vector<StEvtSolution> > pOut(evtsols);
     iEvent.put(pOut);
   }
}
