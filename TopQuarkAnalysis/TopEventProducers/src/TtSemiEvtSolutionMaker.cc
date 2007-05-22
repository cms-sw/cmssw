#include "TopQuarkAnalysis/TopEventProducers/interface/TtSemiEvtSolutionMaker.h"

//
// constructors and destructor
//
TtSemiEvtSolutionMaker::TtSemiEvtSolutionMaker(const edm::ParameterSet& iConfig)
{
   leptonFlavour_   = iConfig.getParameter< string > 	  ("leptonFlavour");
   lJetInput_       = iConfig.getParameter< string > 	  ("lJetInput");
   bJetInput_       = iConfig.getParameter< string > 	  ("bJetInput");
   doKinFit_        = iConfig.getParameter< bool >        ("doKinFit");
   addJetCombProb_  = iConfig.getParameter< bool >        ("addJetCombProb");
   maxNrIter_       = iConfig.getParameter< int >         ("maxNrIter");
   maxDeltaS_       = iConfig.getParameter< double >      ("maxDeltaS");
   maxF_            = iConfig.getParameter< double >      ("maxF");
   param_           = iConfig.getParameter< int >         ("parametrisation");
   constraints_     = iConfig.getParameter< vector<int> > ("constraints");
   matchToGenEvt_   = iConfig.getParameter< bool > 	  ("matchToGenEvt");
   
   // define kinfitter
   if(doKinFit_){
     if(param_ == 1) myKinFitterEtEtaPhi   = new TtSemiKinFitterEtEtaPhi(maxNrIter_, maxDeltaS_, maxF_, constraints_);
     if(param_ == 2) myKinFitterEtThetaPhi = new TtSemiKinFitterEtThetaPhi(maxNrIter_, maxDeltaS_, maxF_, constraints_);
     if(param_ == 3) myKinFitterEMom       = new TtSemiKinFitterEMom(maxNrIter_, maxDeltaS_, maxF_, constraints_);
   }
   
   // define jet combinations related calculators
   mySimpleBestJetComb = new TtSemiSimpleBestJetComb();
   
   produces<vector<TtSemiEvtSolution> >();
}


TtSemiEvtSolutionMaker::~TtSemiEvtSolutionMaker() {}


//
// member functions
//

// ------------ method called to produce the data  ------------
void TtSemiEvtSolutionMaker::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   edm::Handle<vector<TopMuon> >  muons;
   if(leptonFlavour_ == "muon") iEvent.getByType(muons);
   edm::Handle<vector<TopElectron> >  electrons;
   if(leptonFlavour_ == "electron") iEvent.getByType(electrons);
   edm::Handle<vector<TopMET> >  mets;
   iEvent.getByType(mets);
   edm::Handle<vector<TopJet> >  lJets;
   iEvent.getByLabel(lJetInput_,lJets);
   edm::Handle<vector<TopJet> >  bJets;
   iEvent.getByLabel(bJetInput_,bJets);

   //select lepton (the TtLepton vectors are, for the moment, sorted on pT)
   TopMuon selMuon;
   TopElectron selElectron;
   bool leptonFound = false;
   if(leptonFlavour_ == "muon"     &&     muons->size()>=1) { selMuon     = (*muons)[0];     leptonFound = true;};    
   if(leptonFlavour_ == "electron" && electrons->size()>=1) { selElectron = (*electrons)[0]; leptonFound = true;};  

   //select MET (TopMET vector is sorted on ET)
   TopMET selMET;
   bool METFound = false;
   if(mets -> size()>=1) { selMET = (*mets)[0]; METFound = true;};  

   //select Jets (TopJet vector is sorted on recET, so four first elements in both the lJets and bJets vector are the same )
   vector<TopJet> lSelJets, bSelJets;
   bool jetsFound = false;
   if(lJets -> size()>=4 && bJets -> size()>=4 ) { 
     for(int j=0; j<4; j++) {
       lSelJets.push_back((*lJets)[j]);
       bSelJets.push_back((*bJets)[j]);
     }
     jetsFound = true;
   }
   
   vector<TtSemiEvtSolution> *evtsols = new vector<TtSemiEvtSolution>();
   if(leptonFound && METFound && jetsFound){
     //cout<<"constructing solutions"<<endl;
     for (unsigned int p=0; p<4; p++) {
       for (unsigned int q=0; q<4; q++) {
         for (unsigned int bh=0; bh<4; bh++) {
           if(q>p && !(bh==p || bh==q)){
             for (unsigned int bl=0; bl<4; bl++) {
	       if(!(bl==p || bl==q || bl==bh)){
                 TtSemiEvtSolution asol;
		 if(leptonFlavour_ == "muon")     asol.setMuon(selMuon);
		 if(leptonFlavour_ == "electron") asol.setElectron(selElectron);
		 asol.setMET(selMET);
		 asol.setHadp(lSelJets[p]);
		 asol.setHadq(lSelJets[q]);
		 asol.setHadb(bSelJets[bh]);
		 asol.setLepb(bSelJets[bl]);
   		 if(doKinFit_){
      		   if(param_ == 1) asol = myKinFitterEtEtaPhi->addKinFitInfo(&asol);
      		   if(param_ == 2) asol = myKinFitterEtThetaPhi->addKinFitInfo(&asol);
      		   if(param_ == 3) asol = myKinFitterEMom->addKinFitInfo(&asol);
		   asol.setJetParametrisation(param_);
		   asol.setLeptonParametrisation(param_);
		   asol.setMETParametrisation(param_);
      		 }
		/*if(addJetCombProb_){
      		  asol.setPtrueCombExist(jetCombProbs[m].getPTrueCombExist(&afitsol));
      		  asol.setPtrueBJetSel(jetCombProbs[m].getPTrueBJetSel(&afitsol));
      		  asol.setPtrueBhadrSel(jetCombProbs[m].getPTrueBhadrSel(&afitsol));
      		  asol.setPtrueJetComb(afitsol.getPtrueCombExist()*afitsol.getPtrueBJetSel()*afitsol.getPtrueBhadrSel());
      		 }*/
	         evtsols->push_back(asol);
	       } 
	     }
	   }
         } 
       }
     }
     
     // add TtSemiSimpleBestJetComb to solutions
     int simpleBestJetComb = (*mySimpleBestJetComb)(*evtsols);
     for(size_t s=0; s<evtsols->size(); s++) (*evtsols)[s].setSimpleBestSol(simpleBestJetComb);
     
     // if asked for, match the event solutions to the gen Event
     if(matchToGenEvt_){
       edm::Handle<TtGenEvent> genEvt;
       iEvent.getByLabel ("genEvt",genEvt);
       double bestSolDR = 9999.;
       int bestSol = 0;
       for(size_t s=0; s<evtsols->size(); s++) {
         (*evtsols)[s].setGenEvt(genEvt->particles());
         vector<double> bm = BestMatch((*evtsols)[s], false); //false to use DR, true SpaceAngles
         (*evtsols)[s].setSumDeltaRjp(bm[0]);
         (*evtsols)[s].setChangeWQ((int) bm[1]);
         (*evtsols)[s].setDeltaRhadp(bm[2]);
         (*evtsols)[s].setDeltaRhadq(bm[3]);
         (*evtsols)[s].setDeltaRhadb(bm[4]);
         (*evtsols)[s].setDeltaRlepb(bm[5]);
	 if(bm[0]<bestSolDR) { bestSolDR =  bm[0]; bestSol = s; }
       }
       for(size_t s=0; s<evtsols->size(); s++) (*evtsols)[s].setMCBestSol(bestSol);
     }
     
     
     //store the vector of solutions to the event     
     auto_ptr<vector<TtSemiEvtSolution> > pOut(evtsols);
     iEvent.put(pOut);
   }
   else
   {
     /*
     cout<<"No calibrated solutions build, because:  ";
     if(jets->size()<4)      					  cout<<"nr sel jets < 4"<<endl;
     if(leptonFlavour_ == "muon" && muons->size() == 0)    	  cout<<"no good muon candidate"<<endl;
     if(leptonFlavour_ == "electron" && electrons->size() == 0)   cout<<"no good electron candidate"<<endl;
     if(mets->size() == 0)    					  cout<<"no MET reconstruction"<<endl;
     */
     TtSemiEvtSolution asol;
     evtsols->push_back(asol);
     auto_ptr<vector<TtSemiEvtSolution> > pOut(evtsols);
     iEvent.put(pOut);
   }
}
