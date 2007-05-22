#include "TopQuarkAnalysis/TopEventProducers/interface/StEvtSolutionMaker.h"

//
// constructors and destructor
//
StEvtSolutionMaker::StEvtSolutionMaker(const edm::ParameterSet& iConfig)
{
   leptonFlavour_   = iConfig.getParameter< string > 	  ("leptonFlavour");
   jetInput_        = iConfig.getParameter< string > 	  ("jetInput");
   doKinFit_        = iConfig.getParameter< bool >        ("doKinFit");
   addJetCombProb_  = iConfig.getParameter< bool >        ("addJetCombProb");
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
   edm::Handle<vector<TopMuon> >  muons;
   if(leptonFlavour_ == "muon") iEvent.getByType(muons);
   edm::Handle<vector<TopElectron> >  electrons;
   if(leptonFlavour_ == "electron") iEvent.getByType(electrons);
   edm::Handle<vector<TopMET> >  mets;
   iEvent.getByType(mets);
   edm::Handle<vector<TopJet> >  jets;
   iEvent.getByLabel(jetInput_,jets);

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

   //select Jets (TopJet vector is sorted on ET)
   vector<TopJet> selJets;
   bool jetsFound = false;
   unsigned int maxJets=2;//this has to become a custom-defined parameter (we may want 2 or 3 jets)
   if(jets -> size()>=maxJets) { 
     for(unsigned int j=0; j<maxJets; j++) selJets.push_back((*jets)[j]);
     jetsFound = true;
   }
   
   vector<StEvtSolution> *evtsols = new vector<StEvtSolution>();
   if(leptonFound && METFound && jetsFound){
     cout<<"constructing solutions"<<endl;
     for (unsigned int b=0; b<maxJets; b++) {
       for (unsigned int l=0; l<maxJets; l++) {
	 if(b!=l){  // to avoid double counting
	   StEvtSolution asol;
	   if(leptonFlavour_ == "muon")     asol.setMuon(selMuon);
	   if(leptonFlavour_ == "electron") asol.setElectron(selElectron);
	   asol.setMET(selMET);
	   asol.setBottom(selJets[b]);
	   asol.setLight(selJets[l]);
	   if(doKinFit_){
	     if(param_ == 1) asol = myKinFitterEtEtaPhi->addKinFitInfo(&asol);
	     if(param_ == 2) asol = myKinFitterEtThetaPhi->addKinFitInfo(&asol);
	     if(param_ == 3) asol = myKinFitterEMom->addKinFitInfo(&asol);
	   }
     /* to be adapted to ST (Andrea)

	   if(addJetCombProb_){
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
     if(jets->size()<maxJets)      					  cout<<"@ nr sel jets = " << jets->size() << " < " << maxJets <<endl;
     if(leptonFlavour_ == "muon" && muons->size() == 0)    	  cout<<"@ no good muon candidate"<<endl;
     if(leptonFlavour_ == "electron" && electrons->size() == 0)   cout<<"@ no good electron candidate"<<endl;
     if(mets->size() == 0)    					  cout<<"@ no MET reconstruction"<<endl;

     StEvtSolution asol;
     evtsols->push_back(asol);
     auto_ptr<vector<StEvtSolution> > pOut(evtsols);
     iEvent.put(pOut);
   }
}
