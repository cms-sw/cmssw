#include "TopQuarkAnalysis/TopEventProducers/interface/TtSemiEvtSolutionMaker.h"

//
// constructors and destructor
//
TtSemiEvtSolutionMaker::TtSemiEvtSolutionMaker(const edm::ParameterSet& iConfig)
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
   
   // define kinfitter
   if(doKinFit_){
     if(param_ == 1) myKinFitterEtEtaPhi   = new TtSemiKinFitterEtEtaPhi(maxNrIter_, maxDeltaS_, maxF_, constraints_);
     if(param_ == 2) myKinFitterEtThetaPhi = new TtSemiKinFitterEtThetaPhi(maxNrIter_, maxDeltaS_, maxF_, constraints_);
     if(param_ == 3) myKinFitterEMom       = new TtSemiKinFitterEMom(maxNrIter_, maxDeltaS_, maxF_, constraints_);
   }
   
   // define jet combinations related calculators
   mySimpleBestJetComb = new TtSemiSimpleBestJetComb();
   goodEvts = 0;
   goodEvtsFound = 0;
   
   produces<vector<TtSemiEvtSolution> >();
}


TtSemiEvtSolutionMaker::~TtSemiEvtSolutionMaker()
{
  cout<<"Total of good events:"<< goodEvts <<endl;
  cout<<"  of wich are found :"<< goodEvtsFound <<"   ("<<(goodEvtsFound*100.)/(goodEvts*1.)<<"%)"<<endl;
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void TtSemiEvtSolutionMaker::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   Handle<vector<TopMuonObject> >  muons;
   if(leptonFlavour_ == "muon") iEvent.getByType(muons);
   Handle<vector<TopElectronObject> >  electrons;
   if(leptonFlavour_ == "electron") iEvent.getByType(electrons);
   Handle<vector<TopMETObject> >  mets;
   iEvent.getByType(mets);
   Handle<vector<TopJetObject> >  jets;
   iEvent.getByLabel(jetInput_,jets);

   //select lepton (the TtLepton vectors are, for the moment, sorted on pT)
   TopMuonObject selMuon;
   TopElectronObject selElectron;
   bool leptonFound = false;
   if(leptonFlavour_ == "muon"     &&     muons->size()>=1) { selMuon     = (*muons)[0];     leptonFound = true;};    
   if(leptonFlavour_ == "electron" && electrons->size()>=1) { selElectron = (*electrons)[0]; leptonFound = true;};  

   //select MET (TopMET vector is sorted on ET)
   TopMETObject selMET;
   bool METFound = false;
   if(mets -> size()>=1) { selMET = (*mets)[0]; METFound = true;};  

   //select Jets (TopJet vector is sorted on ET)
   vector<TopJetObject> selJets;
   bool jetsFound = false;
   if(jets -> size()>=4) { 
     for(int j=0; j<4; j++) selJets.push_back((*jets)[j]);
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
		 asol.setHadp(selJets[p]);
		 asol.setHadq(selJets[q]);
		 asol.setHadb(selJets[bh]);
		 asol.setLepb(selJets[bl]);
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
     
     // if asked for, match the event solutions to the gen Event
     if(matchToGenEvt_){
       Handle<TtGenEvent> genEvt;
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
       (*evtsols)[bestSol].setBestSol(true);
       
       if(bestSolDR<0.5) ++goodEvts;
       if(bestSolDR<0.5 && (simpleBestJetComb == bestSol)) ++goodEvtsFound;
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
