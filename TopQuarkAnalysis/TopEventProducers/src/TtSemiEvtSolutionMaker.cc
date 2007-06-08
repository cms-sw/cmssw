#include "TopQuarkAnalysis/TopEventProducers/interface/TtSemiEvtSolutionMaker.h"

//
// constructors
//

TtSemiEvtSolutionMaker::TtSemiEvtSolutionMaker(const edm::ParameterSet& iConfig)
{
   leptonFlavour_   = iConfig.getParameter< std::string > 	  ("leptonFlavour");
   lJetInput_       = iConfig.getParameter< std::string > 	  ("lJetInput");
   bJetInput_       = iConfig.getParameter< std::string > 	  ("bJetInput");
   jetEtaCut_  	    = iConfig.getParameter< double >      ("jetEtaCut");	
   recJetETCut_     = iConfig.getParameter< double >      ("recJetETCut");	
   calJetETCut_     = iConfig.getParameter< double >      ("calJetETCut");	
   jetLRCut_        = iConfig.getParameter< double >      ("jetLRCut");
   doKinFit_        = iConfig.getParameter< bool >        ("doKinFit");
   addLRJetComb_    = iConfig.getParameter< bool >        ("addLRJetComb");
   lrJetCombFile_   = iConfig.getParameter< std::string >      ("lrJetCombFile");
   maxNrIter_       = iConfig.getParameter< int >         ("maxNrIter");
   maxDeltaS_       = iConfig.getParameter< double >      ("maxDeltaS");
   maxF_            = iConfig.getParameter< double >      ("maxF");
   param_           = iConfig.getParameter< int >         ("parametrisation");
   constraints_     = iConfig.getParameter< std::vector<int> > ("constraints");
   matchToGenEvt_   = iConfig.getParameter< bool > 	  ("matchToGenEvt");
   
   // define kinematic selection cuts
   jetEtaRangeSelector      = new EtaRangeSelector<TopJet>(-1.*jetEtaCut_,jetEtaCut_);
   recJetEtMinSelector      = new EtMinSelector<TopJet>(recJetETCut_);
   calJetEtMinSelector      = new EtMinSelector<TopJet>(calJetETCut_);
   
   // define kinfitter
   if(doKinFit_){
     if(param_ == 1) myKinFitterEtEtaPhi   = new TtSemiKinFitterEtEtaPhi(maxNrIter_, maxDeltaS_, maxF_, constraints_);
     if(param_ == 2) myKinFitterEtThetaPhi = new TtSemiKinFitterEtThetaPhi(maxNrIter_, maxDeltaS_, maxF_, constraints_);
     if(param_ == 3) myKinFitterEMom       = new TtSemiKinFitterEMom(maxNrIter_, maxDeltaS_, maxF_, constraints_);
   }
   
   // define jet combinations related calculators
   mySimpleBestJetComb      	     = new TtSemiSimpleBestJetComb();
   myTtSemiLRSignalSelObservables    = new TtSemiLRSignalSelObservables();
   myLRJetCombObservables            = new TtSemiLRJetCombObservables();
   if(addLRJetComb_) myLRJetCombCalc = new TtSemiLRJetCombCalc(lrJetCombFile_);
   produces<std::vector<TtSemiEvtSolution> >();
}








//
// destructor
//

TtSemiEvtSolutionMaker::~TtSemiEvtSolutionMaker() {
   delete recJetEtMinSelector;
   delete calJetEtMinSelector;
   delete jetEtaRangeSelector;
   delete mySimpleBestJetComb;
   delete myTtSemiLRSignalSelObservables;
   delete myLRJetCombObservables;
   if(addLRJetComb_) delete myLRJetCombCalc;
}








//
// member functions
//

// ------------ method called to produce the data  ------------
void TtSemiEvtSolutionMaker::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  //
  //  TopObject Selection
  //

   // select lepton (the TtLepton vectors are, for the moment, sorted on pT)
   bool leptonFound = false;
   TopMuon selMuon;
   if(leptonFlavour_ == "muon"){
     edm::Handle<std::vector<TopMuon> >  muons;
     iEvent.getByType(muons);
     std::vector<TopMuon> selMuons;
     for(size_t m=0; m < muons->size(); m++) { 
       selMuons.push_back((*muons)[m]);
     }
     if( selMuons.size() > 0 ){
       //select highest pT muon
       selMuon = selMuons[0];
       leptonFound = true;
     }    
   }  
   TopElectron selElectron;
   if(leptonFlavour_ == "electron"){
     edm::Handle<std::vector<TopElectron> >  electrons;
     iEvent.getByType(electrons);
     std::vector<TopElectron> selElectrons;
     for(size_t e=0; e < electrons->size(); e++) { 
       selElectrons.push_back((*electrons)[e]);
     }
     if( selElectrons.size() > 0 ){
       //select highest pT electron
       selElectron = selElectrons[0];
       leptonFound = true;
     }       
   }  

   // select MET (TopMET vector is sorted on ET)
   bool METFound = false;
   TopMET selMET;
   edm::Handle<std::vector<TopMET> >  METs;
   iEvent.getByType(METs);
   std::vector<TopMET> selMETs;
   for(size_t m=0; m < METs->size(); m++) { 
     selMETs.push_back((*METs)[m]);
   }
   if( selMETs.size() > 0 ){
     //select highest Et MET
     selMET = selMETs[0];
     METFound = true;
   }

   // select Jets (TopJet vector is sorted on recET, so four first elements in both the lJets and bJets vector are the same )
   bool jetsFound = false;
   std::vector<TopJet> lSelJets, bSelJets;
   edm::Handle<std::vector<TopJet> >  lJets;
   iEvent.getByLabel(lJetInput_,lJets);
   edm::Handle<std::vector<TopJet> >  bJets;
   iEvent.getByLabel(bJetInput_,bJets);
   // vectors were sorted on recET
   for(size_t j=0; j < min(lJets -> size(),bJets -> size()); j++) {
     if( (*recJetEtMinSelector)((*lJets)[j].getRecJet()) && (*jetEtaRangeSelector)((*lJets)[j].getRecJet())
      && (*calJetEtMinSelector)((*lJets)[j])             && (*calJetEtMinSelector)((*bJets)[j]) 
      && true){
       lSelJets.push_back((*lJets)[j]);
       bSelJets.push_back((*bJets)[j]);
     }
   }
   if( lSelJets.size() >= 4 ) jetsFound = true;
   
   
   
   
   
   //
   // Build Event solutions according to the ambiguity in the jet combination
   //
   
   std::vector<TtSemiEvtSolution> *evtsols = new std::vector<TtSemiEvtSolution>();
   if(leptonFound && METFound && jetsFound){
     //std::cout<<"constructing solutions"<<std::endl;
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
		 // these lines calculate the observables to be used in the TtSemiSignalSelection LR
		 (*myTtSemiLRSignalSelObservables)(asol);
		 
		 // these lines calculate the observables to be used in the TtSemiJetCombination LR
		 (*myLRJetCombObservables)(asol);
		 
		 //if asked for, calculate with these observable values the LRvalue and probability a jet combination is correct
		 if(addLRJetComb_) (*myLRJetCombCalc)(asol);
	         evtsols->push_back(asol);
	       } 
	     }
	   }
         } 
       }
     }
     
     // add TtSemiSimpleBestJetComb to solutions
     int simpleBestJetComb = (*mySimpleBestJetComb)(*evtsols);
     for(size_t s=0; s<evtsols->size(); s++) (*evtsols)[s].setSimpleCorrJetComb(simpleBestJetComb);
     
     // if asked for, match the event solutions to the gen Event
     if(matchToGenEvt_){
       edm::Handle<TtGenEvent> genEvt;
       iEvent.getByLabel ("genEvt",genEvt);
       double bestSolDR = 9999.;
       int bestSol = 0;
       for(size_t s=0; s<evtsols->size(); s++) {
         (*evtsols)[s].setGenEvt(genEvt->particles());
         std::vector<double> bm = BestMatch((*evtsols)[s], false); //false to use DR, true SpaceAngles
         (*evtsols)[s].setSumDeltaRjp(bm[0]);
         (*evtsols)[s].setChangeWQ((int) bm[1]);
         (*evtsols)[s].setDeltaRhadp(bm[2]);
         (*evtsols)[s].setDeltaRhadq(bm[3]);
         (*evtsols)[s].setDeltaRhadb(bm[4]);
         (*evtsols)[s].setDeltaRlepb(bm[5]);
	 if(bm[0]<bestSolDR) { bestSolDR =  bm[0]; bestSol = s; }
       }
       for(size_t s=0; s<evtsols->size(); s++) (*evtsols)[s].setMCCorrJetComb(bestSol);
     }
     
     
     //store the vector of solutions to the event     
     auto_ptr<std::vector<TtSemiEvtSolution> > pOut(evtsols);
     iEvent.put(pOut);
   }
   else
   {
     /*
     std::cout<<"No calibrated solutions build, because:  ";
     if(jets->size()<4)      					  std::cout<<"nr sel jets < 4"<<std::endl;
     if(leptonFlavour_ == "muon" && muons->size() == 0)    	  std::cout<<"no good muon candidate"<<std::endl;
     if(leptonFlavour_ == "electron" && electrons->size() == 0)   std::cout<<"no good electron candidate"<<std::endl;
     if(mets->size() == 0)    					  std::cout<<"no MET reconstruction"<<std::endl;
     */
     TtSemiEvtSolution asol;
     evtsols->push_back(asol);
     auto_ptr<std::vector<TtSemiEvtSolution> > pOut(evtsols);
     iEvent.put(pOut);
   }
}
