#include "TopQuarkAnalysis/TopEventProducers/interface/TtSemiEvtSolutionMaker.h"

TtSemiEvtSolutionMaker::TtSemiEvtSolutionMaker(const edm::ParameterSet& iConfig)
{
   electronSrc_     = iConfig.getParameter<edm::InputTag> 	("electronSource");
   muonSrc_         = iConfig.getParameter<edm::InputTag> 	("muonSource");
   metSrc_          = iConfig.getParameter<edm::InputTag> 	("metSource");
   lJetSrc_         = iConfig.getParameter<edm::InputTag> 	("lJetSource");
   bJetSrc_         = iConfig.getParameter<edm::InputTag> 	("bJetSource");
   leptonFlavour_   = iConfig.getParameter< std::string > 	("leptonFlavour");
   doKinFit_        = iConfig.getParameter< bool >        	("doKinFit");
   addLRSignalSel_  = iConfig.getParameter< bool >        	("addLRSignalSel");
   lrSignalSelObs_  = iConfig.getParameter< std::vector<int> > 	("lrSignalSelObs");
   lrSignalSelFile_ = iConfig.getParameter< std::string > 	("lrSignalSelFile");
   addLRJetComb_    = iConfig.getParameter< bool >        	("addLRJetComb");
   lrJetCombObs_    = iConfig.getParameter< std::vector<int> > 	("lrJetCombObs");
   lrJetCombFile_   = iConfig.getParameter< std::string > 	("lrJetCombFile");
   maxNrIter_       = iConfig.getParameter< int >         	("maxNrIter");
   maxDeltaS_       = iConfig.getParameter< double >      	("maxDeltaS");
   maxF_            = iConfig.getParameter< double >      	("maxF");
   param_           = iConfig.getParameter< int >         	("parametrisation");
   constraints_     = iConfig.getParameter< std::vector<int> > 	("constraints");
   matchToGenEvt_   = iConfig.getParameter< bool > 	  	("matchToGenEvt");
   
   // define kinfitter
   if(doKinFit_){
     if(param_ == 1) myKinFitterEtEtaPhi   = new TtSemiKinFitterEtEtaPhi(maxNrIter_, maxDeltaS_, maxF_, constraints_);
     if(param_ == 2) myKinFitterEtThetaPhi = new TtSemiKinFitterEtThetaPhi(maxNrIter_, maxDeltaS_, maxF_, constraints_);
     if(param_ == 3) myKinFitterEMom       = new TtSemiKinFitterEMom(maxNrIter_, maxDeltaS_, maxF_, constraints_);
   }
   
   // define jet combinations related calculators
   mySimpleBestJetComb      	         = new TtSemiSimpleBestJetComb();
   myLRSignalSelObservables              = new TtSemiLRSignalSelObservables();
   myLRJetCombObservables                = new TtSemiLRJetCombObservables();
   if(addLRSignalSel_) myLRSignalSelCalc = new TtSemiLRSignalSelCalc(lrSignalSelFile_, lrSignalSelObs_);
   if(addLRJetComb_)   myLRJetCombCalc   = new TtSemiLRJetCombCalc(lrJetCombFile_, lrJetCombObs_);
   produces<std::vector<TtSemiEvtSolution> >();
}

TtSemiEvtSolutionMaker::~TtSemiEvtSolutionMaker()
{
   delete mySimpleBestJetComb;
   delete myLRSignalSelObservables;
   delete myLRJetCombObservables;
   if(addLRSignalSel_) delete myLRSignalSelCalc;
   if(addLRJetComb_)   delete myLRJetCombCalc;
}

void TtSemiEvtSolutionMaker::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
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
   if (lJets->size() >= 4) jetsFound = true;
   
   //
   // Build Event solutions according to the ambiguity in the jet combination
   //
   
   std::vector<TtSemiEvtSolution> *evtsols = new std::vector<TtSemiEvtSolution>();
   if(leptonFound && metFound && jetsFound){
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
		 asol.setHadp((*lJets)[p]);
		 asol.setHadq((*lJets)[q]);
		 asol.setHadb((*bJets)[bh]);
		 asol.setLepb((*bJets)[bl]);
   		 if(doKinFit_){
      		   if(param_ == 1) asol = myKinFitterEtEtaPhi->addKinFitInfo(&asol);
      		   if(param_ == 2) asol = myKinFitterEtThetaPhi->addKinFitInfo(&asol);
      		   if(param_ == 3) asol = myKinFitterEMom->addKinFitInfo(&asol);
		   asol.setJetParametrisation(param_);
		   asol.setLeptonParametrisation(param_);
		   asol.setMETParametrisation(param_);
      		 }
		 // these lines calculate the observables to be used in the TtSemiSignalSelection LR
		 (*myLRSignalSelObservables)(asol);
		 
		 // if asked for, calculate with these observable values the LRvalue and 
		 // (depending on the configuration) probability this event is signal
		 if(addLRSignalSel_) (*myLRSignalSelCalc)(asol);
		 
		 // these lines calculate the observables to be used in the TtSemiJetCombination LR
		 (*myLRJetCombObservables)(asol);
		 
		 // if asked for, calculate with these observable values the LRvalue and 
		 // (depending on the configuration) probability a jet combination is correct
		 if(addLRJetComb_) (*myLRJetCombCalc)(asol);
		 
		 //std::cout<<"SignalSelLRval = "<<asol.getLRSignalEvtLRval()<<"  JetCombProb = "<<asol.getLRSignalEvtProb()<<std::endl;
		 //std::cout<<"JetCombLRval = "<<asol.getLRJetCombLRval()<<"  JetCombProb = "<<asol.getLRJetCombProb()<<std::endl;
		 
		 // fill solution to vector
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
       vector<const reco::Candidate*> quarks;
       const reco::Candidate & genp  = *(genEvt->hadronicQuark());
       const reco::Candidate & genq  = *(genEvt->hadronicQuarkBar());
       const reco::Candidate & genbh = *(genEvt->hadronicB());
       const reco::Candidate & genbl = *(genEvt->leptonicB());
       quarks.push_back( &genp );
       quarks.push_back( &genq );
       quarks.push_back( &genbh );
       quarks.push_back( &genbl );
       vector<const reco::Candidate*> jets;  
       int bestSolution = -999; 
       int bestSolutionChangeWQ = -999;
       for(size_t s=0; s<evtsols->size(); s++) {
         jets.clear();
         const reco::Candidate & jetp  = (*evtsols)[s].getRecHadp();
         const reco::Candidate & jetq  = (*evtsols)[s].getRecHadq();
         const reco::Candidate & jetbh = (*evtsols)[s].getRecHadb();
         const reco::Candidate & jetbl = (*evtsols)[s].getRecLepb();
         jets.push_back( &jetp );
         jets.push_back( &jetq );
         jets.push_back( &jetbh );
         jets.push_back( &jetbl );
         JetPartonMatching aMatch(quarks,jets,1);  // 1: SpaceAngle; 2: DeltaR   
         (*evtsols)[s].setGenEvt(*genEvt);   
         (*evtsols)[s].setMCBestSumAngles(aMatch.getSumAngles());
         (*evtsols)[s].setMCBestAngleHadp(aMatch.getAngleForParton(0));
         (*evtsols)[s].setMCBestAngleHadq(aMatch.getAngleForParton(1));
         (*evtsols)[s].setMCBestAngleHadb(aMatch.getAngleForParton(2));
         (*evtsols)[s].setMCBestAngleLepb(aMatch.getAngleForParton(3));
	 if(aMatch.getMatchForParton(2) == 2 && aMatch.getMatchForParton(3) == 3){
	   if(aMatch.getMatchForParton(0) == 0 && aMatch.getMatchForParton(1) == 1) {
	     bestSolution = s;
	     bestSolutionChangeWQ = 0;
	   } else if(aMatch.getMatchForParton(0) == 1 && aMatch.getMatchForParton(1) == 0) {
	     bestSolution = s;
	     bestSolutionChangeWQ = 1;
	   }
	 }
       }
       for(size_t s=0; s<evtsols->size(); s++) {
         (*evtsols)[s].setMCCorrJetComb(bestSolution);
         (*evtsols)[s].setMCChangeWQ(bestSolutionChangeWQ);     
       }
     }
     
     //store the vector of solutions to the event     
     std::auto_ptr<std::vector<TtSemiEvtSolution> > pOut(evtsols);
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
     std::auto_ptr<std::vector<TtSemiEvtSolution> > pOut(evtsols);
     iEvent.put(pOut);
   }
}
