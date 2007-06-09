#include "TopQuarkAnalysis/TopEventProducers/interface/TtDilepEvtSolutionMaker.h"


//
// constructors and destructor
//
TtDilepEvtSolutionMaker::TtDilepEvtSolutionMaker(const edm::ParameterSet& iConfig)
{
  jetInput_        = iConfig.getParameter< std::string >  ("jetInput");
  matchToGenEvt_   = iConfig.getParameter< bool > 	  ("matchToGenEvt");
  calcTopMass_     = iConfig.getParameter< bool >         ("calcTopMass"); 
  eeChannel_     = iConfig.getParameter< bool >           ("eeChannel"); 
  emuChannel_     = iConfig.getParameter< bool >          ("emuChannel");
  mumuChannel_     = iConfig.getParameter< bool >         ("mumuChannel");
  tmassbegin_     = iConfig.getParameter< double >         ("tmassbegin");
  tmassend_     = iConfig.getParameter< double >         ("tmassend");
  tmassstep_     = iConfig.getParameter< double >         ("tmassstep");
  
  produces<std::vector<TtDilepEvtSolution> >();
}


TtDilepEvtSolutionMaker::~TtDilepEvtSolutionMaker()
{
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void TtDilepEvtSolutionMaker::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

   using namespace edm;
   Handle<std::vector<TopMuon> >  muons;
   iEvent.getByType(muons);
   Handle<std::vector<TopElectron> >  electrons;
   iEvent.getByType(electrons);
   Handle<std::vector<TopMET> >  mets;
   iEvent.getByType(mets);
   Handle<std::vector<TopJet> >  jets;
   iEvent.getByLabel(jetInput_,jets);
   
   //select lepton (the TtLepton vectors are, for the moment, sorted on pT)
   TopMuon selMuonp;
   TopMuon selMuonm;
   TopElectron selElectronp;
   TopElectron selElectronm;
   bool leptonFound = false;
   
   bool mumu = false;
   bool emu = false;
   bool ee = false;
   
   if (muons->size() + electrons->size() >=2) {
     if (electrons->size() == 0) mumu = true;
     
     else if (muons->size() == 0) ee = true;
     
     else if (electrons->size() == 1) {
       if (muons->size() == 1) emu = true;
       else if (PTComp((*electrons)[0], (*muons)[1])) emu = true;
       else  mumu = true;
     }
     
     else if (electrons->size() > 1) {
       if (PTComp((*electrons)[1], (*muons)[0])) ee = true;
       else if (muons->size() == 1) emu = true;
       else if (PTComp((*electrons)[0], (*muons)[1])) emu = true;
       else mumu = true;
     }
  }
  
  if ((ee && emu) || (ee && mumu) || (emu && mumu))
    std::cout << "[TtDilepEvtSolutionMaker]: "
         << "Lepton selection criteria uncorrectly defined" << std::endl;
  
  bool leptonFoundEE = false;
  bool leptonFoundMM = false;
  bool leptonFoundEpMm = false;
  bool leptonFoundEmMp = false;
  if (ee) {
    if (LepDiffCharge((*electrons)[0], (*electrons)[1])) {
      leptonFound = true;
      leptonFoundEE = true;
      if (HasPositiveCharge((*electrons)[0])) {
        selElectronp = (*electrons)[0];
	selElectronm = (*electrons)[1];
      }
      else {
        selElectronp = (*electrons)[1];
	selElectronm = (*electrons)[0];
      }
    }
  }
  
  else if (emu) {
    if (LepDiffCharge((*electrons)[0], (*muons)[0])) {
      leptonFound = true;
      if (HasPositiveCharge((*electrons)[0])) {
        leptonFoundEpMm = true;
        selElectronp = (*electrons)[0];
	selMuonm = (*muons)[0];
      }
      else {
        leptonFoundEmMp = true;
        selMuonp = (*muons)[0];
	selElectronm = (*electrons)[0];
      }
    }
  }
  
  else if (mumu) {
    if (LepDiffCharge((*muons)[0], (*muons)[1])) {
      leptonFound = true;
      leptonFoundMM = true;
      if (HasPositiveCharge((*muons)[0])) {
        selMuonp = (*muons)[0];
	selMuonm = (*muons)[1];
      }
      else {
        selMuonp = (*muons)[1];
	selMuonm = (*muons)[0];
      }
    }
  }
   
   //select MET (TopMET vector is sorted on ET)
   TopMET selMET;
   bool METFound = false;
   if(mets -> size()>=1) { selMET = (*mets)[0]; METFound = true;};  

   //select Jets (TopJet vector is sorted on ET)
   std::vector<TopJet> selJets;
   bool jetsFound = false;
   if(jets -> size()>=2) {
     for(int j=0; j<2; j++) selJets.push_back((*jets)[j]);
     jetsFound = true;
   }
   
   bool correctLepton = (leptonFoundEE && eeChannel_) ||
                        ((leptonFoundEmMp || leptonFoundEpMm) && emuChannel_) ||
			(leptonFoundMM && mumuChannel_);
			
   std::vector<TtDilepEvtSolution> *evtsols = new std::vector<TtDilepEvtSolution>();
   if(correctLepton && METFound && jetsFound){
     //cout<<"constructing solutions"<<endl;
     
     //SaveSolution for both jet-lep pairings
     for (unsigned int ib = 0; ib < 2; ib++) {
       TtDilepEvtSolution asol;
       
       double xconstraint = 0, yconstraint = 0;
       if (leptonFoundEE || leptonFoundEpMm) {
         asol.setElectronLepp(selElectronp);
         xconstraint += selElectronp.px();
	 yconstraint += selElectronp.py();
       }
       if (leptonFoundEE || leptonFoundEmMp) {
         asol.setElectronLepm(selElectronm);
         xconstraint += selElectronm.px();
	 yconstraint += selElectronm.py();
       }
       if (leptonFoundMM || leptonFoundEmMp) {
         asol.setMuonLepp(selMuonp);
         xconstraint += selMuonp.px();
	 yconstraint += selMuonp.py();
       }
       if (leptonFoundMM || leptonFoundEpMm) {
         asol.setMuonLepm(selMuonm);
         xconstraint += selMuonm.px();
	 yconstraint += selMuonm.py();
       }
       
       if (ib == 0) {asol.setB(selJets[0]); asol.setBbar(selJets[1]);}
       if (ib == 1) {asol.setB(selJets[1]); asol.setBbar(selJets[0]);}
       asol.setMET(selMET);
       xconstraint += selJets[0].px() + selJets[1].px() +
                      selMET.px();
       yconstraint += selJets[0].py() + selJets[1].py() +
                      selMET.py();
       
       if (calcTopMass_) {
         TtDilepKinSolver solver(tmassbegin_, tmassend_, tmassstep_, xconstraint, yconstraint);
         asol = solver.addKinSolInfo(&asol);
       }
       
       evtsols->push_back(asol);
     }
     
     // if asked for, match the event solutions to the gen Event
     if(matchToGenEvt_){
       Handle<TtGenEvent> genEvt;
       iEvent.getByLabel ("genEvt",genEvt);
       double bestSolDR = 9999.;
       int bestSol = 0;
       for(size_t s=0; s<evtsols->size(); s++) {
         (*evtsols)[s].setGenEvt(genEvt->particles());
         //FIXME probably this should be moved to BestMatching.h
         double sqrDRBB = pow((*evtsols)[s].getCalJetB().eta() -
	                      (*evtsols)[s].getGenB().eta(), 2) +
	                  pow((*evtsols)[s].getCalJetB().phi() -
			      (*evtsols)[s].getGenB().phi(), 2);
	 double sqrDRBbarBbar = pow((*evtsols)[s].getCalJetBbar().eta() -
	                            (*evtsols)[s].getGenBbar().eta(), 2) +
	                        pow((*evtsols)[s].getCalJetBbar().phi() -
				    (*evtsols)[s].getGenBbar().phi(), 2);
	 double SolDR = sqrt(sqrDRBB) + sqrt(sqrDRBbarBbar);
	 if (SolDR < bestSolDR) { bestSolDR =  SolDR; bestSol = s; }
       }
       (*evtsols)[bestSol].setBestSol(true);
     }
     
     std::auto_ptr<std::vector<TtDilepEvtSolution> > pOut(evtsols);
     iEvent.put(pOut);
   }
   else {
     TtDilepEvtSolution asol;
     evtsols->push_back(asol);
     std::auto_ptr<std::vector<TtDilepEvtSolution> > pOut(evtsols);
     iEvent.put(pOut);
   }
}

bool TtDilepEvtSolutionMaker::PTComp(TopElectron e, TopMuon m) {
  if (e.pt() > m.pt()) return true;
  else return false;
}

bool TtDilepEvtSolutionMaker::LepDiffCharge(TopElectron e, TopMuon m) {
  if (e.charge() != m.charge()) return true;
  else return false;
}

bool TtDilepEvtSolutionMaker::LepDiffCharge(TopElectron e1, TopElectron e2) {
  if (e1.charge() != e2.charge()) return true;
  else return false;
}

bool TtDilepEvtSolutionMaker::LepDiffCharge(TopMuon m1, TopMuon m2) {
  if (m1.charge() != m2.charge()) return true;
  else return false;
}

bool TtDilepEvtSolutionMaker::HasPositiveCharge(TopMuon m) {
  if (m.charge() > 0) return true;
  else return false;
}

bool TtDilepEvtSolutionMaker::HasPositiveCharge(TopElectron e) {
  if (e.charge() > 0) return true;
  else return false;
}
