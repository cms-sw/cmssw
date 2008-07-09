#include "TopQuarkAnalysis/Examples/interface/TtSemiEvtKit.h"

#include "PhysicsTools/CandUtils/interface/AddFourMomenta.h"
#include "DataFormats/Math/interface/LorentzVector.h"

using namespace std;
using namespace pat;

//
// constructors and destructor
//
TtSemiEvtKit::TtSemiEvtKit(const edm::ParameterSet& iConfig) 
  :
  verboseLevel_(0),
  helper_(iConfig)
{

  helper_.bookHistos(this);

  evtsols           = iConfig.getParameter<edm::InputTag> ("EvtSolution");

  cout << "About to book histoTtSemiEvtHypothesis" << endl;


  PhysicsHistograms::KinAxisLimits compositeAxisLimits;

  compositeAxisLimits = helper_.getAxisLimits("topAxis");

  double pt1 = compositeAxisLimits.pt1;
  double pt2 = compositeAxisLimits.pt2;
  double m1  = compositeAxisLimits.m1;
  double m2  = compositeAxisLimits.m2;

  histoTtSemiEvt_ = new HistoComposite("ttSemiEvt", "ttSemiEvt", "ttSemiEvt",
				       pt1, pt2, m1, m2);


  edm::Service<TFileService> fs;
  TFileDirectory ttbar = TFileDirectory( fs->mkdir("ttbar") );

  histoLRJetCombProb_ = new PhysVarHisto( "lrJetCombProb", "Jet Comb Probability",
					  100, 0, 1, &ttbar, "", "vD" );
  histoLRSignalEvtProb_ = new PhysVarHisto( "lrSignalEvtProb", "Event Probability",
					  100, 0, 1, &ttbar, "", "vD" );
  histoKinFitProbChi2_ = new PhysVarHisto( "kinFitProbChi2", "Kin Fitter Chi2 Prob",
					  100, 0, 1, &ttbar, "", "vD" );


  histoLRJetCombProb_ ->makeTH1();
  histoLRSignalEvtProb_ ->makeTH1();
  histoKinFitProbChi2_ ->makeTH1();

  helper_.physHistos_->addHisto( histoLRJetCombProb_ );
  helper_.physHistos_->addHisto( histoLRSignalEvtProb_ );
  helper_.physHistos_->addHisto( histoKinFitProbChi2_ );
  
}

TtSemiEvtKit::~TtSemiEvtKit() 
{
}

//
// member functions
//

// ------------ method called to for each event  ------------
void TtSemiEvtKit::produce( edm::Event& iEvent, const edm::EventSetup& iSetup) 
{
  using namespace edm;

  // INSIDE OF LepJetMetKit::produce:

  // --------------------------------------------------
  //    Step 1: Retrieve objects from data stream
  // --------------------------------------------------
  helper_.getHandles( iEvent,
		      muonHandle_,
		      electronHandle_,
		      tauHandle_,
		      jetHandle_,
		      METHandle_,
		      photonHandle_);

  // --------------------------------------------------
  //    Step 2: invoke PhysicsHistograms to deal with all this.
  //
  //    Note that each handle will dereference into a vector<>,
  //    however the fillCollection() method takes a reference,
  //    so the collections are not copied...
  // --------------------------------------------------

  if ( verboseLevel_ > 10 )
    std::cout << "PatAnalyzerKit::analyze: calling fillCollection()." << std::endl;
  helper_.fillHistograms( iEvent,
			  muonHandle_,
			  electronHandle_,
			  tauHandle_,
			  jetHandle_,
			  METHandle_,
			  photonHandle_);

  // --------------------------------------------------
  //    Step 3: Plot LepJetMet data
  // --------------------------------------------------


  // BEGIN TtSemiEvt analysis here:

   // get the event solution
   edm::Handle< std::vector<TtSemiEvtSolution> > eSols; 
   iEvent.getByLabel(evtsols, eSols);

//    cout << "TtSemiEvtKit: About to do work on sols" << endl;
   const std::vector<TtSemiEvtSolution> & sols = *eSols;
//    cout << "Done getting vector ref to sols" << endl;
  
   if ( sols.size() > 0 ) {

//      cout << "Sols.size() > 0 " << endl;

//      cout << "TtSemiEvtKit: Getting best solution" << endl;
     int bestSol = sols[0].getLRBestJetComb();   
     if ( bestSol >= 0 ) {
       
     
//      cout << "About to fill the ttSemiEvt solution : " << bestSol << endl;
       histoTtSemiEvt_->fill( sols[bestSol].getRecoHyp() );

     
       histoLRJetCombProb_->fill( sols[bestSol].getLRJetCombProb());
       histoLRSignalEvtProb_->fill( sols[bestSol].getLRSignalEvtProb());
       histoKinFitProbChi2_->fill( sols[bestSol].getProbChi2());
     }
   }


   histoLRJetCombProb_->clearVec();
   histoLRSignalEvtProb_->clearVec();
   histoKinFitProbChi2_->clearVec();
   
  
   // cout << "Done with produce" << endl;
}


// ------------ method called once each job just before starting event loop  ------------
void
TtSemiEvtKit::beginJob(const edm::EventSetup& iSetup)
{
}



// ------------ method called once each job just after ending the event loop  ------------
void
TtSemiEvtKit::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(TtSemiEvtKit);
