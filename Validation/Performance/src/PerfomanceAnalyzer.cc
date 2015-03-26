
#include "Validation/Performance/interface/PerformanceAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/HLTReco/interface/ModuleTiming.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include <iostream>
#include "DQMServices/Core/interface/DQMStore.h"

PerformanceAnalyzer::PerformanceAnalyzer(const edm::ParameterSet& ps)
    : fOutputFile( ps.getUntrackedParameter<std::string>("outputFile", "") )
{
  eventTime_Token_ = consumes<edm::EventTime> (edm::InputTag("cputest"));
}

PerformanceAnalyzer::~PerformanceAnalyzer()
{
  // don't try to delete any pointers - they're handled by DQM machinery
}

void PerformanceAnalyzer::bookHistograms(DQMStore::IBooker & iBooker, edm::Run const & iRun, edm::EventSetup const & /* iSetup */)
{
  iBooker.setCurrentFolder("PerformanceV/CPUPerformanceTask");

  fVtxSmeared          = iBooker.book1D("VtxSmeared",          "VtxSmeared",          100, 0., 0.1 );
  fg4SimHits           = iBooker.book1D("g4SimHits",           "g4SimHits",           100, 0., 500.);
  fMixing              = iBooker.book1D("Mixing",              "Mixing",              100, 0., 0.5 );
  fSiPixelDigis        = iBooker.book1D("SiPixelDigis",        "SiPixelDigis",        100, 0., 0.5 );
  fSiStripDigis        = iBooker.book1D("SiStripDigis",        "SiStripDigis",        100, 0., 2.  );
  fEcalUnsuppDigis     = iBooker.book1D("EcalUnsuppDigis",     "EcalUnsuppDigis",     100, 0., 2.  );
  fEcalZeroSuppDigis   = iBooker.book1D("EcalZeroSuppDigis",   "EcalZeroSuppDigis",   100, 0., 2.  );
  fPreShwZeroSuppDigis = iBooker.book1D("PreShwZeroSuppDigis", "PreShwZeroSuppDigis", 100, 0., 0.1 );
  fHcalUnsuppDigis     = iBooker.book1D("HcalUnsuppDigis",     "HcalUnsuppDigis",     100, 0., 0.5 );
  fMuonCSCDigis        = iBooker.book1D("MuonCSCDigis",        "MuonCSCDigis",        100, 0., 0.1 );
  fMuonDTDigis         = iBooker.book1D("MuonDTDigis",         "MuonDTDigis",         100, 0., 0.1 );
  fMuonRPCDigis        = iBooker.book1D("MuonRPCDigis",        "MuonRPCDigis",        100, 0., 0.1 );
}

void PerformanceAnalyzer::analyze(const edm::Event& e, const edm::EventSetup& eventSetup)
{
  if ( e.id().event() == 1) return ; // skip 1st event

  edm::Handle<edm::EventTime> EvtTime;
  e.getByToken(eventTime_Token_, EvtTime ) ;

  for ( unsigned int i=0; i<EvtTime->size(); ++i )
  {
    if ( EvtTime->name(i) == "VtxSmeared" )   fVtxSmeared->Fill( EvtTime->time(i) ) ;
    if ( EvtTime->name(i) == "g4SimHits"  )   fg4SimHits->Fill( EvtTime->time(i) ) ;
    if ( EvtTime->name(i) == "mix" )          fMixing->Fill( EvtTime->time(i) ) ;
    if ( EvtTime->name(i) == "siPixelDigis" ) fSiPixelDigis->Fill( EvtTime->time(i) ) ;
    if ( EvtTime->name(i) == "siStripDigis" ) fSiStripDigis->Fill( EvtTime->time(i) ) ;
    if ( EvtTime->name(i) == "ecalUnsuppressedDigis" ) fEcalUnsuppDigis->Fill( EvtTime->time(i) ) ;
    if ( EvtTime->name(i) == "ecalDigis")     fEcalZeroSuppDigis->Fill( EvtTime->time(i) ) ;
    if ( EvtTime->name(i) == "ecalPreshowerDigis") fPreShwZeroSuppDigis->Fill( EvtTime->time(i) ) ;
    if ( EvtTime->name(i) == "hcalDigis" )    fHcalUnsuppDigis->Fill( EvtTime->time(i) ) ;
    if ( EvtTime->name(i) == "muonCSCDigis" ) fMuonCSCDigis->Fill( EvtTime->time(i) ) ;
    if ( EvtTime->name(i) == "muonDTDigis" )  fMuonDTDigis->Fill( EvtTime->time(i) ) ;
    if ( EvtTime->name(i) == "muonRPCDigis" ) fMuonRPCDigis->Fill( EvtTime->time(i) ) ;
  }
  return ;
}

DEFINE_FWK_MODULE(PerformanceAnalyzer);
