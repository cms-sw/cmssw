#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "AnalysisDataFormats/TopObjects/interface/StGenEvent.h"
#include "TopQuarkAnalysis/TopEventProducers/interface/StDecaySubset.h"

using namespace std;
using namespace reco;

StDecaySubset::StDecaySubset(const edm::ParameterSet& cfg):
  TopDecaySubset(cfg)
{
}

StDecaySubset::~StDecaySubset()
{
}

void
StDecaySubset::produce(edm::Event& evt, const edm::EventSetup& setup)
{     
  edm::Handle<reco::GenParticleCollection> src;
  evt.getByLabel(src_, src);
 
  const reco::GenParticleRefProd ref = evt.getRefBeforePut<reco::GenParticleCollection>(); 
  std::auto_ptr<reco::GenParticleCollection> sel( new reco::GenParticleCollection );

  // clear existing refs
  refs_.clear();  
  // fill output collection depending on
  // what generator listing is expected
  switch(genType_){
  case 0:
    fillPythiaOutput   ( *src, *sel );
    break;
  case 1:
    fillMadgraphOutput ( *src, *sel );
    break;
  case 2:
    fillSingleTopOutput( *src, *sel );
    break;
  }
  // fill references
  fillRefs( ref, *sel );
  // print decay chain for debugging
  // printSource( *src, pdg_);
  print( *sel, pdg_);
  // fan out to event
  evt.put( sel );
}

void StDecaySubset::fillSingleTopOutput(const reco::GenParticleCollection& src, reco::GenParticleCollection& sel)
{
  // this is needed, for example, for the SingleTop generator, since it doesn't save the 
  // intermediate particles (lepton, neutrino and b are directly daughters of the incoming 
  // partons)
  int iP;
  int idx=-1;
  vector<int> ipDaughs;
  for(GenParticleCollection::const_iterator ip1=src.begin(); ip1!=src.end(); ++ip1){
    for(GenParticleCollection::const_iterator ip2=src.begin(); ip2!=src.end(); ++ip2){
      //iterate over the daughters of both
      for(GenParticle::const_iterator td1=ip1->begin(); td1!=ip1->end(); ++td1){
	for(GenParticle::const_iterator td2=ip2->begin(); td2!=ip2->end(); ++td2){
	  if (td1 == td2) { // daughter of both initial state partons
	    // ++idx;
	    // iP=idx;
	    GenParticle* cand = new GenParticle( td2->charge(), getP4( td2->begin(), td2->end(), td2->pdgId() ), 
						 td2->vertex(), td2->pdgId(), td2->status(), false );
	    auto_ptr<GenParticle> ptr( cand );
	    sel.push_back( *ptr );	  
	    ipDaughs.push_back( ++idx ); //push index of daughter
	    iP=idx;
	  }
	  refs_[ iP ]=ipDaughs;
	}
      }// end of double loop on daughters
    }
  }
}
