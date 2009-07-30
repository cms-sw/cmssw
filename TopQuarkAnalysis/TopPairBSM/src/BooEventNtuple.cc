/**_________________________________________________________________
   class:   BooEventNtuple.cc
   package: 


 author: Francisco Yumiceva, Fermilab (yumiceva@fnal.gov)

 version $Id: BooEventNtuple.cc,v 1.1.2.2 2009/04/27 19:16:50 yumiceva Exp $

________________________________________________________________**/

#ifdef NOSCRAMV
#include "BooEventNtuple.h"
#else
#include "TopQuarkAnalysis/TopPairBSM/interface/BooEventNtuple.h"
#endif


ClassImp(BooEventNtuple)

// ROOT

//_______________________________________________________________
BooEventNtuple::BooEventNtuple()
{

    this->Reset();

}



//_______________________________________________________________
BooEventNtuple::~BooEventNtuple()
{
}

//_______________________________________________________________
void BooEventNtuple::Reset()
{

	event = -1;       
    run = -1;    
	dataType = -1;

	njets = -1;       // number of jets
    nmuons = -1 ;      // number of muons
    nvertices = -1;
	ngenjets = -1 ;    // number of generated jets

	jet_pt.clear();
    jet_eta.clear();
    jet_phi.clear();
    jet_e.clear();
    jet_et.clear();
    jet_ntrks.clear();
    jet_flavour.clear();
    jetcorrection.clear();
    genjet_pt.clear();
    genjet_eta.clear();
    genjet_phi.clear();
    genjet_e.clear();
	muon_px.clear();
	muon_py.clear();
	muon_pz.clear();
	muon_e.clear();
	muon_normchi2.clear();
	muon_d0.clear();
	muon_d0Error.clear();
	muon_old_reliso.clear();
	muon_new_reliso.clear();
	muon_ptrel.clear();
	muon_minDeltaR.clear();
	muon_closestJet_px.clear();
	muon_closestJet_py.clear();
	muon_closestJet_pz.clear();
	muon_closestJet_e.clear();
	
	MET.clear();
	Ht.clear();
	genmuon_px.clear();
	genmuon_py.clear();
	genmuon_pz.clear();
	genmuon_e.clear();
	genmuon_pdg.clear();
	genmoun_motherpdg.clear();
	
	gentop_px.clear();
	gentop_py.clear();
	gentop_pz.clear();
	gentop_e.clear();
	gentop_charge.clear();
	gentop_hadronic.clear();
	gennu_px.clear();
	gennu_py.clear();
	gennu_pz.clear();
	gennu_e.clear();
	gennu_pdg.clear();
}

