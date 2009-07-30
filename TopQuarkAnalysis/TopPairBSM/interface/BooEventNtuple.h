#ifndef BooEventNtuple_h
#define BooEventNtuple_h

/**_________________________________________________________________
   class:   BooEventNtuple.h
   package: TopPairBSM


 author: Francisco Yumiceva, Fermilab (yumiceva@fnal.gov)

 version $Id: BooEventNtuple.h,v 1.1.2.4 2009/04/27 19:16:48 yumiceva Exp $

________________________________________________________________**/

#include <vector>

// ROOT
#include "TObject.h"
#include "TMatrixDSym.h"

class BooEventNtuple : public TObject
{

public:

    BooEventNtuple();
    ~BooEventNtuple();

    void                 Reset();

	//_______ event ID_______________________________
    Int_t event;       // event number
    Int_t run;         // run number
	Int_t dataType;    // type of data: MC, cosmics, colisions,
	
    //_______ event ID_______________________________
    Int_t njets;       // number of jets
    Int_t nmuons;      // number of muons
    Int_t nvertices;   // number of vertices
  
   	//_____ total number of MC objects ______________
    Int_t ngenjets;    // number of generated jets
  
	//_____ jets ____________________________________
    std::vector< float > jet_pt;
    std::vector< float > jet_eta;
    std::vector< float > jet_phi;
    std::vector< float > jet_e;
    std::vector< float > jet_et;
    std::vector< int > jet_ntrks;
	
    std::vector< int > jet_flavour;
    std::vector< float > jetcorrection;
    
    std::vector< float > genjet_pt;
    std::vector< float > genjet_eta;
    std::vector< float > genjet_phi;
    std::vector< float > genjet_e;

	//_____ muons ____________________________________
    std::vector< float > muon_px;
    std::vector< float > muon_py;
    std::vector< float > muon_pz;
    std::vector< float > muon_e;
    std::vector< float > muon_normchi2;
    std::vector< float > muon_d0;
    std::vector< float > muon_d0Error;
    std::vector< float > muon_old_reliso;
    std::vector< float > muon_new_reliso;
    std::vector< float > muon_ptrel;
	std::vector< float > muon_minDeltaR;
	std::vector< float > muon_closestJet_px;
	std::vector< float > muon_closestJet_py;
	std::vector< float > muon_closestJet_pz;
	std::vector< float > muon_closestJet_e;
	
	
    std::vector< float > MET;
    std::vector< float > Ht;

    //
    std::vector< float > chi2;
    //std::vector< TLorentzVector > hadronictop;
    //std::vector< TLorentzVector > leptonictop;

    // _____ gen ___
    std::vector< float > genmuon_px;
	std::vector< float > genmuon_py;
	std::vector< float > genmuon_pz;
	std::vector< float > genmuon_e;
	std::vector< int > genmuon_pdg;
	std::vector< int > genmoun_motherpdg;

	std::vector< float > gentop_px;
	std::vector< float > gentop_py;
	std::vector< float > gentop_pz;
	std::vector< float > gentop_e;
	std::vector< float > gentop_charge;
	std::vector< int > gentop_hadronic;
	std::vector< float > gennu_px;
	std::vector< float > gennu_py;
	std::vector< float > gennu_pz;
	std::vector< float > gennu_e;
	std::vector< int > gennu_pdg;

    ClassDef(BooEventNtuple,1);

};

#endif
