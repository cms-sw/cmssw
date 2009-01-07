/**_________________________________________________________________
   class:   BooHistograms.cc
   package: Analyzer/TopTools


 author: Francisco Yumiceva, Fermilab (yumiceva@fnal.gov)

 version $Id: BooHistograms.cc,v 1.1.2.1 2009/01/07 22:32:02 yumiceva Exp $

________________________________________________________________**/


#include "TopQuarkAnalysis/TopPairBSM/interface/BooHistograms.h"
#include "TF1.h"

#include<iostream>

//_______________________________________________________________
BooHistograms::BooHistograms() {

}

//_______________________________________________________________
BooHistograms::~BooHistograms() {

	this->DeleteHisto();
	
}

//_______________________________________________________________
void BooHistograms::Init(TString type, TString suffix1, TString suffix2) {

	if (suffix1 != "") suffix1 = "_" + suffix1;
	if (suffix2 != "") suffix1 += "_" + suffix2;

	if ( type == "counter" )  h1["counter"] = new TH1D("counter","Selection",20,0,20);
	
	if ( type == "generator") {
		h1["gen_top_pt"] = new TH1D("gen_top_pt","top quark p_{T} [GeV/c]",80,0,1200);
		h1["gen_top_eta"] = new TH1D("gen_top_eta","top quark #eta",60,-4,4);
		h1["gen_top_rapidity"] = new TH1D("gen_top_rapidity","top quark rapidity",60,-4,4);
		h1["gen_top_eta1"] = new TH1D("gen_top_eta1","top quark #eta",60,-4,4);
		h1["gen_top_eta2"] = new TH1D("gen_top_eta2","top quark #eta",60,-4,4);
		h1["gen_top_eta3"] = new TH1D("gen_top_eta3","top quark #eta",60,-4,4);
		h1["gen_top_decays"] = new TH1D("gen_top_decays","top decays",6,0,6);
		
		h2["gentop_rapidities"] = new TH2D("gentop_rapidities","y_{t} vs y_{T}",50,-3,3,50,-3,3);
		h1["gen_toppair_mass"] = new TH1D("gen_toppair_mass","top pair Mass [Gev/c^{2}]",100,100,4500);
		h1["gen_toppair_pt"] = new TH1D("gen_toppair_pt","top pair p_{T} [Gev/c]",80,0,500.);
		
		h1["gen_deltaR_qb"] = new TH1D("gen_deltaR_qb","#Delta R(q,Had-b)",35,0.,7.);
		h1["gen_deltaR_pb"] = new TH1D("gen_deltaR_pb","#Delta R(p,Had-b)",35,0.,7.);
		h1["gen_deltaR_pq"] = new TH1D("gen_deltaR_pq","#Delta R(p,q)",35,0.,7.);
		h1["gen_deltaR_lb"] = new TH1D("gen_deltaR_lb","#Delta R(#mu,Lep-b)",35,0.,7.);
		h1["gen_deltaR_qLepb"] = new TH1D("gen_deltaR_qLepb","#Delta R(q,Lep-b)",35,0.,7.);
		h1["gen_deltaR_qmu"] = new TH1D("gen_deltaR_qmu","#Delta R(q,#mu)",35,0.,7.);
		h1["gen_deltaR_muLepb"] = new TH1D("gen_deltaR_muLepb","#Delta R(#mu,Lep-b)",35,0.,7.);
		h2["gen_deltaR_pq_vs_toppt"] = new TH2D("gen_deltaR_pq_vs_toppt","#Delta R(p,q) vs top quark p_{T}",35,0.,7,80,0,1200);
		h2["gen_deltaR_qb_vs_toppt"] = new TH2D("gen_deltaR_qb_vs_toppt","#Delta R(q,Had-b) vs top quark p_{T}",35,0.,7,80,0,1200);
		h2["gen_deltaR_muLepb_vs_toppt"] = new TH2D("gen_deltaR_muLepb_vs_toppt","#Delta R(#mu,Lep-b) vs top quark p_{T}",35,0.,7,80,0,1200);
		h2["gen_muonpt_vs_lepbpt"] = new TH2D("gen_muonpt_vs_lepbpt","muon p_{T} vs leptonic b-jet p_{T}",80,0,100,80,0,500);
		
		h1["gen_nu_pz"] = new TH1D("gen_nu_pz","Neutrino p_{z} [GeV/c]",50,-500.0,500.0);
		h2["gen_nu_pt_vs_pz"+suffix1] = new TH2D("gen_nu_pt_vs_pz"+suffix1,"Neutrino p_{T} vs p_{z} [GeV/c]",50,0,500,50,-500,500);
		h2["gen_toprapidity_vs_psi_pq"] = new TH2D("gen_toprapidity_vs_psi_pq","y_{top} vs #psi(p,q)",50, -2.5,2.5,50,0.,5);
		h2["gen_toprapidity_vs_deltaR_pq"] = new TH2D("gen_toprapidity_vs_deltaR_pq","y_{top} vs #Delta R(p,q)",50, -2.5,2.5,50,0,7.);
		h2["gen_toprapidity_vs_dminij_pq"] = new TH2D("gen_toprapidity_vs_dminij_pq","y_{top} vs d_{min}(p,q)",50, -2.5,2.5,50,0.,0.1);
		h2["gen_toprapidity_vs_dmaxij_pq"] = new TH2D("gen_toprapidity_vs_dmaxij_pq","y_{top} vs d_{max}(p,q)",50, -2.5,2.5,50,0.,0.5);

		h2["gen_Hadb_pT_vs_pL"] = new TH2D("gen_Hadb_pT_vs_pL","p_{T} vs p_{L}",70,-100,400,80,-500,2000);
		h2["gen_HadW_pT_vs_pL"] = new TH2D("gen_HadW_pT_vs_pL","p_{T} vs p_{L}",70,-100,400,80,-500,2000);
		
		h2["gen_cosCM_vs_psi"] = new TH2D("gen_cosCM_vs_psi","cos #theta* vs #psi",80,0,1,50,0,5);
	}
	else if ( type == "Jets") {
		h1["jets"+suffix1]     = new TH1D("jets"+suffix1,"Number of jets",15,0,15);
		h1["jet0_et"+suffix1]  = new TH1D("jet0_et"+suffix1,"Jet #1 E_{T} [GeV]",50,0,500);
		h1["jet1_et"+suffix1]  = new TH1D("jet1_et"+suffix1,"Jet #2 E_{T} [GeV]",50,0,500);
		h1["jet2_et"+suffix1]  = new TH1D("jet2_et"+suffix1,"Jet #3 E_{T} [GeV]",50,0,500);
		h1["jet3_et"+suffix1]  = new TH1D("jet3_et"+suffix1,"Jet #4 E_{T} [GeV]",50,0,500);
		h1["jet0_eta"+suffix1]  = new TH1D("jet0_eta"+suffix1,"Jet #1 #eta [GeV]",50,-3.,3.);
		h1["jet_et"+suffix1]  = new TH1D("jet_et"+suffix1,"Jet E_{T} [GeV]",50,0,500);
		h1["jet_eta"+suffix1] = new TH1D("jet_eta"+suffix1,"Jet #eta",50,-3.,3.);
		h2["jet_ptVseta"+suffix1] = new TH2D("jet_ptVseta"+suffix1,"Jet E_{T} vs #eta",50,0,500,50,-3.,3.);
		h1["jet_phi"+suffix1] = new TH1D("jet_phi"+suffix1,"Jet #phi",30,-3.15,3.15);
		h1["jet_emFrac"+suffix1] = new TH1D("jet_emFrac"+suffix1,"Jet EM energy fraction",30,0,1);
		
		h2["jet_res_et"+suffix1]  = new TH2D("jet_res_et"+suffix1,"Jet E_{T} [GeV]",80,0,300,60,0,30.);

		h1["jet_nocorr_et"+suffix1]  = new TH1D("jet_nocorr_et"+suffix1,"Jet E_{T} [GeV]",50,0,500);
		
		h1["Mjet_mass"+suffix1] = new TH1D("Mjet_mass"+suffix1, "MJet mass [GeV/c^{2}]",80,0,500);
		h1["Mjet_et"+suffix1] = new TH1D("Mjet_et"+suffix1, "MJet E_{T} [GeV]",100,0,2500);
		h1["Mjet_et_subjets"+suffix1] = new TH1D("Mjet_et_subjets"+suffix1, "Jets in MJet E_{T} [GeV]",100,0,2500);
		h1["Mjet_deltaR_mu"+suffix1] = new TH1D("Mjet_deltaR_mu"+suffix1, "#Delta R(MJet,#mu)",35,0,5);
		h1["Mjet_njets"+suffix1]     = new TH1D("Mjet_njets"+suffix1,"Number of jets in Mjet",10,0,10);
		h1["Mjet_deltaR_b"+suffix1] = new TH1D("Mjet_deltaR_b"+suffix1, "#Delta R(MJet,b)",35,0,5);
		h1["bjet_mass"+suffix1] = new TH1D("bjet_mass"+suffix1, "b (leptonic) Jet mass [GeV/c^{2}]",80,0,500);
		h1["bjet_et"+suffix1] = new TH1D("bjet_et"+suffix1, "b (leptonic) Jet E_{T} [GeV/c]",50,0,500);
		h1["topPair_mass"+suffix1] = new TH1D("topPair_mass"+suffix1, "top pair mass [GeV/c^{2}]",100,100,4500);
		//h1["topPair_et"+suffix1] = new TH1D("Mjet_et"+suffix1, "MJet E_{T} [GeV/c^{2}]",100,0,2500);
		h1["jet_deltaR_muon"+suffix1] = new TH1D("jet_deltaR_muon"+suffix1, "#Delta R(Jet,#mu)",35,0,9);
		h1["jet_pTrel_muon"+suffix1] = new TH1D("jet_pTrel_muon"+suffix1, "p_{T}^{Rel}",35,0,10);
		h1["jet_pTrel_muon_b"+suffix1] = new TH1D("jet_pTrel_muon_b"+suffix1, "p_{T}^{Rel}",35,0,10);
		h1["jet_pTrel_muon_c"+suffix1] = new TH1D("jet_pTrel_muon_c"+suffix1, "p_{T}^{Rel}",35,0,10);
		h1["jet_pTrel_muon_udsg"+suffix1] = new TH1D("jet_pTrel_muon_udsg"+suffix1, "p_{T}^{Rel}",35,0,10);
		h1["jet_flavor_closest_muon"+suffix1] = new TH1D("jet_flavor_closest_muon"+suffix1, "Jet Flavor",21,0,21);
		
		h1["jet_pT_closest_muon"+suffix1]  = new TH1D("jet_pT_closest_muon"+suffix1,"Jet p_{T} [GeV]",50,0,500);
		h2["jet_deltaR_muon_vs_RelIso"+suffix1] = new TH2D("jet_deltaR_muon_vs_RelIso"+suffix1, "#Delta R(Jet,#mu) vs RelIso",35,0,9,80,0,5);
	
		h1["jet_deltaR_LeptonicW"+suffix1] = new TH1D("jet_deltaR_LeptonicW"+suffix1, "#Delta R(Jet,W_{#mu+#nu})",35,0,5);
		h1["LeptonicTop_psi"+suffix1] = new TH1D("LeptonicTop_psi"+suffix1, "#psi(jet+W_{#mu+#nu})",100,0.,12.);
		h1["HadronicTop_psi"+suffix1] = new TH1D("HadronicTop_psi"+suffix1, "#psi(jet+leading-jet)",100,0,12.);
		h1["LeptonicTop_pt"+suffix1] = new TH1D("LeptonicTop_pt"+suffix1, "p_{T} (jet+W_{#mu+#nu}) [GeV/c]",100,0,4500.);
		h1["HadronicTop_pt"+suffix1] = new TH1D("HadronicTop_pt"+suffix1, "p_{T} (jet+leading-jet) [GeV/c]",100,0,4500.);
		h1["jet_combinations_ProbChi2"+suffix1] = new TH1D("jet_combinations_ProbChi2"+suffix1, "#chi^{2} Probability",50,0,1.);
		h1["jet_combinations_NormChi2"+suffix1] = new TH1D("jet_combinations_NormChi2"+suffix1, "#chi^{2}/ndf",100,0,500);
		h1["MCjet_combinations_ProbChi2"+suffix1] = new TH1D("MCjet_combinations_ProbChi2"+suffix1, "#chi^{2} Probability",50,0,1.);
                h1["MCjet_combinations_NormChi2"+suffix1] = new TH1D("MCjet_combinations_NormChi2"+suffix1, "#chi^{2}/ndf",100,0,500);

	}
	else if ( type == "DisplayJets") {
		//h2["jet0_calotowerI"+suffix1]  = new TH2D("jet0_calotowerI"+suffix1,"Jet #1 CaloTowers [GeV/c]",40,-1.740,1.740,72,0.,3.141);
		h2["jet0_calotowerI"+suffix1]  = new TH2D("jet0_calotowerI"+suffix1,"Jet #1 CaloTowers [GeV/c]",58,-2.523,2.523,144,-3.14,3.141);
		h2["jet1_calotowerI"+suffix1]  = new TH2D("jet1_calotowerI"+suffix1,"Jet #2 CaloTowers [GeV/c]",58,-2.523,2.523,144,-3.14,3.141);
		h2["jet2_calotowerI"+suffix1]  = new TH2D("jet2_calotowerI"+suffix1,"Jet #3 CaloTowers [GeV/c]",58,-2.523,2.523,144,-3.14,3.141);
		h2["jet3_calotowerI"+suffix1]  = new TH2D("jet3_calotowerI"+suffix1,"Jet #4 CaloTowers [GeV/c]",58,-2.523,2.523,144,-3.14,3.141);

		//Double_t caloEta[8] = {1.740,1.830,1.930,2.043,2.172,2.322,2.500,2.650}; //8 bins
		//h2["jet0_calotowerII"+suffix1]  = new TH2D("jet0_calotowerII"+suffix1,"Jet #1 CaloTowers [GeV/c]",7,caloEta,36,0.,3.141);
		//h2["jet1_calotowerII"+suffix1]  = new TH2D("jet1_calotowerII"+suffix1,"Jet #2 CaloTowers [GeV/c]",7,caloEta,36,0.,3.141);
		//h2["jet2_calotowerII"+suffix1]  = new TH2D("jet2_calotowerII"+suffix1,"Jet #3 CaloTowers [GeV/c]",7,caloEta,36,0.,3.141);
		//h2["jet3_calotowerII"+suffix1]  = new TH2D("jet3_calotowerII"+suffix1,"Jet #4 CaloTowers [GeV/c]",7,caloEta,36,0.,3.141);	
		
	}
	else if ( type == "Muons") {
		h2["muons_vsJets"+suffix1]         = new TH2D("muons_vsJets"+suffix1,"Number of muons vs Jets",4,1,5,4,1,5);
		h1["muons"+suffix1]                = new TH1D("muons"+suffix1,"Number of muons",4,1,5);
		h1["muon_normchi2"+suffix1]        = new TH1D("muon_normchi2"+suffix1,"#chi^{2}/ndof",40,0,30);
		h1["muon_pt"+suffix1]              = new TH1D("muon_pt"+suffix1,"Muon p_{T} [GeV/c]",80,0.0,200.0);
		h1["muon_eta"+suffix1]              = new TH1D("muon_eta"+suffix1,"Muon #eta",50,-3.,3.);
		h1["muon_phi"+suffix1]              = new TH1D("muon_phi"+suffix1,"Muon #phi",30,-3.15,3.15);
		h1["muon_caloIso"+suffix1]       = new TH1D("muon_caloIso"+suffix1,"caloIsolation",80,0.0,300.0);
		h1["muon_trackIso"+suffix1]       = new TH1D("muon_trackIso"+suffix1,"trackIsolation",80,0.0,100.0);
		h1["muon_leptonID"+suffix1]       = new TH1D("muon_leptonID"+suffix1,"leptonID",80,0.0,1.0);
		h1["muon_deltaR_nu"+suffix1]  = new TH1D("muon_deltaR_nu"+suffix1, "#Delta R(#mu,#nu)",35,0,5);
		h1["muon_deltaPhi_nu"+suffix1]  = new TH1D("muon_deltaPhi_nu"+suffix1, "#Delta #phi(#mu,#nu)",35,0,3.142);
		h1["muon_deltaR_b"+suffix1]  = new TH1D("muon_deltaR_b"+suffix1, "#Delta R(#mu,b)",35,0,5);
		h1["muon_RelIso"+suffix1]       = new TH1D("muon_RelIso"+suffix1,"RelIsolation",80,0.0,5.0);
	
	}
	else if ( type == "MET") {

		h2["MET_vsJets"+suffix1] = new TH2D("MET_vsJets"+suffix1,"MET [GeV] vs Jets",100,0.0,1500.0,4,0,4);
		h1["MET"+suffix1] = new TH1D("MET"+suffix1,"MET [GeV]",100,0.0,1500.0);
		h1["MET_eta"+suffix1] = new TH1D("MET_eta"+suffix1,"#eta_{MET}",50,-3.,3.);
		h1["MET_phi"+suffix1] = new TH1D("MET_phi"+suffix1,"#phi_{MET}",30,-3.15,3.15);

		h1["Ht"+suffix1] = new TH1D("Ht"+suffix1,"Ht [GeV]",100,0,2500);
		h2["Ht_vsJets"+suffix1] = new TH2D("Ht_vsJets"+suffix1,"Ht [GeV] vs Jets",100,0,2500,4,0,4);
		
		h1["myMET"+suffix1] = new TH1D("myMET"+suffix1,"MET [GeV]",100,0.0,1500.0);
		h1["MET_deltaR_muon"+suffix1] = new TH1D("MET_deltaR_muon"+suffix1,"#DeltaR(MET,#mu)",35,0.,7.);
		//h1["METcomplex"+suffix1] = new TH1D("METcomplex"+suffix1,"MET [GeV]",80,0.0,300.0);
		h1["nu_pz"+suffix1] = new TH1D("nu_pz"+suffix1,"Neutrino p_{z} [GeV/c]",50,-500.0,500.0);
		h2["nu_pt_vs_pz"+suffix1] = new TH2D("nu_pt_vs_pz"+suffix1,"Neutrino p_{T} vs p_{z} [GeV/c]",50,0,500,50,-500,500);
		h1["nu_eta"+suffix1] = new TH1D("nu_eta"+suffix1,"Neutrino #eta",50,-3.,3.);
		h1["delta_nu_pz"+suffix1] = new TH1D("delta_nu_pz"+suffix1,"Neutrino #Delta(p_{z}-p^{gen}_{z}) [GeV/c]",50,-1000.0,1000.0);
		h1["LeptonicW_psi"+suffix1] = new TH1D("LeptonicW_psi"+suffix1, "#psi(#mu + #nu)",100,0.,12.);
		h1["LeptonicW_dij"+suffix1] = new TH1D("LeptonicW_dij"+suffix1, "d_{ij}(#mu,#nu)",100,0.,0.1);

	}
	else if ( type == "Mass") {

		h1["LeptonicTop_mass"+suffix1] = new TH1D("LeptonicTop_mass"+suffix1, "Mass (jet+W_{#mu+#nu}) [GeV/c^{2}]",20,100.,500.);
		h1["HadronicTop_mass"+suffix1] = new TH1D("HadronicTop_mass"+suffix1, "Mass (j_{1}j_{2}j_{3}) [GeV/c^{2}]",20,100.,500.);
		h1["LeptonicW_mass"+suffix1] = new TH1D("LeptonicW_mass"+suffix1, "Mass(#mu + #nu) [GeV/c^{2}]",20,0,300);
		h1["HadronicW_mass"+suffix1] = new TH1D("HadronicW_mass"+suffix1, "Mass(j_{1}j_{2}) [GeV/c^{2}]",20,0,300);
		h2["LepTop_vs_LepW"+suffix1] = new TH2D("LepTop_vs_LepW"+suffix1, "Mass LepTop vs LepW [GeV/c^{2}]",20,100,500,20,0.,300.);
		h2["HadTop_vs_HadW"+suffix1] = new TH2D("HadTop_vs_HadW"+suffix1, "Mass HadTop vs HadW [GeV/c^{2}]",20,100,500,20,0.,300.);		
		h1["MCLeptonicTop_mass"+suffix1] = new TH1D("MCLeptonicTop_mass"+suffix1, "Mass (jet+W_{#mu+#nu}) [GeV/c^{2}]",20,100.,500.);
                h1["MCHadronicTop_mass"+suffix1] = new TH1D("MCHadronicTop_mass"+suffix1, "Mass (j_{1}j_{2}j_{3}) [GeV/c^{2}]",20,100.,500.);
                h1["MCLeptonicW_mass"+suffix1] = new TH1D("MCLeptonicW_mass"+suffix1, "Mass(#mu + #nu) [GeV/c^{2}]",20,0,300);
                h1["MCHadronicW_mass"+suffix1] = new TH1D("MCHadronicW_mass"+suffix1, "Mass(j_{1}j_{2}) [GeV/c^{2}]",20,0,300);
                h2["MCLepTop_vs_LepW"+suffix1] = new TH2D("MCLepTop_vs_LepW"+suffix1, "Mass LepTop vs LepW [GeV/c^{2}]",20,100,500,20,0.,300.);
                h2["MCHadTop_vs_HadW"+suffix1] = new TH2D("MCHadTop_vs_HadW"+suffix1, "Mass HadTop vs HadW [GeV/c^{2}]",20,100,500,20,0.,300.);


		h1["WTolnu"+suffix1] = new TH1D("WTolnu"+suffix1,"(#mu + #nu) mass [GeV/c^{2}]",80,0.0,300.0);
		h1["tToWlnuj"+suffix1] = new TH1D("tToWlnuj"+suffix1,"(W_{l#nu} + jet) mass [GeV/c^{2}]",50,0.0,500.0);
		h1["tToWlnub"+suffix1] = new TH1D("tToWlnub"+suffix1,"(W_{l#nu} + jet) mass [GeV/c^{2}]",50,0.0,500.0);
		
		h1["WTojj"+suffix1] = new TH1D("WTojj"+suffix1,"(jet+jet) mass [GeV/c^{2}]",80,0.0,300.0);
		h1["tTojjj"+suffix1] = new TH1D("tTojjj"+suffix1,"(three-jet) mass [GeV/c^{2}]",50,0.0,500.0);
		//h1["tToWj"+suffix1] = new TH1D("tToWj"+suffix1,"(W_{jj} + jet) mass [GeV/c^{2}]",50,0.0,500.0);
		h1["topPair"+suffix1] = new TH1D("topPair"+suffix1,"top pair Mass [Gev/c^{2}]",100,100,4500);
		h1["fittopPair"+suffix1] = new TH1D("fittopPair"+suffix1,"top pair Mass [Gev/c^{2}]",100,100,4500);
		h1["topPairRes"+suffix1] = new TH1D("topPairRes"+suffix1,"top pair Mass Resolution",50,-1,1);
		h1["fittopPairRes"+suffix1] = new TH1D("fittopPairRes"+suffix1,"top pair Mass Resolution",50,-1,1);
		//h1["kinfit_chi2"+suffix1] = new TH1D("kinfit_chi2"+suffix1,"#chi^{2} Distribution",100,0.,200.);
		h1["kinfit_probchi2"+suffix1] = new TH1D("kinfit_probchi2"+suffix1,"#chi^{2} Probability",80,0.,1.);

		
		h1["EtpPull"+suffix1] = new TH1D("EtpPull"+suffix1,"(E^{fit}_{T} - E_{T})/#sigma_{E_{T}}",50,-10,10);
		h1["EtqPull"+suffix1] = new TH1D("EtqPull"+suffix1,"(E^{fit}_{T} - E_{T})/#sigma_{E_{T}}",50,-10,10);
		h1["EthbPull"+suffix1] = new TH1D("EthbPull"+suffix1,"(E^{fit}_{T} - E_{T})/#sigma_{E_{T}}",50,-10,10);
		h1["EtlbPull"+suffix1] = new TH1D("EtlbPull"+suffix1,"(E^{fit}_{T} - E_{T})/#sigma_{E_{T}}",50,-10,10);
		h1["EtlPull"+suffix1] = new TH1D("EtlPull"+suffix1,"(E^{fit}_{T} - E_{T})/#sigma_{E_{T}}",50,-10,10);
		h1["fitEtpPull"+suffix1] = new TH1D("fitEtpPull"+suffix1,"|E^{fit}_{T} - E^{gen}_{T}|/#sigma_{E_{T}}",50,-10,10);
		h1["fitEtqPull"+suffix1] = new TH1D("fitEtqPull"+suffix1,"|E^{fit}_{T} - E^{gen}_{T}|/#sigma_{E_{T}}",50,-10,10);
		h1["fitEthbPull"+suffix1] = new TH1D("fitEthbPull"+suffix1,"|E^{fit}_{T} - E^{gen}_{T}|/#sigma_{E_{T}}",50,-10,10);
		h1["fitEtlbPull"+suffix1] = new TH1D("fitEtlbPull"+suffix1,"|E^{fit}_{T} - E^{gen}_{T}|/#sigma_{E_{T}}",50,-10,10);
		h1["fitEtlPull"+suffix1] = new TH1D("fitEtlPull"+suffix1,"|E^{fit}_{T} - E^{gen}_{T}|/#sigma_{E_{T}}",50,-10,10);
		
	}
	else {

		h1["phi_Wb"+suffix1] = new TH1D("phi_Wb"+suffix1,"Phi Difference of W(u+v) and J",50,0.0,3.2);    // phi balance t->W+b
		
		h1["EtnRes"+suffix1] = new TH1D("EtnRes"+suffix1,"|E_{T} - E_{Tfit}|/#sigma_{Efit_{T}}",50,-10,10);
		h1["EtapRes"+suffix1] = new TH1D("EtapRes"+suffix1,"|#eta - #eta_{Tfit}|/#sigma_{#eta fit}",50,-10,10);
		h1["EtaqRes"+suffix1] = new TH1D("EtaqRes"+suffix1,"|#eta - #eta_{Tfit}|/#sigma_{#eta fit}",50,-10,10);
		h1["EtahbRes"+suffix1] = new TH1D("EtahbRes"+suffix1,"|#eta - #eta_{Tfit}|/#sigma_{#eta fit}",50,-10,10);
		h1["EtalbRes"+suffix1] = new TH1D("EtalbRes"+suffix1,"|#eta - #eta_{Tfit}|/#sigma_{#eta fit}",50,-10,10);
		h1["EtalRes"+suffix1] = new TH1D("EtalRes"+suffix1,"|#eta - #eta_{Tfit}|/#sigma_{#eta fit}",50,-10,10);
		h1["EtanRes"+suffix1] = new TH1D("EtanRes"+suffix1,"|#eta - #eta_{Tfit}|/#sigma_{#eta fit",50,-10,10);
		h1["PhipRes"+suffix1] = new TH1D("PhipRes"+suffix1,"|#phi - #phi_{Tfit}|/#sigma_{#phi fit}",50,-10,10);
		h1["PhiqRes"+suffix1] = new TH1D("PhiqRes"+suffix1,"|#phi - #phi_{Tfit}|/#sigma_{#phi fit}",50,-10,10);
		h1["PhihbRes"+suffix1] = new TH1D("PhihbRes"+suffix1,"|#phi - #phi_{Tfit}|/#sigma_{#phi fit}",50,-10,10);
		h1["PhilbRes"+suffix1] = new TH1D("PhilbRes"+suffix1,"|#phi - #phi_{Tfit}|/#sigma_{#phi fit}",50,-10,10);
		h1["PhilRes"+suffix1] = new TH1D("PhilRes"+suffix1,"|#phi - #phi_{Tfit}|/#sigma_{#phi fit}",50,-10,10);
		h1["PhinRes"+suffix1] = new TH1D("PhinRes"+suffix1,"|#phi - #phi_{Tfit}|/#sigma_{#phi fit}",50,-10,10);
		
		
		h1["E0Res"+suffix1] = new TH1D("E0Res"+suffix1,"|E - E_{fit}|/#sigma_{Efit}",50,-10,10);
		h1["E1Res"+suffix1] = new TH1D("E1Res"+suffix1,"|E - E_{fit}|/#sigma_{Efit}",50,-10,10);
		h1["E2Res"+suffix1] = new TH1D("E2Res"+suffix1,"|E - E_{fit}|/#sigma_{Efit}",50,-10,10);
		h1["E3Res"+suffix1] = new TH1D("E3Res"+suffix1,"|E - E_{fit}|/#sigma_{Efit}",50,-10,10);
		h1["PzNuRes"+suffix1] = new TH1D("PzNuRes"+suffix1,"|Pz - Pz_{fit}|/#sigma_{Pzfit}",50,-10,10);
		
		h1["WTojj_nob"+suffix1] = new TH1D("WTojj_nob"+suffix1,"(jet+jet) mass [GeV/c^{2}]",80,0.0,300.0);
		h1["tTojjb"+suffix1] = new TH1D("tTojjb"+suffix1,"(jet+jet+b-jet) mass [GeV/c^{2}]",50,0.0,500.0);
		h1["tToWb"+suffix1] = new TH1D("tToWb"+suffix1,"(W_{jj} + b-jet) mass [GeV/c^{2}]",50,0.0,500.0);
		h1["Ht"+suffix1] = new TH1D("Ht"+suffix1,"Ht [GeV/c]",100,0.0,1000.0);
	}

	for(std::map<TString,TH1* >::const_iterator ih=h1.begin(); ih!=h1.end(); ++ih){
		TH1 *htemp = ih->second;
		htemp->SetXTitle( htemp->GetTitle() );
	}
	for(std::map<TString,TH2* >::const_iterator ih=h2.begin(); ih!=h2.end(); ++ih){
		TH2 *htemp = ih->second;
		htemp->SetXTitle( htemp->GetTitle() );
	}
}

//_______________________________________________________________
void BooHistograms::Fill1d(TString name, Double_t x, Double_t weight) {

	h1[name]->Fill(x,weight);
}

//_______________________________________________________________
void BooHistograms::Fill2d(TString name, Double_t x, Double_t y, Double_t weight) {

	h2[name]->Fill(x,y,weight);
	
}

//_______________________________________________________________
void BooHistograms::FillvsJets2d(TString name, Double_t x, edm::View<pat::Jet> jets, Double_t weight ) {

	if (jets.size() == 1 ) h2[name]->Fill(x,1,weight);
	if (jets.size() == 2 ) h2[name]->Fill(x,2,weight);
	if (jets.size() == 3 ) h2[name]->Fill(x,3,weight);
	if (jets.size() >= 4 ) h2[name]->Fill(x,4,weight);
		
}


//_______________________________________________________________
void BooHistograms::Print(TString extension, TString tag) {

	if ( tag != "" ) tag = "_"+tag;
                
	for(std::map<TString,TCanvas*>::const_iterator icv=cv_map.begin(); icv!=cv_map.end(); ++icv){

		TString tmpname = icv->first;
		TCanvas *acv = icv->second;
		acv->Print(TString(tmpname+tag+"."+extension));
	}               

}
//_______________________________________________________________
void BooHistograms::Save() {
	
	for(std::map<TString,TH1* >::const_iterator ih=h1.begin(); ih!=h1.end(); ++ih){
		//std::cout << "get histo: " << std::endl;
		TH1D *htemp = (TH1D*) ih->second;
		//std::cout << htemp->Print() << std::endl;
		if (htemp->GetEntries() > 0 ) htemp->Write();
	}
	for(std::map<TString,TH2* >::const_iterator ih=h2.begin(); ih!=h2.end(); ++ih){
		//std::cout << "get histo: " << std::endl;
		TH2D *htemp = (TH2D*) ih->second;
		//std::cout << htemp->Print() << std::endl;
		if (htemp->GetEntries() > 0 ) htemp->Write();
	}	
}

//_______________________________________________________________
void BooHistograms::SaveToFile(TString filename) {

	foutfile = new TFile(filename,"RECREATE");
	for(std::map<TString,TH1* >::const_iterator ih=h1.begin(); ih!=h1.end(); ++ih){
		//std::cout << "get histo: " << std::endl;
		TH1D *htemp = (TH1D*) ih->second;
		//std::cout << htemp->Print() << std::endl;
		htemp->Write();
	}
	for(std::map<TString,TH2* >::const_iterator ih=h2.begin(); ih!=h2.end(); ++ih){
		//std::cout << "get histo: " << std::endl;
		TH2D *htemp = (TH2D*) ih->second;
		//std::cout << htemp->Print() << std::endl;
		htemp->Write();
	}
                
	foutfile->Write();
	foutfile->Close();
	
}
