// system include files
#include <memory>

// user include files
#include "Validation/RecoEgamma/plugins/ElectronMcMiniAODSignalValidator.h" 
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/PackedGenParticle.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "TMath.h"
#include "TFile.h"
#include "TH1F.h"
#include "TH1I.h"
#include "TH2F.h"
#include "TProfile.h"
#include "TTree.h"
#include <vector>
#include <iostream>
#include <typeinfo>

using namespace reco;
using namespace pat;

ElectronMcMiniAODSignalValidator::ElectronMcMiniAODSignalValidator(const edm::ParameterSet& iConfig) : ElectronDqmAnalyzerBase(iConfig)
{
    vtxToken_ = consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("vertices"));
    mcTruthCollection_ = consumes<edm::View<reco::GenParticle> >(iConfig.getParameter<edm::InputTag>("mcTruthCollection"));
    electronToken_ = consumes<pat::ElectronCollection>(iConfig.getParameter<edm::InputTag>("electrons"));

  maxPt_ = iConfig.getParameter<double>("MaxPt");
  maxAbsEta_ = iConfig.getParameter<double>("MaxAbsEta");
  deltaR_ = iConfig.getParameter<double>("DeltaR");
  matchingIDs_ = iConfig.getParameter<std::vector<int> >("MatchingID");
  matchingMotherIDs_ = iConfig.getParameter<std::vector<int> >("MatchingMotherID");
  outputInternalPath_ = iConfig.getParameter<std::string>("OutputFolderName") ;

  // histos bining and limits

   edm::ParameterSet histosSet = iConfig.getParameter<edm::ParameterSet>("histosCfg") ;

   xyz_nbin=histosSet.getParameter<int>("Nbinxyz");

   pt_nbin=histosSet.getParameter<int>("Nbinpt");
   pt2D_nbin=histosSet.getParameter<int>("Nbinpt2D");
   pteff_nbin=histosSet.getParameter<int>("Nbinpteff");
   pt_max=histosSet.getParameter<double>("Ptmax");
   
   eta_nbin=histosSet.getParameter<int>("Nbineta");
   eta2D_nbin=histosSet.getParameter<int>("Nbineta2D");
   eta_min=histosSet.getParameter<double>("Etamin");
   eta_max=histosSet.getParameter<double>("Etamax");

  detamatch_nbin=histosSet.getParameter<int>("Nbindetamatch");
  detamatch2D_nbin=histosSet.getParameter<int>("Nbindetamatch2D");
  detamatch_min=histosSet.getParameter<double>("Detamatchmin");
  detamatch_max=histosSet.getParameter<double>("Detamatchmax");

   hoe_nbin= histosSet.getParameter<int>("Nbinhoe");
   hoe_min=histosSet.getParameter<double>("Hoemin");
   hoe_max=histosSet.getParameter<double>("Hoemax");

   poptrue_nbin= histosSet.getParameter<int>("Nbinpoptrue");
   poptrue_min=histosSet.getParameter<double>("Poptruemin");
   poptrue_max=histosSet.getParameter<double>("Poptruemax");

   set_EfficiencyFlag=histosSet.getParameter<bool>("EfficiencyFlag");
   set_StatOverflowFlag=histosSet.getParameter<bool>("StatOverflowFlag");

   // so to please coverity...

   h1_recEleNum = 0 ;

   h1_ele_vertexPt = 0 ;
   h1_ele_vertexEta = 0 ;
   h1_ele_vertexPt_nocut = 0 ;

   h1_scl_SigIEtaIEta_mAOD = 0 ;
   h1_scl_SigIEtaIEta_mAOD_barrel = 0 ;
   h1_scl_SigIEtaIEta_mAOD_endcaps = 0 ;

   h2_ele_PoPtrueVsEta = 0 ;
   h2_ele_sigmaIetaIetaVsPt = 0 ;

   h1_ele_HoE_mAOD = 0 ;
   h1_ele_HoE_mAOD_barrel = 0 ;
   h1_ele_HoE_mAOD_endcaps = 0 ;

   h1_ele_fbrem_mAOD = 0 ;
   h1_ele_fbrem_barrel_mAOD = 0 ;
   h1_ele_fbrem_endcaps_mAOD = 0 ;
   
   h1_ele_dEtaSc_propVtx_mAOD = 0 ;
   h1_ele_dEtaSc_propVtx_mAOD_barrel = 0 ;
   h1_ele_dEtaSc_propVtx_mAOD_endcaps = 0 ;

   h1_ele_chargedHadronRelativeIso_mAOD = 0 ;
   h1_ele_chargedHadronRelativeIso_barrel_mAOD = 0 ;
   h1_ele_chargedHadronRelativeIso_endcaps_mAOD = 0 ;
   h1_ele_neutralHadronRelativeIso_mAOD = 0 ;
   h1_ele_neutralHadronRelativeIso_barrel_mAOD = 0 ;
   h1_ele_neutralHadronRelativeIso_endcaps_mAOD = 0 ;
   h1_ele_photonRelativeIso_mAOD = 0 ;
   h1_ele_photonRelativeIso_barrel_mAOD = 0 ;
   h1_ele_photonRelativeIso_endcaps_mAOD = 0 ;


}

ElectronMcMiniAODSignalValidator::~ElectronMcMiniAODSignalValidator()
{
}

void ElectronMcMiniAODSignalValidator::bookHistograms( DQMStore::IBooker & iBooker, edm::Run const &, edm::EventSetup const & )
 {
  iBooker.setCurrentFolder(outputInternalPath_) ;

  setBookIndex(-1) ;
  setBookPrefix("h") ;
  setBookEfficiencyFlag(set_EfficiencyFlag);
  setBookStatOverflowFlag( set_StatOverflowFlag ) ;

  // rec event collections sizes
  h1_recEleNum = bookH1(iBooker, "recEleNum","# rec electrons",11, -0.5,10.5,"N_{ele}");
  // matched electrons
  setBookPrefix("h_mc") ;
  setBookPrefix("h_ele") ;
  h1_ele_vertexPt = bookH1withSumw2(iBooker, "vertexPt","ele transverse momentum",pt_nbin,0.,pt_max,"p_{T vertex} (GeV/c)");
  h1_ele_vertexEta = bookH1withSumw2(iBooker, "vertexEta","ele momentum eta",eta_nbin,eta_min,eta_max,"#eta");
  h1_ele_vertexPt_nocut = bookH1withSumw2(iBooker, "vertexPt_nocut","pT of prunned electrons",pt_nbin,0.,20.,"p_{T vertex} (GeV/c)");
  h2_ele_PoPtrueVsEta = bookH2withSumw2(iBooker, "PoPtrueVsEta","ele momentum / gen momentum vs eta",eta2D_nbin,eta_min,eta_max,50,poptrue_min,poptrue_max);
  h2_ele_sigmaIetaIetaVsPt = bookH2(iBooker,"sigmaIetaIetaVsPt","SigmaIetaIeta vs pt",pt_nbin,0.,pt_max,100,0.,0.05);

  // matched electron, superclusters
  setBookPrefix("h_scl") ;
  h1_scl_SigIEtaIEta_mAOD = bookH1withSumw2(iBooker, "sigietaieta_mAOD","ele supercluster sigma ieta ieta",100,0.,0.05,"#sigma_{i#eta i#eta}","Events","ELE_LOGY E1 P");
  h1_scl_SigIEtaIEta_mAOD_barrel = bookH1withSumw2(iBooker, "SigIEtaIEta_mAOD_barrel","ele supercluster sigma ieta ieta, barrel",100,0.,0.05,"#sigma_{i#eta i#eta}","Events","ELE_LOGY E1 P");
  h1_scl_SigIEtaIEta_mAOD_endcaps = bookH1withSumw2(iBooker, "SigIEtaIEta_mAOD_endcaps","ele supercluster sigma ieta ieta, endcaps",100,0.,0.05,"#sigma_{i#eta i#eta}","Events","ELE_LOGY E1 P");
 
  // matched electrons, matching
  setBookPrefix("h_ele") ;
  h1_ele_HoE_mAOD = bookH1withSumw2(iBooker, "HoE_mAOD","ele hadronic energy / em energy",hoe_nbin, hoe_min, hoe_max,"H/E","Events","ELE_LOGY E1 P") ;
  h1_ele_HoE_mAOD_barrel = bookH1withSumw2(iBooker, "HoE_mAOD_barrel","ele hadronic energy / em energy, barrel",hoe_nbin, hoe_min, hoe_max,"H/E","Events","ELE_LOGY E1 P") ;
  h1_ele_HoE_mAOD_endcaps = bookH1withSumw2(iBooker, "HoE_mAOD_endcaps","ele hadronic energy / em energy, endcaps",hoe_nbin, hoe_min, hoe_max,"H/E","Events","ELE_LOGY E1 P") ;
  h1_ele_dEtaSc_propVtx_mAOD = bookH1withSumw2(iBooker, "dEtaSc_propVtx_mAOD","ele #eta_{sc} - #eta_{tr}, prop from vertex",detamatch_nbin,detamatch_min,detamatch_max,"#eta_{sc} - #eta_{tr}","Events","ELE_LOGY E1 P");
  h1_ele_dEtaSc_propVtx_mAOD_barrel = bookH1withSumw2(iBooker, "dEtaSc_propVtx_mAOD_barrel","ele #eta_{sc} - #eta_{tr}, prop from vertex, barrel",detamatch_nbin,detamatch_min,detamatch_max,"#eta_{sc} - #eta_{tr}","Events","ELE_LOGY E1 P");
  h1_ele_dEtaSc_propVtx_mAOD_endcaps = bookH1withSumw2(iBooker, "dEtaSc_propVtx_mAOD_endcaps","ele #eta_{sc} - #eta_{tr}, prop from vertex, endcaps",detamatch_nbin,detamatch_min,detamatch_max,"#eta_{sc} - #eta_{tr}","Events","ELE_LOGY E1 P");

  // fbrem
  h1_ele_fbrem_mAOD = bookH1withSumw2(iBooker, "fbrem_mAOD","ele brem fraction, mode of GSF components",100,0.,1.,"P_{in} - P_{out} / P_{in}");
  h1_ele_fbrem_barrel_mAOD = bookH1withSumw2(iBooker, "fbrem_barrel_mAOD","ele brem fraction for barrel, mode of GSF components", 100, 0.,1.,"P_{in} - P_{out} / P_{in}");
  h1_ele_fbrem_endcaps_mAOD = bookH1withSumw2(iBooker, "fbrem_endcaps_mAOD", "ele brem franction for endcaps, mode of GSF components", 100, 0.,1.,"P_{in} - P_{out} / P_{in}");

  // -- pflow over pT
  h1_ele_chargedHadronRelativeIso_mAOD = bookH1withSumw2(iBooker, "chargedHadronRelativeIso_mAOD","chargedHadronRelativeIso",100,0.0,2.,"chargedHadronRelativeIso","Events","ELE_LOGY E1 P");
  h1_ele_chargedHadronRelativeIso_barrel_mAOD = bookH1withSumw2(iBooker, "chargedHadronRelativeIso_barrel_mAOD","chargedHadronRelativeIso for barrel",100,0.0,2.,"chargedHadronRelativeIso_barrel","Events","ELE_LOGY E1 P");
  h1_ele_chargedHadronRelativeIso_endcaps_mAOD = bookH1withSumw2(iBooker, "chargedHadronRelativeIso_endcaps_mAOD","chargedHadronRelativeIso for endcaps",100,0.0,2.,"chargedHadronRelativeIso_endcaps","Events","ELE_LOGY E1 P");
  h1_ele_neutralHadronRelativeIso_mAOD = bookH1withSumw2(iBooker, "neutralHadronRelativeIso_mAOD","neutralHadronRelativeIso",100,0.0,2.,"neutralHadronRelativeIso","Events","ELE_LOGY E1 P");
  h1_ele_neutralHadronRelativeIso_barrel_mAOD = bookH1withSumw2(iBooker, "neutralHadronRelativeIso_barrel_mAOD","neutralHadronRelativeIso for barrel",100,0.0,2.,"neutralHadronRelativeIso_barrel","Events","ELE_LOGY E1 P");
  h1_ele_neutralHadronRelativeIso_endcaps_mAOD = bookH1withSumw2(iBooker, "neutralHadronRelativeIso_endcaps_mAOD","neutralHadronRelativeIso for endcaps",100,0.0,2.,"neutralHadronRelativeIso_endcaps","Events","ELE_LOGY E1 P");
  h1_ele_photonRelativeIso_mAOD = bookH1withSumw2(iBooker, "photonRelativeIso_mAOD","photonRelativeIso",100,0.0,2.,"photonRelativeIso","Events","ELE_LOGY E1 P");
  h1_ele_photonRelativeIso_barrel_mAOD = bookH1withSumw2(iBooker, "photonRelativeIso_barrel_mAOD","photonRelativeIso for barrel",100,0.0,2.,"photonRelativeIso_barrel","Events","ELE_LOGY E1 P");
  h1_ele_photonRelativeIso_endcaps_mAOD = bookH1withSumw2(iBooker, "photonRelativeIso_endcaps_mAOD","photonRelativeIso for endcaps",100,0.0,2.,"photonRelativeIso_endcaps","Events","ELE_LOGY E1 P");

 }
 
void ElectronMcMiniAODSignalValidator::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
    // get collections
    edm::Handle<reco::VertexCollection> vertices;
    iEvent.getByToken(vtxToken_, vertices);
    if (vertices->empty()) return; // skip the event if no PV found
/*    const reco::Vertex &PV = vertices->front();*/

    edm::Handle<pat::ElectronCollection> electrons;
    iEvent.getByToken(electronToken_, electrons);
    
    edm::Handle<edm::View<reco::GenParticle> > genParticles ;
    iEvent.getByToken(mcTruthCollection_, genParticles) ;  

    for (const pat::Electron &el : *electrons) {
        if (el.pt() < 5) continue;
//        printf("elec with pt %4.1f, supercluster eta %+5.3f, sigmaIetaIeta %.3f (%.3f with full5x5 shower shapes), lost hits %d, pass conv veto %d\n",
//                    el.pt(), el.superCluster()->eta(), el.sigmaIetaIeta(), el.full5x5_sigmaIetaIeta(), el.gsfTrack()->trackerExpectedHitsInner().numberOfLostHits(), el.passConversionVeto());
/*        printf("elec with pt %4.1f, supercluster eta %+5.3f, sigmaIetaIeta %.3f (%.3f with full5x5 shower shapes), pass conv veto %d\n",
                    el.pt(), el.superCluster()->eta(), el.sigmaIetaIeta(), el.full5x5_sigmaIetaIeta(), el.passConversionVeto());*/
    }

    edm::LogInfo("ElectronMcMiniAODSignalValidator::analyze")
      <<"Treating event "<<iEvent.id()
      <<" with "<<electrons.product()->size()<<" electrons" ;
    h1_recEleNum->Fill((*electrons).size()) ;

    //===============================================
    // all rec electrons
    //===============================================


  //===============================================
  // charge mis-ID
  //===============================================

  int mcNum=0, gamNum=0, eleNum=0 ;
//  bool matchingID;//, matchingMotherID ;
  bool matchingMotherID ;

  //===============================================
  // association mc-reco
  //===============================================

  for(size_t i=0; i<genParticles->size(); i++)
   {  
    // number of mc particles
    mcNum++ ;

    // counts photons
    if ( (*genParticles)[i].pdgId() == 22 )
     { gamNum++ ; }

    // select requested matching gen particle
/*    matchingID = false ;
    for ( unsigned int i=0 ; i<matchingIDs_.size() ; i++ )
     {
      if ( (*genParticles)[i].pdgId() == matchingIDs_[i] )
       { 
         matchingID=true ; 
         std::cout << "matchingID = TRUE" << " " << matchingIDs_[i] << std::endl; 
       }
     }
    if (!matchingID) continue ;
*/
    // select requested mother matching gen particle
    // always include single particle with no mother
    const Candidate * mother = (*genParticles)[i].mother(0) ;
    matchingMotherID = false ;
    for ( unsigned int ii=0 ; ii<matchingMotherIDs_.size() ; ii++ )
     {
      if ( (mother == 0) || ((mother != 0) &&  mother->pdgId() == matchingMotherIDs_[ii]) )
		{ matchingMotherID = true ; 
//			std::cout << "matchingMotherID :" << matchingMotherIDs_[ii] << std::endl ;
		}
     }
    if (!matchingMotherID) continue ;

    // electron preselection
    if ((*genParticles)[i].pt()> maxPt_ || std::abs((*genParticles)[i].eta())> maxAbsEta_)
     { continue ; }
    eleNum++;

    // find best matched electron
    bool okGsfFound = false ;
    bool passMiniAODSelection = true ;
    double gsfOkRatio = 999999. ;
    pat::Electron bestGsfElectron ;
    for (const pat::Electron &el : *electrons ) {
        passMiniAODSelection = el.pt() >= 5;
//        std::cout << "pt=" << el.pt() << ", bool=" << passMiniAODSelection << std::endl ;
		double dphi = el.phi()-(*genParticles)[i].phi() ;
		if (std::abs(dphi)>CLHEP::pi)
		{ dphi = dphi < 0? (CLHEP::twopi) + dphi : dphi - CLHEP::twopi ; }
		double deltaR = sqrt(pow((el.eta()-(*genParticles)[i].eta()),2) + pow(dphi,2));
		if ( deltaR < deltaR_ )
		{
			if ( ( ((*genParticles)[i].pdgId() == 11) && (el.charge() < 0.) ) ||
             ( ((*genParticles)[i].pdgId() == -11) && (el.charge() > 0.) ) )
			{
				double tmpGsfRatio = el.p()/(*genParticles)[i].p() ;
				if ( std::abs(tmpGsfRatio-1) < std::abs(gsfOkRatio-1) )
				{
					gsfOkRatio = tmpGsfRatio;
					bestGsfElectron=el;
					okGsfFound = true;
				}
			}
		}
    }
     
    if (! okGsfFound) continue ;

    //------------------------------------
    // analysis when the mc track is found
    //------------------------------------

    // electron related distributions
    h1_ele_vertexPt->Fill( bestGsfElectron.pt() );
    h1_ele_vertexEta->Fill( bestGsfElectron.eta() );
    if ( (bestGsfElectron.scSigmaIEtaIEta()==0.) && (bestGsfElectron.fbrem()==0.) ) h1_ele_vertexPt_nocut->Fill( bestGsfElectron.pt() );
    
    // generated distributions for matched electrons
    h2_ele_PoPtrueVsEta->Fill( bestGsfElectron.eta(), bestGsfElectron.p()/(*genParticles)[i].p());
    h2_ele_sigmaIetaIetaVsPt->Fill( bestGsfElectron.pt(), bestGsfElectron.scSigmaIEtaIEta());

    // supercluster related distributions
    if ( passMiniAODSelection ) { // Pt > 5.
        h1_scl_SigIEtaIEta_mAOD->Fill(bestGsfElectron.scSigmaIEtaIEta());
        if (bestGsfElectron.isEB()) h1_scl_SigIEtaIEta_mAOD_barrel->Fill(bestGsfElectron.scSigmaIEtaIEta());
        if (bestGsfElectron.isEE()) h1_scl_SigIEtaIEta_mAOD_endcaps->Fill(bestGsfElectron.scSigmaIEtaIEta());
    }
    if (passMiniAODSelection) { // Pt > 5.
        h1_ele_dEtaSc_propVtx_mAOD->Fill(bestGsfElectron.deltaEtaSuperClusterTrackAtVtx());
        if (bestGsfElectron.isEB()) h1_ele_dEtaSc_propVtx_mAOD_barrel->Fill(bestGsfElectron.deltaEtaSuperClusterTrackAtVtx());
        if (bestGsfElectron.isEE())h1_ele_dEtaSc_propVtx_mAOD_endcaps->Fill(bestGsfElectron.deltaEtaSuperClusterTrackAtVtx());
    }
   
    // match distributions
    if (passMiniAODSelection) { // Pt > 5.
        h1_ele_HoE_mAOD->Fill(bestGsfElectron.hcalOverEcal());
        if (bestGsfElectron.isEB()) h1_ele_HoE_mAOD_barrel->Fill(bestGsfElectron.hcalOverEcal());
        if (bestGsfElectron.isEE()) h1_ele_HoE_mAOD_endcaps->Fill(bestGsfElectron.hcalOverEcal());
    }

    // fbrem

//    double fbrem_mode =  bestGsfElectron.fbrem();
    if (passMiniAODSelection) { // Pt > 5.
        h1_ele_fbrem_mAOD->Fill( bestGsfElectron.fbrem() );
        if (bestGsfElectron.isEB()) h1_ele_fbrem_barrel_mAOD->Fill( bestGsfElectron.fbrem() );
        if (bestGsfElectron.isEE()) h1_ele_fbrem_endcaps_mAOD->Fill( bestGsfElectron.fbrem() );
    }

	// -- pflow over pT
    if (passMiniAODSelection) { // Pt > 5.
        h1_ele_chargedHadronRelativeIso_mAOD->Fill(bestGsfElectron.pfIsolationVariables().sumChargedHadronPt / bestGsfElectron.pt());
        if (bestGsfElectron.isEB()) h1_ele_chargedHadronRelativeIso_barrel_mAOD->Fill(bestGsfElectron.pfIsolationVariables().sumChargedHadronPt / bestGsfElectron.pt());
        if (bestGsfElectron.isEE()) h1_ele_chargedHadronRelativeIso_endcaps_mAOD->Fill(bestGsfElectron.pfIsolationVariables().sumChargedHadronPt / bestGsfElectron.pt());

        h1_ele_neutralHadronRelativeIso_mAOD->Fill(bestGsfElectron.pfIsolationVariables().sumNeutralHadronEt / bestGsfElectron.pt());
        if (bestGsfElectron.isEB()) h1_ele_neutralHadronRelativeIso_barrel_mAOD->Fill(bestGsfElectron.pfIsolationVariables().sumNeutralHadronEt / bestGsfElectron.pt());
        if (bestGsfElectron.isEE()) h1_ele_neutralHadronRelativeIso_endcaps_mAOD->Fill(bestGsfElectron.pfIsolationVariables().sumNeutralHadronEt / bestGsfElectron.pt());

        h1_ele_photonRelativeIso_mAOD->Fill(bestGsfElectron.pfIsolationVariables().sumPhotonEt / bestGsfElectron.pt());
        if (bestGsfElectron.isEB()) h1_ele_photonRelativeIso_barrel_mAOD->Fill(bestGsfElectron.pfIsolationVariables().sumPhotonEt / bestGsfElectron.pt());
        if (bestGsfElectron.isEE()) h1_ele_photonRelativeIso_endcaps_mAOD->Fill(bestGsfElectron.pfIsolationVariables().sumPhotonEt / bestGsfElectron.pt());
    }

    } // fin boucle mcIter

//    std::cout << ("fin analyze\n");
}

