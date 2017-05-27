// system include files
//#include <memory>

// user include files
#include "Validation/RecoEgamma/plugins/ElectronMcMiniAODSignalValidator.h" 
#include "CLHEP/Units/GlobalPhysicalConstants.h"

// user include files

using namespace reco;
using namespace pat;

typedef edm::Ptr<pat::Electron> PatElectronPtr;

ElectronMcSignalValidatorMiniAOD::ElectronMcSignalValidatorMiniAOD(const edm::ParameterSet& iConfig) : ElectronDqmAnalyzerBase(iConfig)
{
   mcTruthCollection_ = consumes<edm::View<reco::GenParticle> >(iConfig.getParameter<edm::InputTag>("mcTruthCollection")); // prunedGenParticles
   electronToken_     = consumes<pat::ElectronCollection>      (iConfig.getParameter<edm::InputTag>("electrons"));         // slimmedElectrons

   edm::ParameterSet histosSet = iConfig.getParameter<edm::ParameterSet>("histosCfg") ;
   edm::ParameterSet isolationSet = iConfig.getParameter<edm::ParameterSet>("isolationCfg") ;

   //recomp
   pfSumChargedHadronPtTmp_  = consumes<edm::ValueMap<float> > (isolationSet.getParameter<edm::InputTag>( "pfSumChargedHadronPtTmp" ) ) ; // iConfig 
   pfSumNeutralHadronEtTmp_  = consumes<edm::ValueMap<float> > (isolationSet.getParameter<edm::InputTag>( "pfSumNeutralHadronEtTmp" ) ) ; // iConfig 
   pfSumPhotonEtTmp_ = consumes<edm::ValueMap<float> > (isolationSet.getParameter<edm::InputTag>( "pfSumPhotonEtTmp" ) ) ; // iConfig 
    
   maxPt_ = iConfig.getParameter<double>("MaxPt");
   maxAbsEta_ = iConfig.getParameter<double>("MaxAbsEta");
   deltaR_ = iConfig.getParameter<double>("DeltaR");
   deltaR2_ = deltaR_ * deltaR_;
   matchingIDs_ = iConfig.getParameter<std::vector<int> >("MatchingID");
   matchingMotherIDs_ = iConfig.getParameter<std::vector<int> >("MatchingMotherID");
   outputInternalPath_ = iConfig.getParameter<std::string>("OutputFolderName") ;

   // histos bining and limits

   xyz_nbin=histosSet.getParameter<int>("Nbinxyz");

   pt_nbin=histosSet.getParameter<int>("Nbinpt");
   pt2D_nbin=histosSet.getParameter<int>("Nbinpt2D");
   pteff_nbin=histosSet.getParameter<int>("Nbinpteff");
   pt_max=histosSet.getParameter<double>("Ptmax");

   fhits_nbin=histosSet.getParameter<int>("Nbinfhits");
   fhits_max=histosSet.getParameter<double>("Fhitsmax");

   eta_nbin=histosSet.getParameter<int>("Nbineta");
   eta2D_nbin=histosSet.getParameter<int>("Nbineta2D");
   eta_min=histosSet.getParameter<double>("Etamin");
   eta_max=histosSet.getParameter<double>("Etamax");

   detamatch_nbin=histosSet.getParameter<int>("Nbindetamatch");
   detamatch2D_nbin=histosSet.getParameter<int>("Nbindetamatch2D");
   detamatch_min=histosSet.getParameter<double>("Detamatchmin");
   detamatch_max=histosSet.getParameter<double>("Detamatchmax");

   dphi_nbin=histosSet.getParameter<int>("Nbindphi");
   dphi_min=histosSet.getParameter<double>("Dphimin");
   dphi_max=histosSet.getParameter<double>("Dphimax");

   dphimatch_nbin=histosSet.getParameter<int>("Nbindphimatch");
   dphimatch2D_nbin=histosSet.getParameter<int>("Nbindphimatch2D");
   dphimatch_min=histosSet.getParameter<double>("Dphimatchmin");
   dphimatch_max=histosSet.getParameter<double>("Dphimatchmax");

   hoe_nbin= histosSet.getParameter<int>("Nbinhoe");
   hoe_min=histosSet.getParameter<double>("Hoemin");
   hoe_max=histosSet.getParameter<double>("Hoemax");

   mee_nbin= histosSet.getParameter<int>("Nbinmee");
   mee_min=histosSet.getParameter<double>("Meemin");
   mee_max=histosSet.getParameter<double>("Meemax");

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

   h2_ele_foundHitsVsEta = 0 ;
   h2_ele_foundHitsVsEta_mAOD = 0 ;

   h2_ele_PoPtrueVsEta = 0 ;
   h2_ele_sigmaIetaIetaVsPt = 0 ;

   h1_ele_HoE_mAOD = 0 ;
   h1_ele_HoE_mAOD_barrel = 0 ;
   h1_ele_HoE_mAOD_endcaps = 0 ;
   h1_ele_mee_all = 0 ;
   h1_ele_mee_os = 0 ;

   h1_ele_fbrem_mAOD = 0 ;
   h1_ele_fbrem_mAOD_barrel = 0 ;
   h1_ele_fbrem_mAOD_endcaps = 0 ;
   
   h1_ele_dEtaSc_propVtx_mAOD = 0 ;
   h1_ele_dEtaSc_propVtx_mAOD_barrel = 0 ;
   h1_ele_dEtaSc_propVtx_mAOD_endcaps = 0 ;
   h1_ele_dPhiCl_propOut_mAOD = 0 ;
   h1_ele_dPhiCl_propOut_mAOD_barrel = 0 ;
   h1_ele_dPhiCl_propOut_mAOD_endcaps = 0 ;

   h1_ele_chargedHadronRelativeIso_mAOD = 0 ;
   h1_ele_chargedHadronRelativeIso_mAOD_barrel = 0 ;
   h1_ele_chargedHadronRelativeIso_mAOD_endcaps = 0 ;
   h1_ele_neutralHadronRelativeIso_mAOD = 0 ;
   h1_ele_neutralHadronRelativeIso_mAOD_barrel = 0 ;
   h1_ele_neutralHadronRelativeIso_mAOD_endcaps = 0 ;
   h1_ele_photonRelativeIso_mAOD = 0 ;
   h1_ele_photonRelativeIso_mAOD_barrel = 0 ;
   h1_ele_photonRelativeIso_mAOD_endcaps = 0 ;

   h1_ele_chargedHadronRelativeIso_mAOD_recomp = 0 ;
   h1_ele_neutralHadronRelativeIso_mAOD_recomp = 0 ;
   h1_ele_photonRelativeIso_mAOD_recomp = 0 ;    

}

ElectronMcSignalValidatorMiniAOD::~ElectronMcSignalValidatorMiniAOD()
{
}

void ElectronMcSignalValidatorMiniAOD::bookHistograms( DQMStore::IBooker & iBooker, edm::Run const &, edm::EventSetup const & )
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
//  h2_ele_sigmaIetaIetaVsPt = bookH2(iBooker,"sigmaIetaIetaVsPt","SigmaIetaIeta vs pt",pt_nbin,0.,pt_max,100,0.,0.05);
  h2_ele_sigmaIetaIetaVsPt = bookH2(iBooker,"sigmaIetaIetaVsPt","SigmaIetaIeta vs pt",100,0.,pt_max,100,0.,0.05);

  // all electrons
  setBookPrefix("h_ele") ;
  h1_ele_mee_all = bookH1withSumw2(iBooker, "mee_all","ele pairs invariant mass, all reco electrons",mee_nbin, mee_min, mee_max,"m_{ee} (GeV/c^{2})","Events","ELE_LOGY E1 P");
  h1_ele_mee_os = bookH1withSumw2(iBooker, "mee_os","ele pairs invariant mass, opp. sign",mee_nbin, mee_min, mee_max,"m_{e^{+}e^{-}} (GeV/c^{2})","Events","ELE_LOGY E1 P");

  // matched electron, superclusters
  setBookPrefix("h_scl") ;
  h1_scl_SigIEtaIEta_mAOD = bookH1withSumw2(iBooker, "SigIEtaIEta_mAOD","ele supercluster sigma ieta ieta",100,0.,0.05,"#sigma_{i#eta i#eta}","Events","ELE_LOGY E1 P");
  h1_scl_SigIEtaIEta_mAOD_barrel = bookH1withSumw2(iBooker, "SigIEtaIEta_mAOD_barrel","ele supercluster sigma ieta ieta, barrel",100,0.,0.05,"#sigma_{i#eta i#eta}","Events","ELE_LOGY E1 P");
  h1_scl_SigIEtaIEta_mAOD_endcaps = bookH1withSumw2(iBooker, "SigIEtaIEta_mAOD_endcaps","ele supercluster sigma ieta ieta, endcaps",100,0.,0.05,"#sigma_{i#eta i#eta}","Events","ELE_LOGY E1 P");
 
  // matched electron, gsf tracks
  setBookPrefix("h_ele") ;
  h2_ele_foundHitsVsEta = bookH2(iBooker, "foundHitsVsEta","ele track # found hits vs eta",eta2D_nbin,eta_min,eta_max,fhits_nbin,0.,fhits_max);
  h2_ele_foundHitsVsEta_mAOD = bookH2(iBooker, "foundHitsVsEta_mAOD","ele track # found hits vs eta",eta2D_nbin,eta_min,eta_max,fhits_nbin,0.,fhits_max);

  // matched electrons, matching
  setBookPrefix("h_ele") ;
  h1_ele_HoE_mAOD = bookH1withSumw2(iBooker, "HoE_mAOD","ele hadronic energy / em energy",hoe_nbin, hoe_min, hoe_max,"H/E","Events","ELE_LOGY E1 P") ;
  h1_ele_HoE_mAOD_barrel = bookH1withSumw2(iBooker, "HoE_mAOD_barrel","ele hadronic energy / em energy, barrel",hoe_nbin, hoe_min, hoe_max,"H/E","Events","ELE_LOGY E1 P") ;
  h1_ele_HoE_mAOD_endcaps = bookH1withSumw2(iBooker, "HoE_mAOD_endcaps","ele hadronic energy / em energy, endcaps",hoe_nbin, hoe_min, hoe_max,"H/E","Events","ELE_LOGY E1 P") ;
  h1_ele_dEtaSc_propVtx_mAOD = bookH1withSumw2(iBooker, "dEtaSc_propVtx_mAOD","ele #eta_{sc} - #eta_{tr}, prop from vertex",detamatch_nbin,detamatch_min,detamatch_max,"#eta_{sc} - #eta_{tr}","Events","ELE_LOGY E1 P");
  h1_ele_dEtaSc_propVtx_mAOD_barrel = bookH1withSumw2(iBooker, "dEtaSc_propVtx_mAOD_barrel","ele #eta_{sc} - #eta_{tr}, prop from vertex, barrel",detamatch_nbin,detamatch_min,detamatch_max,"#eta_{sc} - #eta_{tr}","Events","ELE_LOGY E1 P");
  h1_ele_dEtaSc_propVtx_mAOD_endcaps = bookH1withSumw2(iBooker, "dEtaSc_propVtx_mAOD_endcaps","ele #eta_{sc} - #eta_{tr}, prop from vertex, endcaps",detamatch_nbin,detamatch_min,detamatch_max,"#eta_{sc} - #eta_{tr}","Events","ELE_LOGY E1 P");
  h1_ele_dPhiCl_propOut_mAOD = bookH1withSumw2(iBooker, "dPhiCl_propOut_mAOD","ele #phi_{cl} - #phi_{tr}, prop from outermost",dphimatch_nbin,dphimatch_min,dphimatch_max,"#phi_{seedcl} - #phi_{tr} (rad)","Events","ELE_LOGY E1 P");
  h1_ele_dPhiCl_propOut_mAOD_barrel = bookH1withSumw2(iBooker, "dPhiCl_propOut_mAOD_barrel","ele #phi_{cl} - #phi_{tr}, prop from outermost, barrel",dphimatch_nbin,dphimatch_min,dphimatch_max,"#phi_{seedcl} - #phi_{tr} (rad)","Events","ELE_LOGY E1 P");
  h1_ele_dPhiCl_propOut_mAOD_endcaps = bookH1withSumw2(iBooker, "dPhiCl_propOut_mAOD_endcaps","ele #phi_{cl} - #phi_{tr}, prop from outermost, endcaps",dphimatch_nbin,dphimatch_min,dphimatch_max,"#phi_{seedcl} - #phi_{tr} (rad)","Events","ELE_LOGY E1 P");

  // fbrem
  h1_ele_fbrem_mAOD = bookH1withSumw2(iBooker, "fbrem_mAOD","ele brem fraction, mode of GSF components",100,0.,1.,"P_{in} - P_{out} / P_{in}");
  h1_ele_fbrem_mAOD_barrel = bookH1withSumw2(iBooker, "fbrem_mAOD_barrel","ele brem fraction for barrel, mode of GSF components", 100, 0.,1.,"P_{in} - P_{out} / P_{in}");
  h1_ele_fbrem_mAOD_endcaps = bookH1withSumw2(iBooker, "fbrem_mAOD_endcaps", "ele brem franction for endcaps, mode of GSF components", 100, 0.,1.,"P_{in} - P_{out} / P_{in}");

  // -- pflow over pT
  h1_ele_chargedHadronRelativeIso_mAOD = bookH1withSumw2(iBooker, "chargedHadronRelativeIso_mAOD","chargedHadronRelativeIso",100,0.0,2.,"chargedHadronRelativeIso","Events","ELE_LOGY E1 P");
  h1_ele_chargedHadronRelativeIso_mAOD_barrel = bookH1withSumw2(iBooker, "chargedHadronRelativeIso_mAOD_barrel","chargedHadronRelativeIso for barrel",100,0.0,2.,"chargedHadronRelativeIso_barrel","Events","ELE_LOGY E1 P");
  h1_ele_chargedHadronRelativeIso_mAOD_endcaps = bookH1withSumw2(iBooker, "chargedHadronRelativeIso_mAOD_endcaps","chargedHadronRelativeIso for endcaps",100,0.0,2.,"chargedHadronRelativeIso_endcaps","Events","ELE_LOGY E1 P");
  h1_ele_neutralHadronRelativeIso_mAOD = bookH1withSumw2(iBooker, "neutralHadronRelativeIso_mAOD","neutralHadronRelativeIso",100,0.0,2.,"neutralHadronRelativeIso","Events","ELE_LOGY E1 P");
  h1_ele_neutralHadronRelativeIso_mAOD_barrel = bookH1withSumw2(iBooker, "neutralHadronRelativeIso_mAOD_barrel","neutralHadronRelativeIso for barrel",100,0.0,2.,"neutralHadronRelativeIso_barrel","Events","ELE_LOGY E1 P");
  h1_ele_neutralHadronRelativeIso_mAOD_endcaps = bookH1withSumw2(iBooker, "neutralHadronRelativeIso_mAOD_endcaps","neutralHadronRelativeIso for endcaps",100,0.0,2.,"neutralHadronRelativeIso_endcaps","Events","ELE_LOGY E1 P");
  h1_ele_photonRelativeIso_mAOD = bookH1withSumw2(iBooker, "photonRelativeIso_mAOD","photonRelativeIso",100,0.0,2.,"photonRelativeIso","Events","ELE_LOGY E1 P");
  h1_ele_photonRelativeIso_mAOD_barrel = bookH1withSumw2(iBooker, "photonRelativeIso_mAOD_barrel","photonRelativeIso for barrel",100,0.0,2.,"photonRelativeIso_barrel","Events","ELE_LOGY E1 P");
  h1_ele_photonRelativeIso_mAOD_endcaps = bookH1withSumw2(iBooker, "photonRelativeIso_mAOD_endcaps","photonRelativeIso for endcaps",100,0.0,2.,"photonRelativeIso_endcaps","Events","ELE_LOGY E1 P");

    // -- recomputed pflow over pT
  h1_ele_chargedHadronRelativeIso_mAOD_recomp = bookH1withSumw2(iBooker, "chargedHadronRelativeIso_mAOD_recomp","recomputed chargedHadronRelativeIso",100,0.0,2.,"chargedHadronRelativeIso","Events","ELE_LOGY E1 P");
  h1_ele_neutralHadronRelativeIso_mAOD_recomp = bookH1withSumw2(iBooker, "neutralHadronRelativeIso_mAOD_recomp","recomputed neutralHadronRelativeIso",100,0.0,2.,"neutralHadronRelativeIso","Events","ELE_LOGY E1 P");
  h1_ele_photonRelativeIso_mAOD_recomp = bookH1withSumw2(iBooker, "photonRelativeIso_mAOD_recomp","recomputed photonRelativeIso",100,0.0,2.,"photonRelativeIso","Events","ELE_LOGY E1 P");

 }
 
void ElectronMcSignalValidatorMiniAOD::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
    // get collections
    edm::Handle<pat::ElectronCollection> electrons;
    iEvent.getByToken(electronToken_, electrons);
    
    edm::Handle<edm::View<reco::GenParticle> > genParticles ;
    iEvent.getByToken(mcTruthCollection_, genParticles) ;  

    //recomp
    edm::Handle <edm::ValueMap <float> > pfSumChargedHadronPtTmp;
    edm::Handle <edm::ValueMap <float> > pfSumNeutralHadronEtTmp;
    edm::Handle <edm::ValueMap <float> > pfSumPhotonEtTmp;/**/
  
    //recomp
    iEvent.getByToken( pfSumChargedHadronPtTmp_ , pfSumChargedHadronPtTmp);
    iEvent.getByToken( pfSumNeutralHadronEtTmp_ , pfSumNeutralHadronEtTmp);
    iEvent.getByToken( pfSumPhotonEtTmp_ , pfSumPhotonEtTmp);/**/

    edm::LogInfo("ElectronMcSignalValidatorMiniAOD::analyze") << "Treating event " << iEvent.id() << " with " << electrons.product()->size() << " electrons" ;
    h1_recEleNum->Fill((*electrons).size()) ;

    //===============================================
    // all rec electrons
    //===============================================

    pat::Electron gsfElectron ;

    pat::ElectronCollection::const_iterator el1 ;
    pat::ElectronCollection::const_iterator el2 ;
    for(el1=electrons->begin(); el1!=electrons->end(); el1++) {
        for (el2=el1+1 ; el2!=electrons->end() ; el2++ )
        {
            math::XYZTLorentzVector p12 = el1->p4()+el2->p4();
            float mee2 = p12.Dot(p12);
            h1_ele_mee_all->Fill(sqrt(mee2));
            if ( el1->charge() * el2->charge() < 0. )
            {
                h1_ele_mee_os->Fill(sqrt(mee2));
            }
        }
    }

    //===============================================
    // charge mis-ID
    //===============================================

    int mcNum=0, gamNum=0, eleNum=0 ;
//    bool matchingID;//, matchingMotherID ;
    bool matchingMotherID ;

    //===============================================
    // association mc-reco
    //===============================================

    for(size_t i=0; i<genParticles->size(); i++) {  
/*                // DEBUG LINES - KEEP IT !
        std::cout << "\nevt ID = " << iEvent.id() ; 
        std::cout << ",  mcIter position : " << i << std::endl; 
        std::cout << "pdgID : " << (*genParticles)[i].pdgId() << ", Pt : " << (*genParticles)[i].pt() ; 
        std::cout << ", eta : " << (*genParticles)[i].eta() << ", phi : " << (*genParticles)[i].phi() << std::endl; 
                // DEBUG LINES - KEEP IT !  */ 

                // number of mc particles
        mcNum++ ;

        // counts photons
        if ( (*genParticles)[i].pdgId() == 22 )       
            { gamNum++ ; }

        // select requested mother matching gen particle
        // always include single particle with no mother
        const Candidate * mother = (*genParticles)[i].mother(0) ;
        matchingMotherID = false ;
        for ( unsigned int ii=0 ; ii<matchingMotherIDs_.size() ; ii++ ) {

/*                // DEBUG LINES - KEEP IT !
                std::cout << "Matching : matchingMotherID[" << ii << "] : "<< matchingMotherIDs_[ii]  << ", evt ID = " << iEvent.id() << ", mother : "  << mother ; 
                if (mother != 0) { 
			        std::cout << "mother : " << mother << ", mother pdgID : " << mother->pdgId() << std::endl ; 
                    std::cout << "mother pdgID : " << mother->pdgId() << ", Pt : " << mother->pt() << ", eta : " << mother->eta() << ", phi : " << mother->phi() << std::endl; 
                }
                else { 
                    std::cout << std::endl; 
                } 
                // DEBUG LINES - KEEP IT !  */ 

            if ( mother == 0 ) {
                matchingMotherID = true ; 
            }
            else if ( mother->pdgId() == matchingMotherIDs_[ii] ) { 
                if ( mother->numberOfDaughters() <= 2 ) {
                    matchingMotherID = true ;
                    //std::cout << "evt ID = " << iEvent.id() ;                                                                               // debug lines
                    //std::cout << " - nb of Daughters : " << mother->numberOfDaughters() << " - pdgId() : " << mother->pdgId() << std::endl; // debug lines
                }
            } // end of mother if test

/*                // DEBUG LINES - KEEP IT !
            if (mother != 0) { 
                std::cout << "mother : " << mother << ", mother pdgID : " << mother->pdgId() << std::endl ; 
                std::cout << "mother pdgID : " << mother->pdgId() << ", Pt : " << mother->pt() << ", eta : " << mother->eta() << ", phi : " << mother->phi() << std::endl; 
            } 
                // DEBUG LINES - KEEP IT !  */ 
        } // end of for loop
    if (!matchingMotherID) {continue ;}

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
            double dphi = el.phi()-(*genParticles)[i].phi() ;
            if (std::abs(dphi)>CLHEP::pi)
                { dphi = dphi < 0? (CLHEP::twopi) + dphi : dphi - CLHEP::twopi ; }
            double deltaR2 = (el.eta()-(*genParticles)[i].eta()) * (el.eta()-(*genParticles)[i].eta()) + dphi * dphi;
            if ( deltaR2 < deltaR2_ )
            {
                if ( ( ((*genParticles)[i].pdgId() == 11) && (el.charge() < 0.) ) ||
                     ( ((*genParticles)[i].pdgId() == -11) && (el.charge() > 0.) ) )
                {
                    double tmpGsfRatio = el.p()/(*genParticles)[i].p() ;
                    if ( std::abs(tmpGsfRatio-1) < std::abs(gsfOkRatio-1) )
                        {
                        gsfOkRatio = tmpGsfRatio;
                        bestGsfElectron=el;
                        PatElectronPtr elePtr(electrons, &el-&(*electrons)[0]) ;
                        pt_ = elePtr->pt();
                        sumChargedHadronPt_recomp    =  (*pfSumChargedHadronPtTmp)[elePtr];
                        relisoChargedHadronPt_recomp = sumChargedHadronPt_recomp/pt_;

                        sumNeutralHadronPt_recomp    = (*pfSumNeutralHadronEtTmp)[elePtr];
                        relisoNeutralHadronPt_recomp = sumNeutralHadronPt_recomp/pt_;

                        sumPhotonPt_recomp    =  (*pfSumPhotonEtTmp)[elePtr];
                        relisoPhotonPt_recomp = sumPhotonPt_recomp/pt_;

                        okGsfFound = true;
                        
                // DEBUG LINES - KEEP IT !
                //        std::cout << "evt ID : " << iEvent.id() << " - Pt : " << bestGsfElectron.pt() << " - eta : " << bestGsfElectron.eta() << " - phi : " << bestGsfElectron.phi() << std::endl;
                // DEBUG LINES - KEEP IT !  /**/ 
                    }
                }
            }
        }
     
        if (! okGsfFound) continue ;

        //------------------------------------
        // analysis when the mc track is found
        //------------------------------------
        passMiniAODSelection = bestGsfElectron.pt() >= 5.;

        // electron related distributions
        h1_ele_vertexPt->Fill( bestGsfElectron.pt() );
        h1_ele_vertexEta->Fill( bestGsfElectron.eta() );
        if ( (bestGsfElectron.scSigmaIEtaIEta()==0.) && (bestGsfElectron.fbrem()==0.) ) h1_ele_vertexPt_nocut->Fill( bestGsfElectron.pt() );
    
        // generated distributions for matched electrons
        h2_ele_PoPtrueVsEta->Fill( bestGsfElectron.eta(), bestGsfElectron.p()/(*genParticles)[i].p());
        if ( passMiniAODSelection ) { // Pt > 5.
            h2_ele_sigmaIetaIetaVsPt->Fill( bestGsfElectron.pt(), bestGsfElectron.scSigmaIEtaIEta());
        }

        // supercluster related distributions
        if ( passMiniAODSelection ) { // Pt > 5.
            h1_scl_SigIEtaIEta_mAOD->Fill(bestGsfElectron.scSigmaIEtaIEta());
            h1_ele_dEtaSc_propVtx_mAOD->Fill(bestGsfElectron.deltaEtaSuperClusterTrackAtVtx());
            h1_ele_dPhiCl_propOut_mAOD->Fill(bestGsfElectron.deltaPhiSeedClusterTrackAtCalo());
            if (bestGsfElectron.isEB()) {
                h1_scl_SigIEtaIEta_mAOD_barrel->Fill(bestGsfElectron.scSigmaIEtaIEta());
                h1_ele_dEtaSc_propVtx_mAOD_barrel->Fill(bestGsfElectron.deltaEtaSuperClusterTrackAtVtx());
                h1_ele_dPhiCl_propOut_mAOD_barrel->Fill(bestGsfElectron.deltaPhiSeedClusterTrackAtCalo());
            }
            if (bestGsfElectron.isEE()) {
                h1_scl_SigIEtaIEta_mAOD_endcaps->Fill(bestGsfElectron.scSigmaIEtaIEta());
                h1_ele_dEtaSc_propVtx_mAOD_endcaps->Fill(bestGsfElectron.deltaEtaSuperClusterTrackAtVtx());
                h1_ele_dPhiCl_propOut_mAOD_endcaps->Fill(bestGsfElectron.deltaPhiSeedClusterTrackAtCalo());
            }

        }
   
        // track related distributions
        h2_ele_foundHitsVsEta->Fill( bestGsfElectron.eta(), bestGsfElectron.gsfTrack()->numberOfValidHits() );
        if (passMiniAODSelection) { // Pt > 5.
            h2_ele_foundHitsVsEta_mAOD->Fill( bestGsfElectron.eta(), bestGsfElectron.gsfTrack()->numberOfValidHits() );
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
            if (bestGsfElectron.isEB()) h1_ele_fbrem_mAOD_barrel->Fill( bestGsfElectron.fbrem() );
            if (bestGsfElectron.isEE()) h1_ele_fbrem_mAOD_endcaps->Fill( bestGsfElectron.fbrem() );

        // -- pflow over pT
            double one_over_pt = 1. / bestGsfElectron.pt();

            h1_ele_chargedHadronRelativeIso_mAOD->Fill(bestGsfElectron.pfIsolationVariables().sumChargedHadronPt * one_over_pt );
            h1_ele_neutralHadronRelativeIso_mAOD->Fill(bestGsfElectron.pfIsolationVariables().sumNeutralHadronEt * one_over_pt );
            h1_ele_photonRelativeIso_mAOD->Fill(bestGsfElectron.pfIsolationVariables().sumPhotonEt * one_over_pt );

            if (bestGsfElectron.isEB()) {
                h1_ele_chargedHadronRelativeIso_mAOD_barrel->Fill(bestGsfElectron.pfIsolationVariables().sumChargedHadronPt * one_over_pt );
                h1_ele_neutralHadronRelativeIso_mAOD_barrel->Fill(bestGsfElectron.pfIsolationVariables().sumNeutralHadronEt * one_over_pt );
                h1_ele_photonRelativeIso_mAOD_barrel->Fill(bestGsfElectron.pfIsolationVariables().sumPhotonEt * one_over_pt );
            }

            if (bestGsfElectron.isEE()) {
                h1_ele_chargedHadronRelativeIso_mAOD_endcaps->Fill(bestGsfElectron.pfIsolationVariables().sumChargedHadronPt * one_over_pt );
                h1_ele_neutralHadronRelativeIso_mAOD_endcaps->Fill(bestGsfElectron.pfIsolationVariables().sumNeutralHadronEt * one_over_pt );
                h1_ele_photonRelativeIso_mAOD_endcaps->Fill(bestGsfElectron.pfIsolationVariables().sumPhotonEt * one_over_pt );
            }

        // -- recomputed pflow over pT
            h1_ele_chargedHadronRelativeIso_mAOD_recomp->Fill( relisoChargedHadronPt_recomp );
            h1_ele_neutralHadronRelativeIso_mAOD_recomp->Fill( relisoNeutralHadronPt_recomp );
            h1_ele_photonRelativeIso_mAOD_recomp->Fill( relisoPhotonPt_recomp );/**/
                        
        }

    } // fin boucle size_t i

}

