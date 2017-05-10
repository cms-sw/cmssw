#include "TwolSelection.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/View.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "CommonTools/CandAlgos/interface/CandMatcher.h"
#include "PhysicsTools/HepMCCandAlgos/interface/MCTruthPairSelector.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/Candidate/interface/CandMatchMap.h"
#include "RecoVertex/VertexTools/interface/VertexDistance3D.h"
#include "RecoVertex/VertexTools/interface/VertexDistanceXY.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/Common/interface/MergeableCounter.h"

#include "PhysicsTools/CandUtils/interface/CandMatcherNew.h"

#include "TrackingTools/IPTools/interface/IPTools.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"


using namespace std;
using namespace edm;
using namespace reco;
using namespace tools;
using namespace math;
using namespace reco::tau;

TwolSelection::TwolSelection(const edm::ParameterSet & iConfig) :
_relIsoCutE(0.15),
_relIsoCutMu(0.10),
_relIsoCutEloose(999.),
_relIsoCutMuloose(999.),
_chargeConsistency(true),
_minPt0(10.),
_minPt1(20.),
_tightD0Mu(0.01),
_tightD0E(0.02),
_looseD0Mu(0.01),
_looseD0E(0.02),
_jetPtCut(30.),
_jetEtaCut(2.5)
{
    Sample              = iConfig.getUntrackedParameter<std::string>("SampleLabel") ;
    IT_muon             = iConfig.getParameter<edm::InputTag>("MuonLabel") ;
    IT_electron         = iConfig.getParameter<edm::InputTag>("ElectronLabel") ;
    IT_jet              = iConfig.getParameter<edm::InputTag>("JetLabel");
    IT_pfmet            = iConfig.getParameter<edm::InputTag>("METLabel")  ;
    IT_beamspot         = iConfig.getParameter<edm::InputTag>("BeamSpotLabel");
    IT_hltresults       = iConfig.getParameter<edm::InputTag>("HLTResultsLabel");
}


void TwolSelection::beginJob()
{
    Nvtx           = fs->make<TH1F>("N_{vtx}"        , "Number of vertices;N_{vtx};events / 1"  ,    40, 0., 40.);
    
    _hCounter = fs->make<TH1D>("hCounter", "Events counter", 5,0,5);
    
    outputTree = new TTree("TwolTree","TwolTree");
    
    _leptonP4 = new TClonesArray("TLorentzVector", 4);
    for (int i=0; i!=4; ++i) {
        new ( (*_leptonP4)[i] ) TLorentzVector();
    }
    outputTree->Branch("_leptonP4", "TClonesArray", &_leptonP4, 32000, 0);
    
    _jetP4 = new TClonesArray("TLorentzVector", 20);
    for (int i=0; i!=20; ++i) {
        new ( (*_jetP4)[i] ) TLorentzVector();
    }
    //outputTree->Branch("_jetP4", "TClonesArray", &_jetP4, 32000, 0);
    
    
    outputTree->Branch("_eventNb",   &_eventNb,   "_eventNb/l");
    outputTree->Branch("_runNb",     &_runNb,     "_runNb/l");
    outputTree->Branch("_lumiBlock", &_lumiBlock, "_lumiBlock/l");
    
    outputTree->Branch("_nLeptons", &_nLeptons, "_nLeptons/I");
    outputTree->Branch("_flavors", &_flavors, "_flavors[4]/I");
    outputTree->Branch("_charges", &_charges, "_charges[4]/D");
    outputTree->Branch("_isolation", &_isolation, "_isolation[4]/D");
    outputTree->Branch("_mt", &_mt, "_mt[4]/D");
    outputTree->Branch("_isloose", &_isloose, "_isloose[4]/O");
    outputTree->Branch("_istight", &_istight, "_istight[4]/O");
    
    outputTree->Branch("_origin", &_origin, "_origin[4]/I");
    outputTree->Branch("_originReduced", &_originReduced, "_originReduced[4]/I");


    outputTree->Branch("_ipPV", &_ipPV, "_ipPV[4]/D");
    outputTree->Branch("_ipPVerr", &_ipPVerr, "_ipPVerr[4]/D");
    outputTree->Branch("_ipZPV", &_ipZPV, "_ipZPV[4]/D");
    outputTree->Branch("_ipZPVerr", &_ipZPVerr, "_ipZPVerr[4]/D");
    
    outputTree->Branch("_3dIP", &_3dIP, "_3dIP[4]/D");
    outputTree->Branch("_3dIPerr", &_3dIPerr, "_3dIPerr[4]/D");
    outputTree->Branch("_3dIPsig", &_3dIPsig, "_3dIPsig[4]/D");

    outputTree->Branch("_n_PV", &_n_PV, "_n_PV/I");
    outputTree->Branch("_PVchi2", &_PVchi2, "_PVchi2/D");
    outputTree->Branch("_PVerr", &_PVerr, "_PVerr[3]/D");
    
    outputTree->Branch("_met", &_met, "_met/D");
    outputTree->Branch("_met_phi", &_met_phi, "_met_phi/D");
    outputTree->Branch("HT", &HT, "HT/D");
    
    outputTree->Branch("_n_bJets", &_n_bJets, "_n_bJets/I");
    outputTree->Branch("_n_Jets", &_n_Jets, "_n_Jets/I");
    outputTree->Branch("_bTagged", &_bTagged, "_bTagged[20]/O");
    outputTree->Branch("_jetEta", &_jetEta, "_jetEta[20]/D");
    outputTree->Branch("_jetPhi", &_jetPhi, "_jetPhi[20]/D");
    outputTree->Branch("_jetPt", &_jetPt, "_jetPt[20]/D");
    outputTree->Branch("_csv", &_csv, "_csv[20]/D");
    
    _nEventsTotal = 0;
    _nEventsFiltered = 0;
    _nEventsTotalCounted = 0;
    
    GPM = GenParticleManager();
    
}

void TwolSelection::endJob() {
    // store nEventsTotal and nEventsFiltered in preferred way
    std::cout<<_nEventsTotal<<std::endl;
    std::cout<<_nEventsFiltered<<std::endl;
}

void TwolSelection::analyze(const edm::Event& iEvent, const edm::EventSetup& iEventSetup)
{
    edm::Handle<GenParticleCollection> TheGenParticles;
    //bool islepton;
    if (Sample=="ElectronsMC") {
        //******************************************************************************************************************
        // Gen level particles                  ****************************************************************************
        //******************************************************************************************************************
        iEvent.getByLabel("prunedGenParticles", TheGenParticles);
        if (TheGenParticles.isValid()) {
            GPM.SetCollection(TheGenParticles);
        }
    }
    
    _runNb = iEvent.id().run();
    _eventNb = iEvent.id().event();
    _lumiBlock = iEvent.luminosityBlock();

    //============ Total number of events is the sum of the events ============
    //============ in each of these luminosity blocks ============
    _nEventsTotalCounted++;
    _hCounter->Fill(0);
    /*edm::Handle<edm::MergeableCounter> nEventsTotalCounter;
     const edm::LuminosityBlock & lumi = iEvent.getLuminosityBlock();
     
     lumi.getByLabel("nEventsTotal", nEventsTotalCounter);
     _nEventsTotal += nEventsTotalCounter->value;
     _hCounter->Fill(1, nEventsTotalCounter->value);
     
     edm::Handle<edm::MergeableCounter> nEventsFilteredCounter;
     lumi.getByLabel("nEventsFiltered", nEventsFilteredCounter);
     _nEventsFiltered += nEventsFilteredCounter->value;
     _hCounter->Fill(2, nEventsFilteredCounter->value);
     _hCounter->Fill(3, double(nEventsTotalCounter->value)/double(nEventsFilteredCounter->value));
     */
    //============ Counter done ============
    
    //============ Beamspot ============
    edm::Handle< reco::BeamSpot > theBeamSpot;
    iEvent.getByLabel( IT_beamspot, theBeamSpot );
    if( ! theBeamSpot.isValid() ) ERR( IT_beamspot ) ;
    BeamSpot::Point  BS= theBeamSpot->position();;
    //==================================
    
    //============ Primary vertices ============
    edm::InputTag IT_goodVtx = edm::InputTag("goodOfflinePrimaryVertices");
    edm::Handle<std::vector<Vertex> > theVertices;
    iEvent.getByLabel( "goodOfflinePrimaryVertices", theVertices) ;
    if( ! theVertices.isValid() ) ERR(IT_goodVtx ) ;
    int nvertex = theVertices->size();
    
    _n_PV = nvertex;
    
    Nvtx->Fill(TMath::Min(nvertex,39));
    if(! nvertex ){
        cout << "[WARNING]: No candidate primary vertices passed the quality cuts, so skipping event" << endl;
        return ;
    }
    
    Vertex::Point PV = theVertices->begin()->position();
    const Vertex* PVtx = &((*theVertices)[0]);
    _PVchi2 = PVtx->chi2();
    _PVerr[0] = PVtx->xError();
    _PVerr[1] = PVtx->yError();
    _PVerr[2] = PVtx->zError();
    //==================================
    
    //============ Pat MET ============
    edm::Handle< vector<pat::MET> > ThePFMET;
    iEvent.getByLabel(IT_pfmet, ThePFMET);
    if( ! ThePFMET.isValid() ) ERR( IT_pfmet );
    const vector<pat::MET> *pfmetcol = ThePFMET.product();
    const pat::MET *pfmet;
    pfmet = &(pfmetcol->front());
    _met = pfmet->pt();
    _met_phi = pfmet->phi();
    //==================================
    
    //============ Pat Muons ============
    edm::Handle< std::vector<pat::Muon> > thePatMuons;
    iEvent.getByLabel( IT_muon, thePatMuons );
    if( ! thePatMuons.isValid() )  ERR(IT_muon) ;
    //==================================
    
    //============ Pat Electrons ============
    edm::Handle< std::vector<pat::Electron> > thePatElectrons;
    iEvent.getByLabel( IT_electron, thePatElectrons );
    if( ! thePatElectrons.isValid() ) ERR( IT_electron );
    //==================================
    
    //============ Conversions ============
    edm::Handle< std::vector<reco::Conversion> > theConversions;
    iEvent.getByLabel("reducedEgamma","reducedConversions", theConversions);
    //==================================
    
    //============ Pat Jets ============
    edm::Handle< std::vector< pat::Jet> > thePatJets;
    iEvent.getByLabel(IT_jet , thePatJets );
    if( ! thePatJets.isValid() ) ERR(IT_jet);
    //==================================
    
    //============ Rho ============
    edm::Handle<double> rhoJets;
    iEvent.getByLabel(edm::InputTag("fixedGridRhoAll","") , rhoJets);//kt6PFJets
    double myRhoJets = *rhoJets;
    //==================================
    
    //============= 3D IP ==============
    ESHandle<TransientTrackBuilder> theTTBuilder;
    iEventSetup.get<TransientTrackRecord>().get("TransientTrackBuilder",theTTBuilder);
    //==================================

    std::vector<const pat::Muon* > sMu = ssbMuonSelector( *thePatMuons, _minPt0, PV, _looseD0Mu);
    std::vector<const pat::Electron* > sEl = ssbElectronSelector( *thePatElectrons, _minPt0, PV, _looseD0E, _chargeConsistency, theConversions, BS);
    std::vector<const pat::Jet* > SelectedJets = JetSelector(*thePatJets, _jetPtCut, _jetEtaCut);

    if (sEl.size() + sMu.size() < 2) return;
    
    int leptonCounter = 0;
    
    //Muon selection
    for(unsigned int i = 0 ; i < sMu.size() ;i++ ){
        
        const pat::Muon *iM = sMu[i];
        
        if (leptonCounter == 4) continue;

        _flavors[leptonCounter] = 1;
        _charges[leptonCounter] = iM->charge();
        _isolation[leptonCounter] = pfRelIso(iM);
        
        _ipPV[leptonCounter] = TMath::Abs(iM->innerTrack()->dxy(PV));
        _ipPVerr[leptonCounter] = iM->innerTrack()->dxyError();
        
        _ipZPV[leptonCounter] = iM->innerTrack()->dz(PV);
        _ipZPVerr[leptonCounter] = iM->innerTrack()->dzError();
        
        const TrackRef siTrack = iM->innerTrack();
        
        if (siTrack.isNonnull()) {
            TransientTrack tt  = theTTBuilder->build(siTrack);
            Measurement1D ip3D = IPTools::absoluteImpactParameter3D(tt, *theVertices->begin()).second;
            _3dIP[leptonCounter]    = ip3D.value();
            _3dIPerr[leptonCounter] = ip3D.error();
            _3dIPsig[leptonCounter] = ip3D.significance();
            
        } else {
            _3dIP[leptonCounter] = -1;
            _3dIPerr[leptonCounter] = -1;
            _3dIPsig[leptonCounter] = -1;
        }
        
        _isloose[leptonCounter] = ( fabs(_ipPV[leptonCounter])<_looseD0Mu ) && (_isolation[leptonCounter]<_relIsoCutMuloose);
        if (_isloose[leptonCounter])
            _istight[leptonCounter] = ( fabs(_ipPV[leptonCounter])<_tightD0Mu ) && (_isolation[leptonCounter]<_relIsoCutMu);
        else _istight[leptonCounter] = false;
        
        ((TLorentzVector *)_leptonP4->At(leptonCounter))->SetPtEtaPhiE(iM->pt(), iM->eta(), iM->phi(), iM->energy());
        
        _mt[leptonCounter] = MT_calc(*((TLorentzVector *)_leptonP4->At(leptonCounter)), _met, _met_phi);
        
        
        if (Sample=="ElectronsMC") {
            //**************************************************************************************
            // MC
            //**************************************************************************************
            
            const GenParticle* mc = GPM.matchedMC(iM);
            if ( mc!=0 ) {
                _origin[leptonCounter] = GPM.origin(mc);
                _originReduced[leptonCounter] = GPM.originReduced(_origin[leptonCounter]);
                
                //GPM.printInheritance(&(*mc));
            }
            else {
                //std::cout<<"No match mu"<<std::endl;
                _origin[leptonCounter] = 4;
                _originReduced[leptonCounter] = 19;
            }
            //**************************************************************************************
            // MC *
            //**************************************************************************************
        }
        

        leptonCounter++;
    }
    
    //Electron selection
    for(unsigned int i = 0 ; i < sEl.size() ;i++ ){
        
        const pat::Electron *iE = sEl[i];
        if (leptonCounter == 4) continue;
        
        _flavors[leptonCounter] = 0;
        _charges[leptonCounter] = iE->charge();
        _isolation[leptonCounter] = pfRelIso(iE, myRhoJets);
        _ipPV[leptonCounter] = TMath::Abs(iE->gsfTrack()->dxy(PV));
        
        _isloose[leptonCounter] = ( fabs(_ipPV[leptonCounter])<_looseD0E ) && (_isolation[leptonCounter]<_relIsoCutEloose);
        if (_isloose[leptonCounter])
            _istight[leptonCounter] = ( fabs(_ipPV[leptonCounter])<_tightD0E ) && (_isolation[leptonCounter]<_relIsoCutE);
        else _istight[leptonCounter] = false;
        
        ((TLorentzVector *)_leptonP4->At(leptonCounter))->SetPtEtaPhiE(iE->pt(), iE->eta(), iE->phi(), iE->energy());
        
        _mt[leptonCounter] = MT_calc(*((TLorentzVector *)_leptonP4->At(leptonCounter)), _met, _met_phi);
        
        TransientTrack tt  = theTTBuilder->build(iE->gsfTrack());
        Measurement1D ip3D = IPTools::absoluteImpactParameter3D(tt, *theVertices->begin()).second;
        _3dIP[leptonCounter]    = ip3D.value();
        _3dIPerr[leptonCounter] = ip3D.error();
        _3dIPsig[leptonCounter] = ip3D.significance();
        
        if (Sample=="ElectronsMC") {
            //**************************************************************************************
            // MC
            //**************************************************************************************
            
            const GenParticle* mc = GPM.matchedMC(iE);
            if ( mc!=0 ) {
                _origin[leptonCounter] = GPM.origin(mc);
                _originReduced[leptonCounter] = GPM.originReduced(_origin[leptonCounter]);
                
                //GPM.printInheritance(&(*mc));
            }
            else {
                //std::cout<<"No match mu"<<std::endl;
                _origin[leptonCounter] = 4;
                _originReduced[leptonCounter] = 19;
            }
            //**************************************************************************************
            // MC *
            //**************************************************************************************
        }
        
        leptonCounter++;
    }
    
    if (leptonCounter < 2) return;
    
    _nLeptons = leptonCounter;
    
    _n_Jets = 0;
    _n_bJets = 0;
    HT = 0;
    for(unsigned int i = 0 ; i < SelectedJets.size() ;i++ ){
        _jetEta[_n_Jets] = SelectedJets[i]->eta();
        _jetPhi[_n_Jets] = SelectedJets[i]->phi();
        _jetPt[_n_Jets] = SelectedJets[i]->pt();
        
        TLorentzVector jt; jt.SetPtEtaPhiM(_jetPt[_n_Jets],_jetEta[_n_Jets],_jetPhi[_n_Jets],0);
        bool clean = true;
        for (int j=0; j!=_nLeptons; ++j) {
            if (!_istight[j]) {
                double dR = ((TLorentzVector *)_leptonP4->At(j))->DeltaR( jt );
                clean = clean && (dR > 0.4);
            }
        }
        if (!clean) continue;
        _csv[_n_Jets] = SelectedJets[i]->bDiscriminator("combinedSecondaryVertexBJetTags");
        
        if(_csv[_n_Jets] > 0.679) {
            _bTagged[_n_Jets] = true;
            _n_bJets++;
        } else _bTagged[_n_Jets] = false;
        
        HT+= _jetPt[_n_Jets];
        _n_Jets++;
    }

    outputTree->Fill();

    
}

DEFINE_FWK_MODULE(TwolSelection);