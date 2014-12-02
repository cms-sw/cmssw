#include "SUSYAnalyzer/PatAnalyzer/interface/Tools.h"

void tools::ERR( edm::InputTag& IT )
{
    cerr << "[ERROR] : " << IT << " is not a valid input label for this event.  SKIPPING EVENT " << endl;
}

//Muon pfRelIso
double tools::pfRelIso(const pat::Muon *mu)
{
    double chargedHadronIso = mu->pfIsolationR03().sumChargedHadronPt;
    double neutralHadronIso = mu->pfIsolationR03().sumNeutralHadronEt;
    double photonIso = mu->pfIsolationR03().sumPhotonEt;
    double beta = mu->pfIsolationR03().sumPUPt;
    double pfRelIsoMu  = ( chargedHadronIso + TMath::Max ( 0.0 ,neutralHadronIso + photonIso - 0.5 * beta ) )/mu->pt() ;
    return pfRelIsoMu;
}

//Electron pfRelIso
double tools::pfRelIso(const pat::Electron *iE, double myRho)
{
    double  Aeff[ 7 ] = { 0.13, 0.14, 0.07, 0.09, 0.11, 0.11, 0.14  };
    double CorrectedTerm=0.0;

    if( TMath::Abs( iE->superCluster()->eta() ) < 1.0 ) CorrectedTerm = myRho * Aeff[ 0 ];
    else if( TMath::Abs( iE->superCluster()->eta() ) > 1.0 && TMath::Abs( iE->superCluster()->eta() ) < 1.479  )   CorrectedTerm = myRho * Aeff[ 1 ];
    else if( TMath::Abs( iE->superCluster()->eta() ) > 1.479 && TMath::Abs( iE->superCluster()->eta() ) < 2.0  )   CorrectedTerm = myRho * Aeff[ 2 ];
    else if( TMath::Abs( iE->superCluster()->eta() ) > 2.0 && TMath::Abs( iE->superCluster()->eta() ) < 2.2  )     CorrectedTerm = myRho * Aeff[ 3 ];
    else if( TMath::Abs( iE->superCluster()->eta() ) > 2.2 && TMath::Abs( iE->superCluster()->eta() ) < 2.3  )     CorrectedTerm = myRho * Aeff[ 4 ];
    else if( TMath::Abs( iE->superCluster()->eta() ) > 2.3 && TMath::Abs( iE->superCluster()->eta() ) < 2.4  )     CorrectedTerm = myRho * Aeff[ 5 ];
    else CorrectedTerm = myRho * Aeff[ 6 ];
    
    double pfRelIsoE = (iE->chargedHadronIso() + TMath::Max(0.0, iE->neutralHadronIso() + iE->photonIso() - CorrectedTerm ) ) /iE->pt() ;

    return pfRelIsoE;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
std::vector<const pat::Muon* > tools::ssbMuonSelector(const std::vector<pat::Muon>  & thePatMuons,
                                                      double v_muon_pt,
                                                      reco::Vertex::Point PV,
                                                      double v_muon_d0)
{
    double v_muon_eta = 2.4;
    double v_muon_dz = 0.1;

    int v_muon_numberOfMatchedStations = 2;
    int v_muon_nPixelValidHits = 1;
    int v_muon_numberOfValidMuonHits = 1;
    int v_muon_nValidHits = 6;
    double v_muon_chi2Norm = 10.;
    
    double v_muon_emVetoEt=4.;
    double v_muon_hadVetoEt= 6.;
    
    std::vector<const pat::Muon* > vMuons;
    for( std::vector<pat::Muon>::const_iterator mu = thePatMuons.begin() ; mu != thePatMuons.end() ; mu++ )
	{
        if ( mu->pt()  < v_muon_pt ) continue;
        if ( TMath::Abs( mu->eta() ) > v_muon_eta ) continue;
        //if ( !mu->isTrackerMuon() ) continue;
        if ( !mu->isGlobalMuon()  ) continue;
        if ( !mu->isPFMuon() ) continue;
        if ( mu->numberOfMatchedStations() < v_muon_numberOfMatchedStations ) continue;   //we should add this to skim
        
        const reco::TrackRef innerTrack = mu->innerTrack();
        if( innerTrack.isNull() ) continue;
        if( innerTrack->hitPattern().trackerLayersWithMeasurement() < v_muon_nValidHits ) continue;
        if( innerTrack->hitPattern().numberOfValidPixelHits() < v_muon_nPixelValidHits  ) continue;
        
        const reco::TrackRef globalTrack = mu->globalTrack() ;
        if( globalTrack.isNull() ) continue;
        if( globalTrack->normalizedChi2() > v_muon_chi2Norm ) continue;
        if( globalTrack->hitPattern().numberOfValidMuonHits() < v_muon_numberOfValidMuonHits ) continue;
        
        if(TMath::Abs(innerTrack->dxy(PV)) > v_muon_d0  ) continue;
        if(TMath::Abs(innerTrack->dz(PV))   > v_muon_dz  ) continue;
        
        if( mu->isolationR03().emVetoEt > v_muon_emVetoEt ) continue;
        if( mu->isolationR03().hadVetoEt> v_muon_hadVetoEt ) continue;
        
        vMuons.push_back(&*mu);
    }
    
    return vMuons;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<const pat::Electron* > tools::ssbElectronSelector(const std::vector<pat::Electron>  & thePatElectrons,
                                                              double v_electron_pt,
                                                              reco::Vertex::Point PV,
                                                              double v_electron_d0,
                                                              bool bool_electron_chargeConsistency,
                                                              edm::Handle< std::vector<reco::Conversion> > &theConversions,
                                                              reco::BeamSpot::Point BS)
{
    double v_electron_eta = 2.4;
    double v_electron_dz = 0.1;
    //bool bool_electron_ecalDriven = true;
    
    std::vector<const pat::Electron* > vElectrons;
    for( std::vector<pat::Electron>::const_iterator el = thePatElectrons.begin() ; el != thePatElectrons.end() ; el++ ) {
        const reco::GsfTrackRef gsfTrack = el->gsfTrack();
        if (!gsfTrack.isNonnull()) {
            continue;
        }
        
        if( el->pt() < v_electron_pt ) continue;
        if( TMath::Abs(el->eta()) > v_electron_eta ) continue;
        if( TMath::Abs(el->superCluster()->eta()) > 1.4442 && TMath::Abs(el->superCluster()->eta()) < 1.566 ) continue;
        //if( bool_electron_ecalDriven && !el->ecalDrivenSeed() ) continue;
        
        //if (!el->trackerDrivenSeed() ) continue;
        
        if( TMath::Abs(gsfTrack->dxy(PV)) > v_electron_d0  )  continue;
        if( TMath::Abs(gsfTrack->dz(PV)) > v_electron_dz  ) continue;
        
        if( bool_electron_chargeConsistency && !el->isGsfCtfScPixChargeConsistent() )  continue;

        /*
        if( el->pt() < 20. ){
            if( ! ( el->fbrem() > 0.15 || ( TMath::Abs( el->superCluster()->eta() ) < 1.0 && el->eSuperClusterOverP() > 0.95 ) ) ) continue;
        }*/
        
        if( TMath::Abs(1.0/el->ecalEnergy() - el->eSuperClusterOverP()/el->ecalEnergy()) > 0.05 ) continue;
        
        bool vtxFitConversion = ConversionTools::hasMatchedConversion(reco::GsfElectron (*el), theConversions, BS);
        if( vtxFitConversion )  continue;
        
        if( gsfTrack->hitPattern().numberOfHits(reco::HitPattern::MISSING_INNER_HITS) > 0 ) continue;
        
        if( TMath::Abs( el->eta()) < 1.5  ) {
            if( TMath::Abs(el->deltaPhiSuperClusterTrackAtVtx()) > 0.06  ) continue;
            if( TMath::Abs(el->deltaEtaSuperClusterTrackAtVtx()) > 0.004 ) continue;
            if( TMath::Abs(el->scSigmaIEtaIEta()) > 0.01 ) continue;
            if( TMath::Abs(el->hadronicOverEm())  > 0.1  ) continue;  //recommended is 0.12 but HLT applies 0.1
	    }
        else if( TMath::Abs( el->eta() ) < 2.4 ) {
            if( TMath::Abs(el->deltaPhiSuperClusterTrackAtVtx()) > 0.03 ) continue;
            if( TMath::Abs(el->deltaEtaSuperClusterTrackAtVtx()) > 0.007 ) continue;
            if( TMath::Abs(el->scSigmaIEtaIEta()) > 0.03 ) continue;
            if( TMath::Abs(el->hadronicOverEm()) > 0.075 ) continue;   /// at the HLT 0.075  recommended is 0.1
	    }
        
        vElectrons.push_back(&*el );
	}
    return vElectrons;
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


std::vector<const pat::Jet* > tools::JetSelector(const std::vector<pat::Jet>  & thePatJets,
                                                 double  v_jet_pt,
                                                 double  v_jet_eta)
{
    bool    bool_jet_id= true;
    
    std::vector< const pat::Jet* > vJets;
    
    for( std::vector<pat::Jet>::const_iterator jet = thePatJets.begin(); jet != thePatJets.end(); jet++ )
	{
        if( jet->pt() < v_jet_pt )continue;
        if( TMath::Abs( jet->eta() ) > v_jet_eta) continue;
        if( bool_jet_id )
	    {
            if( jet->neutralHadronEnergyFraction() >= 0.99 ) continue;
            if( jet->neutralEmEnergyFraction() >= 0.99 ) continue;
            if( ( jet->neutralHadronMultiplicity() + jet->chargedHadronMultiplicity() ) < 2 ) continue;
            if( TMath::Abs( jet->eta() ) < 2.4 )
            {
                if( jet->chargedHadronEnergyFraction() == 0. ) continue;
                if( jet->chargedEmEnergyFraction() >= 0.99 ) continue;
                if( jet->chargedMultiplicity() == 0 ) continue;
            }
	    }
        vJets.push_back( &*jet );
    }
    return vJets;
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
std::vector<const pat::Jet* > tools::JetSelector(const std::vector<pat::Jet>  & thePatJets,
                                                 double  v_jet_pt,
                                                 double  v_jet_eta,
                                                 std::vector<const pat::Electron*> vElectrons,
                                                 std::vector<const pat::Muon*> vMuons)
{
    bool    bool_jet_id= true;
    double  v_jet_leptonVetoDR=0.4;
    //double  v_jet_leptonVetoDR = -1.;
    
    std::vector< const pat::Jet* > vJets;
    
    for( std::vector<pat::Jet>::const_iterator jet = thePatJets.begin(); jet != thePatJets.end(); jet++ )
	{
        if( jet->pt() < v_jet_pt )continue;
        if( TMath::Abs( jet->eta() ) > v_jet_eta) continue;
        if( bool_jet_id )
	    {
            if( jet->neutralHadronEnergyFraction() >= 0.99 ) continue;
            if( jet->neutralEmEnergyFraction() >= 0.99 ) continue;
            if( ( jet->neutralHadronMultiplicity() + jet->chargedHadronMultiplicity() ) < 2 ) continue;
            if( TMath::Abs( jet->eta() ) < 2.4 )
            {
                if( jet->chargedHadronEnergyFraction() == 0. ) continue;
                if( jet->chargedEmEnergyFraction() >= 0.99 ) continue;
                if( jet->chargedMultiplicity() == 0 ) continue;
            }
	    }
        
        bool vetoJet = false;
        for(unsigned int i = 0 ; i < vMuons.size() ;i++ )
        {
            const pat::Muon *mu = vMuons[i];
            float dphi = TMath::ACos( TMath::Cos( mu->phi()-jet->phi() ));
            float deta = mu->eta()-jet->eta();
            float dr = TMath::Sqrt( dphi*dphi + deta*deta) ;
            
            if(dr < v_jet_leptonVetoDR )
            {
                vetoJet = true;
                break;
            }
	    }
        if( vetoJet ) continue;
        
        for(unsigned int i = 0 ; i < vElectrons.size() ;i++ )
        {
            const pat::Electron *el = vElectrons[i];
            float dphi = TMath::ACos( TMath::Cos( el->phi()-jet->phi() ) );
            float deta = el->eta() - jet->eta();
            float dr = TMath::Sqrt( dphi*dphi + deta*deta );
            
            if(dr < v_jet_leptonVetoDR)
            {
                vetoJet = true;
                break;
            }
	    }
	    
        if( vetoJet ) continue;
	    
        
        vJets.push_back( &*jet );
    }
    return vJets;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

double tools::MT_calc(TLorentzVector Vect, double MET, double MET_Phi){
    
    double MT=sqrt(2* Vect.Pt() * MET * ( 1 - (TMath::Cos(Vect.Phi() - MET_Phi )) ) );
    
    return MT;
}

double tools::Mll_calc(TLorentzVector Vect1, TLorentzVector Vect2){
    return (Vect1 + Vect2).Mag();
}