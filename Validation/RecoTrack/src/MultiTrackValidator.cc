#include "Validation/RecoTrack/interface/MultiTrackValidator.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

#include "Math/ProbFuncMathMore.h"

using namespace ROOT::Math;

void MultiTrackValidator::beginJob( const EventSetup & setup) {

  for (unsigned int j=0;j<label.size();j++){
      
    vector<double> etaintervalsv;
    vector<double> hitsetav;
    vector<int>    totSIMv,totRECv;
    vector<TH1F*>  ptdistribv;
    vector<TH1F*>  etadistribv;
  
    double step=(max-min)/nint;
    ostringstream title,name;
    etaintervalsv.push_back(0);
    for (double d=min;d<max;d=d+step) {
      etaintervalsv.push_back(d+step);
      totSIMv.push_back(0);
      totRECv.push_back(0);
      hitsetav.push_back(0);
      name.str("");
      title.str("");
      name <<"pt["<<d<<","<<d+step<<"]";
      title <<"p_{t} residue "<< d << "<#eta<"<<d+step;
      ptdistribv.push_back(new TH1F(name.str().c_str(),title.str().c_str(), 200, -2, 2 ));
      name.str("");
      title.str("");
      name <<"eta["<<d<<","<<d+step<<"]";
      title <<"eta residue "<< d << "<#eta<"<<d+step;
      etadistribv.push_back(new TH1F(name.str().c_str(),title.str().c_str(), 200, -0.2, 0.2 ));
    }
    etaintervals.push_back(etaintervalsv);
    totSIM.push_back(totSIMv);
    totREC.push_back(totRECv);
    hitseta.push_back(hitsetav);
    ptdistrib.push_back(ptdistribv);
    etadistrib.push_back(etadistribv);
     
    h_ptSIM.push_back( new TH1F("ptSIM", "generated p_{t}", 5500, 0, 110 ) );
    h_etaSIM.push_back( new TH1F("etaSIM", "generated pseudorapidity", 500, 0, 5 ) );
    h_tracksSIM.push_back( new TH1F("tracksSIM","number of simluated tracks",100,-0.5,99.5) );
    h_vertposSIM.push_back( new TH1F("vertposSIM","Transverse position of sim vertices",1000,-0.5,10000.5) );
      
    //     h_pt     = new TH1F("pt", "p_{t} residue", 2000, -500, 500 );
    h_pt.push_back( new TH1F("pullPt", "pull of p_{t}", 100, -10, 10 ) );
    h_pt2.push_back( new TH1F("pt2", "p_{t} residue (#tracks>1)", 300, -15, 15 ) );
    h_eta.push_back( new TH1F("eta", "pseudorapidity residue", 1000, -0.1, 0.1 ) );
    h_tracks.push_back( new TH1F("tracks","number of reconstructed tracks",10,-0.5,9.5) );
    h_nchi2.push_back( new TH1F("nchi2", "normalized chi2", 200, 0, 20 ) );
    h_nchi2_prob.push_back( new TH1F("chi2_prob", "normalized chi2 probability",100,0,1));
    h_hits.push_back( new TH1F("hits", "number of hits per track", 30, -0.5, 29.5 ) );
    h_effic.push_back( new TH1F("effic","efficiency vs #eta",nint,&etaintervals[j][0]) );
    h_ptrmsh.push_back( new TH1F("PtRMS","PtRMS vs #eta",nint,&etaintervals[j][0]) );
    h_deltaeta.push_back( new TH1F("etaRMS","etaRMS vs #eta",nint,&etaintervals[j][0]) );
    h_hits_eta.push_back( new TH1F("hits_eta","hits_eta",nint,&etaintervals[j][0]) );
    h_charge.push_back( new TH1F("charge","charge",3,-1.5,1.5) );
      
    h_pullTheta.push_back( new TH1F("pullTheta","pull of theta parameter",100,-10,10) );
    h_pullPhi0.push_back( new TH1F("pullPhi0","pull of phi0 parameter",100,-10,10) );
    h_pullD0.push_back( new TH1F("pullD0","pull of d0 parameter",100,-10,10) );
    h_pullDz.push_back( new TH1F("pullDz","pull of dz parameter",100,-10,10) );
    h_pullK.push_back( new TH1F("pullK","pull of k parameter",100,-10,10) );
      
    chi2_vs_nhits.push_back( new TH2F("chi2_vs_nhits","chi2 vs nhits",25,0,25,100,0,10) );
    chi2_vs_eta.push_back( new TH2F("chi2_vs_eta","chi2 vs eta",nint,min,max,100,0,10) );
    nhits_vs_eta.push_back( new TH2F("nhits_vs_eta","nhits vs eta",nint,min,max,25,0,25) );
    ptres_vs_eta.push_back( new TH2F("ptres_vs_eta","ptresidue vs eta",nint,min,max,200,-2,2) );
    etares_vs_eta.push_back( new TH2F("etares_vs_eta","etaresidue vs eta",nint,min,max,200,-0.1,0.1) );
      
    h_assochi2.push_back( new TH1F("assochi2","track association chi2",200,0,50) );
    h_assochi2_prob.push_back(new TH1F("assochi2prob","probability of association chi2",100,0,1));
  }

  setup.get<IdealMagneticFieldRecord>().get(theMF);  
  associator = new TrackAssociatorByChi2(setup);
}

void MultiTrackValidator::analyze(const edm::Event& event, const edm::EventSetup& setup){

  edm::Handle<SimTrackContainer> simTrackCollection;
  event.getByLabel(sim, simTrackCollection);
  const SimTrackContainer simTC = *(simTrackCollection.product());
  
  edm::Handle<SimVertexContainer> simVertexCollection;
  event.getByLabel(sim, simVertexCollection);
  const SimVertexContainer simVC = *(simVertexCollection.product());
  
  for (unsigned int w=0;w<label.size();w++){
    //
    //get collections from the event
    //
    edm::Handle<reco::TrackCollection> trackCollection;
    event.getByLabel(label[w], trackCollection);
    const reco::TrackCollection tC = *(trackCollection.product());

    //     edm::Handle<TrackingParticleCollection>  TPCollectionH ;
    //     event.getByLabel("trackingtruth","TrackTruth",TPCollectionH);
    //     const TrackingParticleCollection tPC = *(TPCollectionH.product());
      
    //       reco::RecoToSimCollection p = associator->compareTracksParam(trackCollection,TPCollectionH);
    TrackAssociatorByChi2::RecoToSimPairAssociation q =  associator->compareTracksParam(tC,simTC,simVC);

    //
    //fill simulation histograms
    //
    int st=0;
    for (SimTrackContainer::const_iterator simTrack=simTC.begin(); simTrack!=simTC.end(); simTrack++){
      if (abs(simTrack->momentum().pseudoRapidity())>max || 
	  abs(simTrack->momentum().pseudoRapidity())<min) continue;
      st++;
      h_ptSIM[w]->Fill(simTrack->momentum().perp());
      h_etaSIM[w]->Fill(simTrack->momentum().pseudoRapidity());
      h_vertposSIM[w]->Fill(simVC[simTrack->vertIndex()].position().perp());
	
    }
    if (st!=0) h_tracksSIM[w]->Fill(st);
      
    //
    //fill reconstructed track histograms
    //
    int rt=0;
    for (TrackAssociatorByChi2::RecoToSimPairAssociation::iterator vit=q.begin();vit!=q.end();++vit){

      if (vit->second.size()==0) continue;

      reco::Track * track = &(vit->first);
      SimTrack * assocTrack = &(vit->second.begin()->second);

      h_assochi2[w]->Fill(vit->second.begin()->first);//chi2 of best association
      h_assochi2_prob[w]->Fill(chisquared_prob((vit->second.begin()->first)*5,5));

      if (abs(track->eta())>max || abs(track->eta())<min) continue;

      rt++;

      //nchi2 and hits global distributions
      h_nchi2[w]->Fill(track->normalizedChi2());
      h_nchi2_prob[w]->Fill(chisquared_prob(track->chi2(),track->ndof()));
      h_hits[w]->Fill(track->numberOfValidHits());
      chi2_vs_nhits[w]->Fill(track->numberOfValidHits(),track->normalizedChi2());
      chi2_vs_eta[w]->Fill(track->eta(),track->normalizedChi2());
      nhits_vs_eta[w]->Fill(track->eta(),track->numberOfValidHits());
      //       h_hits_eta[w]->Fill(track->eta(),track->numberOfValidHits());
      h_charge[w]->Fill( track->charge() );

      //pt, eta residue, theta, phi0, d0, dz pull
      double ptres=track->pt()-assocTrack->momentum().perp(); 
      double etares=track->eta()-assocTrack->momentum().pseudoRapidity();
      double thetares=(track->theta()-assocTrack->momentum().theta())/track->thetaError();
      double phi0res=(track->phi0()-assocTrack->momentum().phi())/track->phi0Error();
      double d0res=(track->d0()-simVC[assocTrack->vertIndex()].position().perp())/track->d0Error();
      double dzres=(track->dz()-simVC[assocTrack->vertIndex()].position().z())/track->dzError();
      const HepLorentzVector vertexPosition = simVC[assocTrack->vertIndex()].position(); 
      GlobalVector magField=theMF->inTesla(GlobalPoint(vertexPosition.x(),vertexPosition.y(),vertexPosition.z()));
      double simTrCurv = -track->charge()*2.99792458e-3 * magField.z()/assocTrack->momentum().perp();
      double kres=(track->transverseCurvature()-simTrCurv)/track->transverseCurvatureError();

      h_pt[w]->Fill(ptres/(track->transverseCurvatureError()
			   /track->transverseCurvature()*track-> pt()));
      h_eta[w]->Fill(etares);
      ptres_vs_eta[w]->Fill(track->eta(),ptres);
      etares_vs_eta[w]->Fill(track->eta(),etares);
      h_pullTheta[w]->Fill(thetares);
      h_pullPhi0[w]->Fill(phi0res);
      h_pullD0[w]->Fill(d0res);
      h_pullDz[w]->Fill(dzres);
      h_pullK[w]->Fill(kres);
	
      //pt residue distribution per eta interval
      int i=0;
      for (vector<TH1F*>::iterator h=ptdistrib[w].begin(); h!=ptdistrib[w].end(); h++){
	if (abs(assocTrack->momentum().pseudoRapidity())>etaintervals[w][i]&&
	    abs(assocTrack->momentum().pseudoRapidity())<etaintervals[w][i+1]) {
	  (*h)->Fill(track->pt()-assocTrack->momentum().perp());
	}
	i++;
      }
	
      //eta residue distribution per eta interval
      i=0;
      for (vector<TH1F*>::iterator h=etadistrib[w].begin(); h!=etadistrib[w].end(); h++){
	if (abs(assocTrack->momentum().pseudoRapidity())>etaintervals[w][i]&&
	    abs(assocTrack->momentum().pseudoRapidity())<etaintervals[w][i+1]) {
	  (*h)->Fill(etares);
	}
	i++;
      }

      //compute number of tracks per eta interval with pt residue better than 10% pt simulated
      i=0;
      for (vector<double>::iterator h=etaintervals[w].begin(); h!=etaintervals[w].end()-1; h++){
	if (abs(assocTrack->momentum().pseudoRapidity())>etaintervals[w][i]&&
	    abs(assocTrack->momentum().pseudoRapidity())<etaintervals[w][i+1]) {
	  totSIM[w][i]++;
	  if (abs(track->pt()-assocTrack->momentum().perp())<(assocTrack->momentum().perp()*0.1)) {
	    totREC[w][i]++;
	    hitseta[w][i]+=track->numberOfValidHits();
	  }
	}
	i++;
      }
    }
    
    if (rt!=0) h_tracks[w]->Fill(rt);
  }

}

void MultiTrackValidator::endJob() {

  delete associator;

  for (unsigned int w=0;w<label.size();w++){
    TDirectory * p = hFile->mkdir(label[w].c_str());
      
    //write simulation histos
    TDirectory * simD = p->mkdir("simulation");
    simD->cd();
    h_ptSIM[w]->Write();
    h_etaSIM[w]->Write();
    h_tracksSIM[w]->Write();
    h_vertposSIM[w]->Write();
      
    //fill pt rms plot versus eta and write pt residue distribution per eta interval histo
    TDirectory * ptD = p->mkdir("ptdistribution");
    ptD->cd();
    int i=0;
    for (vector<TH1F*>::iterator h=ptdistrib[w].begin(); h!=ptdistrib[w].end(); h++){
      (*h)->Write();
      h_ptrmsh[w]->Fill(etaintervals[w][i+1]-0.00001 ,(*h)->GetRMS());
      i++;
    }
      
    //fill eta rms plot versus eta and write eta residue distribution per eta interval histo
    TDirectory * etaD = p->mkdir("etadistribution");
    etaD->cd();
    i=0;
    for (vector<TH1F*>::iterator h=etadistrib[w].begin(); h!=etadistrib[w].end(); h++){
      (*h)->Write();
      h_deltaeta[w]->Fill(etaintervals[w][i+1]-0.00001 ,(*h)->GetRMS());
      i++;
    }
      
    //write the other histos
    p->cd();
    int j=0;
    for (vector<int>::iterator h=totSIM[w].begin(); h!=totSIM[w].end(); h++){
      if (totSIM[w][j])
	h_effic[w]->Fill(etaintervals[w][j+1]-0.00001, ((double) totREC[w][j])/((double) totSIM[w][j]));
      else h_effic[w]->Fill(etaintervals[w][j+1]-0.00001, 0);
      j++;
    }

    for (unsigned int rr=0; rr<hitseta[w].size(); rr++){
      if (totREC[w][rr])
      h_hits_eta[w]->Fill(etaintervals[w][rr+1]-0.00001,((double)  hitseta[w][rr])/((double) totREC[w][rr]));
      else h_effic[w]->Fill(etaintervals[w][j+1]-0.00001, 0);
      rr++;
    }
      
    h_pt[w]->Write();
    h_pt2[w]->Write();
    h_eta[w]->Write();
    h_tracks[w]->Write();
    h_nchi2[w]->Write();
    h_nchi2_prob[w]->Write();
    h_hits[w]->Write();
    h_effic[w]->Write();
    h_ptrmsh[w]->Write();
    h_deltaeta[w]->Write();
    chi2_vs_nhits[w]->Write();
    chi2_vs_eta[w]->Write();
    nhits_vs_eta[w]->Write();
    ptres_vs_eta[w]->Write();
    etares_vs_eta[w]->Write();
    h_charge[w]->Write();
      
    h_pullTheta[w]->Write();
    h_pullPhi0[w]->Write();
    h_pullD0[w]->Write();
    h_pullDz[w]->Write();
    h_pullK[w]->Write();

    h_assochi2[w]->Write();
    h_assochi2_prob[w]->Write();
    h_hits_eta[w]->Write();
  }

  hFile->Close();

}




