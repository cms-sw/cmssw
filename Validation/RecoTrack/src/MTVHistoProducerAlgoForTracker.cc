#include "Validation/RecoTrack/interface/MTVHistoProducerAlgoForTracker.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/TrackReco/interface/DeDxData.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/Ref.h"
#include <DataFormats/TrackReco/interface/TrackFwd.h>

#include "DQMServices/ClientConfig/interface/FitSlicesYTool.h"

#include "TMath.h"
#include <TF1.h>

using namespace std;

MTVHistoProducerAlgoForTracker::MTVHistoProducerAlgoForTracker(const edm::ParameterSet& pset): MTVHistoProducerAlgo(pset){
  //parameters for _vs_eta plots
  minEta  = pset.getParameter<double>("minEta");  
  maxEta  = pset.getParameter<double>("maxEta");
  nintEta = pset.getParameter<int>("nintEta");
  useFabsEta = pset.getParameter<bool>("useFabsEta");

  //parameters for _vs_pt plots
  minPt  = pset.getParameter<double>("minPt");
  maxPt  = pset.getParameter<double>("maxPt");
  nintPt = pset.getParameter<int>("nintPt");
  useInvPt = pset.getParameter<bool>("useInvPt");
  useLogPt = pset.getUntrackedParameter<bool>("useLogPt",false);

  //parameters for _vs_Hit plots
  minHit  = pset.getParameter<double>("minHit");
  maxHit  = pset.getParameter<double>("maxHit");
  nintHit = pset.getParameter<int>("nintHit");
  
  //parameters for _vs_phi plots
  minPhi  = pset.getParameter<double>("minPhi");
  maxPhi  = pset.getParameter<double>("maxPhi");
  nintPhi = pset.getParameter<int>("nintPhi");

  //parameters for _vs_Dxy plots
  minDxy  = pset.getParameter<double>("minDxy");
  maxDxy  = pset.getParameter<double>("maxDxy");
  nintDxy = pset.getParameter<int>("nintDxy");
    
  //parameters for _vs_Dz plots
  minDz  = pset.getParameter<double>("minDz");
  maxDz  = pset.getParameter<double>("maxDz");
  nintDz = pset.getParameter<int>("nintDz");

  //parameters for _vs_ProductionVertexTransvPosition plots
  minVertpos  = pset.getParameter<double>("minVertpos");
  maxVertpos  = pset.getParameter<double>("maxVertpos");
  nintVertpos = pset.getParameter<int>("nintVertpos");

  //parameters for _vs_ProductionVertexZPosition plots
  minZpos  = pset.getParameter<double>("minZpos");
  maxZpos  = pset.getParameter<double>("maxZpos");
  nintZpos = pset.getParameter<int>("nintZpos");

  //parameters for dE/dx plots
  minDeDx  = pset.getParameter<double>("minDeDx");
  maxDeDx  = pset.getParameter<double>("maxDeDx");
  nintDeDx = pset.getParameter<int>("nintDeDx");
  
  //parameters for resolution plots
  ptRes_rangeMin = pset.getParameter<double>("ptRes_rangeMin");
  ptRes_rangeMax = pset.getParameter<double>("ptRes_rangeMax");
  ptRes_nbin = pset.getParameter<int>("ptRes_nbin");

  phiRes_rangeMin = pset.getParameter<double>("phiRes_rangeMin");
  phiRes_rangeMax = pset.getParameter<double>("phiRes_rangeMax");
  phiRes_nbin = pset.getParameter<int>("phiRes_nbin");

  cotThetaRes_rangeMin = pset.getParameter<double>("cotThetaRes_rangeMin");
  cotThetaRes_rangeMax = pset.getParameter<double>("cotThetaRes_rangeMax");
  cotThetaRes_nbin = pset.getParameter<int>("cotThetaRes_nbin");

  dxyRes_rangeMin = pset.getParameter<double>("dxyRes_rangeMin");
  dxyRes_rangeMax = pset.getParameter<double>("dxyRes_rangeMax");
  dxyRes_nbin = pset.getParameter<int>("dxyRes_nbin");

  dzRes_rangeMin = pset.getParameter<double>("dzRes_rangeMin");
  dzRes_rangeMax = pset.getParameter<double>("dzRes_rangeMax");
  dzRes_nbin = pset.getParameter<int>("dzRes_nbin");


  //--- tracking particle selectors for efficiency measurements
  using namespace edm;

  ParameterSet generalTpSelectorPSet = pset.getParameter<ParameterSet>("generalTpSelector");
  ParameterSet TpSelectorForEfficiencyVsEtaPSet = pset.getParameter<ParameterSet>("TpSelectorForEfficiencyVsEta");
  ParameterSet TpSelectorForEfficiencyVsPhiPSet = pset.getParameter<ParameterSet>("TpSelectorForEfficiencyVsPhi");
  ParameterSet TpSelectorForEfficiencyVsPtPSet = pset.getParameter<ParameterSet>("TpSelectorForEfficiencyVsPt");
  ParameterSet TpSelectorForEfficiencyVsVTXRPSet = pset.getParameter<ParameterSet>("TpSelectorForEfficiencyVsVTXR");
  ParameterSet TpSelectorForEfficiencyVsVTXZPSet = pset.getParameter<ParameterSet>("TpSelectorForEfficiencyVsVTXZ");
  
  using namespace reco::modules;
  generalTpSelector             = new TrackingParticleSelector(ParameterAdapter<TrackingParticleSelector>::make(generalTpSelectorPSet));
  TpSelectorForEfficiencyVsEta  = new TrackingParticleSelector(ParameterAdapter<TrackingParticleSelector>::make(TpSelectorForEfficiencyVsEtaPSet));
  TpSelectorForEfficiencyVsPhi  = new TrackingParticleSelector(ParameterAdapter<TrackingParticleSelector>::make(TpSelectorForEfficiencyVsPhiPSet));
  TpSelectorForEfficiencyVsPt   = new TrackingParticleSelector(ParameterAdapter<TrackingParticleSelector>::make(TpSelectorForEfficiencyVsPtPSet));
  TpSelectorForEfficiencyVsVTXR = new TrackingParticleSelector(ParameterAdapter<TrackingParticleSelector>::make(TpSelectorForEfficiencyVsVTXRPSet));
  TpSelectorForEfficiencyVsVTXZ = new TrackingParticleSelector(ParameterAdapter<TrackingParticleSelector>::make(TpSelectorForEfficiencyVsVTXZPSet));

  // fix for the LogScale by Ryan
  if(useLogPt){
    maxPt=log10(maxPt);
    if(minPt > 0){
      minPt=log10(minPt);
    }
    else{
      edm::LogWarning("MultiTrackValidator") 
	<< "minPt = "
	<< minPt << " <= 0 out of range while requesting log scale.  Using minPt = 0.1.";
      minPt=log10(0.1);
    }
  }

}

MTVHistoProducerAlgoForTracker::~MTVHistoProducerAlgoForTracker(){
  delete generalTpSelector;
  delete TpSelectorForEfficiencyVsEta;
  delete TpSelectorForEfficiencyVsPhi;
  delete TpSelectorForEfficiencyVsPt;
  delete TpSelectorForEfficiencyVsVTXR;
  delete TpSelectorForEfficiencyVsVTXZ;
}


void MTVHistoProducerAlgoForTracker::setUpVectors(){
  std::vector<double> etaintervalsv;
  std::vector<double> phiintervalsv;
  std::vector<double> pTintervalsv;
  std::vector<double> dxyintervalsv;
  std::vector<double> dzintervalsv;
  std::vector<double> vertposintervalsv;
  std::vector<double> zposintervalsv;
  std::vector<int>    totSIMveta,totASSveta,totASS2veta,totRECveta;
  std::vector<int>    totSIMvpT,totASSvpT,totASS2vpT,totRECvpT;
  std::vector<int>    totSIMv_hit,totASSv_hit,totASS2v_hit,totRECv_hit;
  std::vector<int>    totSIMv_phi,totASSv_phi,totASS2v_phi,totRECv_phi;
  std::vector<int>    totSIMv_dxy,totASSv_dxy,totASS2v_dxy,totRECv_dxy;
  std::vector<int>    totSIMv_dz,totASSv_dz,totASS2v_dz,totRECv_dz;
  std::vector<int>    totSIMv_vertpos,totASSv_vertpos,totSIMv_zpos,totASSv_zpos; 

  double step=(maxEta-minEta)/nintEta;
  //std::ostringstream title,name; ///BM, what is this?
  etaintervalsv.push_back(minEta);
  for (int k=1;k<nintEta+1;k++) {
    double d=minEta+k*step;
    etaintervalsv.push_back(d);
    totSIMveta.push_back(0);
    totASSveta.push_back(0);
    totASS2veta.push_back(0);
    totRECveta.push_back(0);
  }   
  etaintervals.push_back(etaintervalsv);
  totSIMeta.push_back(totSIMveta);
  totASSeta.push_back(totASSveta);
  totASS2eta.push_back(totASS2veta);
  totRECeta.push_back(totRECveta);
  
  double stepPt = (maxPt-minPt)/nintPt;
  pTintervalsv.push_back(minPt);
  for (int k=1;k<nintPt+1;k++) {
    double d=0;
    if(useLogPt)d=pow(10,minPt+k*stepPt);
    else d=minPt+k*stepPt;
    pTintervalsv.push_back(d);
    totSIMvpT.push_back(0);
    totASSvpT.push_back(0);
    totASS2vpT.push_back(0);
    totRECvpT.push_back(0);
  }
  pTintervals.push_back(pTintervalsv);
  totSIMpT.push_back(totSIMvpT);
  totASSpT.push_back(totASSvpT);
  totASS2pT.push_back(totASS2vpT);
  totRECpT.push_back(totRECvpT);
  
  for (int k=1;k<nintHit+1;k++) {
    totSIMv_hit.push_back(0);
    totASSv_hit.push_back(0);
    totASS2v_hit.push_back(0);
    totRECv_hit.push_back(0);
  }
  totSIM_hit.push_back(totSIMv_hit);
  totASS_hit.push_back(totASSv_hit);
  totASS2_hit.push_back(totASS2v_hit);
  totREC_hit.push_back(totRECv_hit);
  
  double stepPhi = (maxPhi-minPhi)/nintPhi;
  phiintervalsv.push_back(minPhi);
  for (int k=1;k<nintPhi+1;k++) {
    double d=minPhi+k*stepPhi;
    phiintervalsv.push_back(d);
    totSIMv_phi.push_back(0);
    totASSv_phi.push_back(0);
    totASS2v_phi.push_back(0);
    totRECv_phi.push_back(0);
  }
  phiintervals.push_back(phiintervalsv);
  totSIM_phi.push_back(totSIMv_phi);
  totASS_phi.push_back(totASSv_phi);
  totASS2_phi.push_back(totASS2v_phi);
  totREC_phi.push_back(totRECv_phi);
  
  double stepDxy = (maxDxy-minDxy)/nintDxy;
  dxyintervalsv.push_back(minDxy);
  for (int k=1;k<nintDxy+1;k++) {
    double d=minDxy+k*stepDxy;
    dxyintervalsv.push_back(d);
    totSIMv_dxy.push_back(0);
    totASSv_dxy.push_back(0);
    totASS2v_dxy.push_back(0);
    totRECv_dxy.push_back(0);
  }
  dxyintervals.push_back(dxyintervalsv);
  totSIM_dxy.push_back(totSIMv_dxy);
  totASS_dxy.push_back(totASSv_dxy);
  totASS2_dxy.push_back(totASS2v_dxy);
  totREC_dxy.push_back(totRECv_dxy);
  
  
  double stepDz = (maxDz-minDz)/nintDz;
  dzintervalsv.push_back(minDz);
  for (int k=1;k<nintDz+1;k++) {
    double d=minDz+k*stepDz;
    dzintervalsv.push_back(d);
    totSIMv_dz.push_back(0);
    totASSv_dz.push_back(0);
    totASS2v_dz.push_back(0);
    totRECv_dz.push_back(0);
  }
  dzintervals.push_back(dzintervalsv);
  totSIM_dz.push_back(totSIMv_dz);
  totASS_dz.push_back(totASSv_dz);
  totASS2_dz.push_back(totASS2v_dz);
  totREC_dz.push_back(totRECv_dz);
  
  double stepVertpos = (maxVertpos-minVertpos)/nintVertpos;
  vertposintervalsv.push_back(minVertpos);
  for (int k=1;k<nintVertpos+1;k++) {
    double d=minVertpos+k*stepVertpos;
    vertposintervalsv.push_back(d);
    totSIMv_vertpos.push_back(0);
    totASSv_vertpos.push_back(0);
  }
  vertposintervals.push_back(vertposintervalsv);
  totSIM_vertpos.push_back(totSIMv_vertpos);
  totASS_vertpos.push_back(totASSv_vertpos);
    
  double stepZpos = (maxZpos-minZpos)/nintZpos;
  zposintervalsv.push_back(minZpos);
  for (int k=1;k<nintZpos+1;k++) {
    double d=minZpos+k*stepZpos;
    zposintervalsv.push_back(d);
    totSIMv_zpos.push_back(0);
    totASSv_zpos.push_back(0);
  }
  zposintervals.push_back(zposintervalsv);
  totSIM_zpos.push_back(totSIMv_zpos);
  totASS_zpos.push_back(totASSv_zpos);
}

void MTVHistoProducerAlgoForTracker::bookSimHistos(){
  h_ptSIM.push_back( dbe_->book1D("ptSIM", "generated p_{t}", 5500, 0, 110 ) );
  h_etaSIM.push_back( dbe_->book1D("etaSIM", "generated pseudorapidity", 500, -2.5, 2.5 ) );
  h_tracksSIM.push_back( dbe_->book1D("tracksSIM","number of simulated tracks",200,-0.5,99.5) );
  h_vertposSIM.push_back( dbe_->book1D("vertposSIM","Transverse position of sim vertices",100,0.,120.) );  
}


void MTVHistoProducerAlgoForTracker::bookRecoHistos(){
  h_tracks.push_back( dbe_->book1D("tracks","number of reconstructed tracks",200,-0.5,19.5) );
  h_fakes.push_back( dbe_->book1D("fakes","number of fake reco tracks",20,-0.5,19.5) );
  h_charge.push_back( dbe_->book1D("charge","charge",3,-1.5,1.5) );
  
  h_hits.push_back( dbe_->book1D("hits", "number of hits per track", nintHit,minHit,maxHit ) );
  h_losthits.push_back( dbe_->book1D("losthits", "number of lost hits per track", nintHit,minHit,maxHit) );
  h_nchi2.push_back( dbe_->book1D("chi2", "normalized #chi^{2}", 200, 0, 20 ) );
  h_nchi2_prob.push_back( dbe_->book1D("chi2_prob", "normalized #chi^{2} probability",100,0,1));

  /// this are needed to calculate efficiency during tha harvesting for the automated validation
  h_recoeta.push_back( dbe_->book1D("num_reco_eta","N of reco track vs eta",nintEta,minEta,maxEta) );
  h_assoceta.push_back( dbe_->book1D("num_assoc(simToReco)_eta","N of associated tracks (simToReco) vs eta",nintEta,minEta,maxEta) );
  h_assoc2eta.push_back( dbe_->book1D("num_assoc(recoToSim)_eta","N of associated (recoToSim) tracks vs eta",nintEta,minEta,maxEta) );
  h_simuleta.push_back( dbe_->book1D("num_simul_eta","N of simulated tracks vs eta",nintEta,minEta,maxEta) );
  h_recopT.push_back( dbe_->book1D("num_reco_pT","N of reco track vs pT",nintPt,minPt,maxPt) );
  h_assocpT.push_back( dbe_->book1D("num_assoc(simToReco)_pT","N of associated tracks (simToReco) vs pT",nintPt,minPt,maxPt) );
  h_assoc2pT.push_back( dbe_->book1D("num_assoc(recoToSim)_pT","N of associated (recoToSim) tracks vs pT",nintPt,minPt,maxPt) );
  h_simulpT.push_back( dbe_->book1D("num_simul_pT","N of simulated tracks vs pT",nintPt,minPt,maxPt) );
  //
  h_recohit.push_back( dbe_->book1D("num_reco_hit","N of reco track vs hit",nintHit,minHit,maxHit) );
  h_assochit.push_back( dbe_->book1D("num_assoc(simToReco)_hit","N of associated tracks (simToReco) vs hit",nintHit,minHit,maxHit) );
  h_assoc2hit.push_back( dbe_->book1D("num_assoc(recoToSim)_hit","N of associated (recoToSim) tracks vs hit",nintHit,minHit,maxHit) );
  h_simulhit.push_back( dbe_->book1D("num_simul_hit","N of simulated tracks vs hit",nintHit,minHit,maxHit) );
  //
  h_recophi.push_back( dbe_->book1D("num_reco_phi","N of reco track vs phi",nintPhi,minPhi,maxPhi) );
  h_assocphi.push_back( dbe_->book1D("num_assoc(simToReco)_phi","N of associated tracks (simToReco) vs phi",nintPhi,minPhi,maxPhi) );
  h_assoc2phi.push_back( dbe_->book1D("num_assoc(recoToSim)_phi","N of associated (recoToSim) tracks vs phi",nintPhi,minPhi,maxPhi) );
  h_simulphi.push_back( dbe_->book1D("num_simul_phi","N of simulated tracks vs phi",nintPhi,minPhi,maxPhi) );
  
  h_recodxy.push_back( dbe_->book1D("num_reco_dxy","N of reco track vs dxy",nintDxy,minDxy,maxDxy) );
  h_assocdxy.push_back( dbe_->book1D("num_assoc(simToReco)_dxy","N of associated tracks (simToReco) vs dxy",nintDxy,minDxy,maxDxy) );
  h_assoc2dxy.push_back( dbe_->book1D("num_assoc(recoToSim)_dxy","N of associated (recoToSim) tracks vs dxy",nintDxy,minDxy,maxDxy) );
  h_simuldxy.push_back( dbe_->book1D("num_simul_dxy","N of simulated tracks vs dxy",nintDxy,minDxy,maxDxy) );
  
  h_recodz.push_back( dbe_->book1D("num_reco_dz","N of reco track vs dz",nintDz,minDz,maxDz) );
  h_assocdz.push_back( dbe_->book1D("num_assoc(simToReco)_dz","N of associated tracks (simToReco) vs dz",nintDz,minDz,maxDz) );
  h_assoc2dz.push_back( dbe_->book1D("num_assoc(recoToSim)_dz","N of associated (recoToSim) tracks vs dz",nintDz,minDz,maxDz) );
  h_simuldz.push_back( dbe_->book1D("num_simul_dz","N of simulated tracks vs dz",nintDz,minDz,maxDz) );
  
  h_assocvertpos.push_back( dbe_->book1D("num_assoc(simToReco)_vertpos",
					 "N of associated tracks (simToReco) vs transverse vert position",	 
					 nintVertpos,minVertpos,maxVertpos) );
  h_simulvertpos.push_back( dbe_->book1D("num_simul_vertpos","N of simulated tracks vs transverse vert position",
					 nintVertpos,minVertpos,maxVertpos) );
  
  h_assoczpos.push_back( dbe_->book1D("num_assoc(simToReco)_zpos","N of associated tracks (simToReco) vs z vert position",
				      nintZpos,minZpos,maxZpos) );
  h_simulzpos.push_back( dbe_->book1D("num_simul_zpos","N of simulated tracks vs z vert position",nintZpos,minZpos,maxZpos) );
  

  /////////////////////////////////
  
  h_eta.push_back( dbe_->book1D("eta", "pseudorapidity residue", 1000, -0.1, 0.1 ) );
  h_pt.push_back( dbe_->book1D("pullPt", "pull of p_{t}", 100, -10, 10 ) );
  h_pullTheta.push_back( dbe_->book1D("pullTheta","pull of #theta parameter",250,-25,25) );
  h_pullPhi.push_back( dbe_->book1D("pullPhi","pull of #phi parameter",250,-25,25) );
  h_pullDxy.push_back( dbe_->book1D("pullDxy","pull of dxy parameter",250,-25,25) );
  h_pullDz.push_back( dbe_->book1D("pullDz","pull of dz parameter",250,-25,25) );
  h_pullQoverp.push_back( dbe_->book1D("pullQoverp","pull of qoverp parameter",250,-25,25) );
  
  /* TO BE FIXED -----------
  if (associators[ww]=="TrackAssociatorByChi2"){
    h_assochi2.push_back( dbe_->book1D("assocChi2","track association #chi^{2}",1000000,0,100000) );
    h_assochi2_prob.push_back(dbe_->book1D("assocChi2_prob","probability of association #chi^{2}",100,0,1));
  } else if (associators[ww]=="TrackAssociatorByHits"){
    h_assocFraction.push_back( dbe_->book1D("assocFraction","fraction of shared hits",200,0,2) );
    h_assocSharedHit.push_back(dbe_->book1D("assocSharedHit","number of shared hits",20,0,20));
  }
  */
  h_assocFraction.push_back( dbe_->book1D("assocFraction","fraction of shared hits",200,0,2) );
  h_assocSharedHit.push_back(dbe_->book1D("assocSharedHit","number of shared hits",20,0,20));
  // ----------------------

  chi2_vs_nhits.push_back( dbe_->book2D("chi2_vs_nhits","#chi^{2} vs nhits",25,0,25,100,0,10) );
  
  etares_vs_eta.push_back( dbe_->book2D("etares_vs_eta","etaresidue vs eta",nintEta,minEta,maxEta,200,-0.1,0.1) );
  nrec_vs_nsim.push_back( dbe_->book2D("nrec_vs_nsim","nrec vs nsim",20,-0.5,19.5,20,-0.5,19.5) );
  
  chi2_vs_eta.push_back( dbe_->book2D("chi2_vs_eta","chi2_vs_eta",nintEta,minEta,maxEta, 200, 0, 20 ));
  chi2_vs_phi.push_back( dbe_->book2D("chi2_vs_phi","#chi^{2} vs #phi",nintPhi,minPhi,maxPhi, 200, 0, 20 ) );
  
  nhits_vs_eta.push_back( dbe_->book2D("nhits_vs_eta","nhits vs eta",nintEta,minEta,maxEta,nintHit,minHit,maxHit) );
  nPXBhits_vs_eta.push_back( dbe_->book2D("nPXBhits_vs_eta","# PXB its vs eta",nintEta,minEta,maxEta,nintHit,minHit,maxHit) );
  nPXFhits_vs_eta.push_back( dbe_->book2D("nPXFhits_vs_eta","# PXF hits vs eta",nintEta,minEta,maxEta,nintHit,minHit,maxHit) );
  nTIBhits_vs_eta.push_back( dbe_->book2D("nTIBhits_vs_eta","# TIB hits vs eta",nintEta,minEta,maxEta,nintHit,minHit,maxHit) );
  nTIDhits_vs_eta.push_back( dbe_->book2D("nTIDhits_vs_eta","# TID hits vs eta",nintEta,minEta,maxEta,nintHit,minHit,maxHit) );
  nTOBhits_vs_eta.push_back( dbe_->book2D("nTOBhits_vs_eta","# TOB hits vs eta",nintEta,minEta,maxEta,nintHit,minHit,maxHit) );
  nTEChits_vs_eta.push_back( dbe_->book2D("nTEChits_vs_eta","# TEC hits vs eta",nintEta,minEta,maxEta,nintHit,minHit,maxHit) );

  nLayersWithMeas_vs_eta.push_back( dbe_->book2D("nLayersWithMeas_vs_eta","# Layers with measurement vs eta",
						nintEta,minEta,maxEta,nintHit,minHit,maxHit) );
  nPXLlayersWithMeas_vs_eta.push_back( dbe_->book2D("nPXLlayersWithMeas_vs_eta","# PXL Layers with measurement vs eta",
						   nintEta,minEta,maxEta,nintHit,minHit,maxHit) );
  nSTRIPlayersWithMeas_vs_eta.push_back( dbe_->book2D("nSTRIPlayersWithMeas_vs_eta","# STRIP Layers with measurement vs eta",
						      nintEta,minEta,maxEta,nintHit,minHit,maxHit) );
  nSTRIPlayersWith1dMeas_vs_eta.push_back( dbe_->book2D("nSTRIPlayersWith1dMeas_vs_eta","# STRIP Layers with 1D measurement vs eta",
							nintEta,minEta,maxEta,nintHit,minHit,maxHit) );
  nSTRIPlayersWith2dMeas_vs_eta.push_back( dbe_->book2D("nSTRIPlayersWith2dMeas_vs_eta","# STRIP Layers with 2D measurement vs eta",
						       nintEta,minEta,maxEta,nintHit,minHit,maxHit) );
  
  nhits_vs_phi.push_back( dbe_->book2D("nhits_vs_phi","#hits vs #phi",nintPhi,minPhi,maxPhi,nintHit,minHit,maxHit) );
  
  nlosthits_vs_eta.push_back( dbe_->book2D("nlosthits_vs_eta","nlosthits vs eta",nintEta,minEta,maxEta,nintHit,minHit,maxHit) );
  
  //resolution of track parameters
  //                       dPt/Pt    cotTheta        Phi            TIP            LIP
  // log10(pt)<0.5        100,0.1    240,0.08     100,0.015      100,0.1000    150,0.3000
  // 0.5<log10(pt)<1.5    100,0.1    120,0.01     100,0.003      100,0.0100    150,0.0500
  // >1.5                 100,0.3    100,0.005    100,0.0008     100,0.0060    120,0.0300
  
  ptres_vs_eta.push_back(dbe_->book2D("ptres_vs_eta","ptres_vs_eta",
				      nintEta,minEta,maxEta, ptRes_nbin, ptRes_rangeMin, ptRes_rangeMax));
  
  ptres_vs_phi.push_back( dbe_->book2D("ptres_vs_phi","p_{t} res vs #phi",
				       nintPhi,minPhi,maxPhi, ptRes_nbin, ptRes_rangeMin, ptRes_rangeMax));
  
  ptres_vs_pt.push_back(dbe_->book2D("ptres_vs_pt","ptres_vs_pt",nintPt,minPt,maxPt, ptRes_nbin, ptRes_rangeMin, ptRes_rangeMax));
  
  cotThetares_vs_eta.push_back(dbe_->book2D("cotThetares_vs_eta","cotThetares_vs_eta",
					    nintEta,minEta,maxEta,cotThetaRes_nbin, cotThetaRes_rangeMin, cotThetaRes_rangeMax));

  
  cotThetares_vs_pt.push_back(dbe_->book2D("cotThetares_vs_pt","cotThetares_vs_pt",
					   nintPt,minPt,maxPt, cotThetaRes_nbin, cotThetaRes_rangeMin, cotThetaRes_rangeMax));      


  phires_vs_eta.push_back(dbe_->book2D("phires_vs_eta","phires_vs_eta",
				       nintEta,minEta,maxEta, phiRes_nbin, phiRes_rangeMin, phiRes_rangeMax));

  phires_vs_pt.push_back(dbe_->book2D("phires_vs_pt","phires_vs_pt",
				      nintPt,minPt,maxPt, phiRes_nbin, phiRes_rangeMin, phiRes_rangeMax));

  phires_vs_phi.push_back(dbe_->book2D("phires_vs_phi","#phi res vs #phi",
				       nintPhi,minPhi,maxPhi,phiRes_nbin, phiRes_rangeMin, phiRes_rangeMax));

  dxyres_vs_eta.push_back(dbe_->book2D("dxyres_vs_eta","dxyres_vs_eta",
				       nintEta,minEta,maxEta,dxyRes_nbin, dxyRes_rangeMin, dxyRes_rangeMax));
  
  dxyres_vs_pt.push_back( dbe_->book2D("dxyres_vs_pt","dxyres_vs_pt",
				       nintPt,minPt,maxPt,dxyRes_nbin, dxyRes_rangeMin, dxyRes_rangeMax));
  
  dzres_vs_eta.push_back(dbe_->book2D("dzres_vs_eta","dzres_vs_eta",
				      nintEta,minEta,maxEta,dzRes_nbin, dzRes_rangeMin, dzRes_rangeMax));

  dzres_vs_pt.push_back(dbe_->book2D("dzres_vs_pt","dzres_vs_pt",nintPt,minPt,maxPt,dzRes_nbin, dzRes_rangeMin, dzRes_rangeMax));
  
  ptmean_vs_eta_phi.push_back(dbe_->bookProfile2D("ptmean_vs_eta_phi","mean p_{t} vs #eta and #phi",
						  nintPhi,minPhi,maxPhi,nintEta,minEta,maxEta,1000,0,1000));
  phimean_vs_eta_phi.push_back(dbe_->bookProfile2D("phimean_vs_eta_phi","mean #phi vs #eta and #phi",
						   nintPhi,minPhi,maxPhi,nintEta,minEta,maxEta,nintPhi,minPhi,maxPhi));
  
  //pulls of track params vs eta: to be used with fitslicesytool
  dxypull_vs_eta.push_back(dbe_->book2D("dxypull_vs_eta","dxypull_vs_eta",nintEta,minEta,maxEta,100,-10,10));
  ptpull_vs_eta.push_back(dbe_->book2D("ptpull_vs_eta","ptpull_vs_eta",nintEta,minEta,maxEta,100,-10,10)); 
  dzpull_vs_eta.push_back(dbe_->book2D("dzpull_vs_eta","dzpull_vs_eta",nintEta,minEta,maxEta,100,-10,10)); 
  phipull_vs_eta.push_back(dbe_->book2D("phipull_vs_eta","phipull_vs_eta",nintEta,minEta,maxEta,100,-10,10)); 
  thetapull_vs_eta.push_back(dbe_->book2D("thetapull_vs_eta","thetapull_vs_eta",nintEta,minEta,maxEta,100,-10,10));
  
  //      h_ptshiftetamean.push_back( dbe_->book1D("h_ptshifteta_Mean","<#deltapT/pT>[%] vs #eta",nintEta,minEta,maxEta) ); 
  

  //pulls of track params vs phi
  ptpull_vs_phi.push_back(dbe_->book2D("ptpull_vs_phi","p_{t} pull vs #phi",nintPhi,minPhi,maxPhi,100,-10,10)); 
  phipull_vs_phi.push_back(dbe_->book2D("phipull_vs_phi","#phi pull vs #phi",nintPhi,minPhi,maxPhi,100,-10,10)); 
  thetapull_vs_phi.push_back(dbe_->book2D("thetapull_vs_phi","#theta pull vs #phi",nintPhi,minPhi,maxPhi,100,-10,10));

  
  nrecHit_vs_nsimHit_sim2rec.push_back( dbe_->book2D("nrecHit_vs_nsimHit_sim2rec","nrecHit vs nsimHit (Sim2RecAssoc)",
						     nintHit,minHit,maxHit, nintHit,minHit,maxHit ));
  nrecHit_vs_nsimHit_rec2sim.push_back( dbe_->book2D("nrecHit_vs_nsimHit_rec2sim","nrecHit vs nsimHit (Rec2simAssoc)",
						     nintHit,minHit,maxHit, nintHit,minHit,maxHit ));

  // dE/dx stuff
  // FIXME: it would be nice to have an array
  h_dedx_estim1.push_back( dbe_->book1D("h_dedx_estim1","dE/dx estimator 1",nintDeDx,minDeDx,maxDeDx) ); 
  h_dedx_estim2.push_back( dbe_->book1D("h_dedx_estim2","dE/dx estimator 2",nintDeDx,minDeDx,maxDeDx) ); 
  h_dedx_nom1.push_back( dbe_->book1D("h_dedx_nom1","dE/dx number of measurements",nintHit,minHit,maxHit) ); 
  h_dedx_nom2.push_back( dbe_->book1D("h_dedx_nom2","dE/dx number of measurements",nintHit,minHit,maxHit) ); 
  h_dedx_sat1.push_back( dbe_->book1D("h_dedx_sat1","dE/dx number of measurements with saturation",nintHit,minHit,maxHit) ); 
  h_dedx_sat2.push_back( dbe_->book1D("h_dedx_sat2","dE/dx number of measurements with saturation",nintHit,minHit,maxHit) ); 


  if(useLogPt){
    BinLogX(dzres_vs_pt.back()->getTH2F());
    BinLogX(dxyres_vs_pt.back()->getTH2F());
    BinLogX(phires_vs_pt.back()->getTH2F());
    BinLogX(cotThetares_vs_pt.back()->getTH2F());
    BinLogX(ptres_vs_pt.back()->getTH2F());
    BinLogX(h_recopT.back()->getTH1F());
    BinLogX(h_assocpT.back()->getTH1F());
    BinLogX(h_assoc2pT.back()->getTH1F());
    BinLogX(h_simulpT.back()->getTH1F());
  }  
}

void MTVHistoProducerAlgoForTracker::bookRecoHistosForStandaloneRunning(){
  h_effic.push_back( dbe_->book1D("effic","efficiency vs #eta",nintEta,minEta,maxEta) );
  h_efficPt.push_back( dbe_->book1D("efficPt","efficiency vs pT",nintPt,minPt,maxPt) );
  h_effic_vs_hit.push_back( dbe_->book1D("effic_vs_hit","effic vs hit",nintHit,minHit,maxHit) );
  h_effic_vs_phi.push_back( dbe_->book1D("effic_vs_phi","effic vs phi",nintPhi,minPhi,maxPhi) );
  h_effic_vs_dxy.push_back( dbe_->book1D("effic_vs_dxy","effic vs dxy",nintDxy,minDxy,maxDxy) );
  h_effic_vs_dz.push_back( dbe_->book1D("effic_vs_dz","effic vs dz",nintDz,minDz,maxDz) );
  h_effic_vs_vertpos.push_back( dbe_->book1D("effic_vs_vertpos","effic vs vertpos",nintVertpos,minVertpos,maxVertpos) );
  h_effic_vs_zpos.push_back( dbe_->book1D("effic_vs_zpos","effic vs zpos",nintZpos,minZpos,maxZpos) );

  h_fakerate.push_back( dbe_->book1D("fakerate","fake rate vs #eta",nintEta,minEta,maxEta) );
  h_fakeratePt.push_back( dbe_->book1D("fakeratePt","fake rate vs pT",nintPt,minPt,maxPt) );
  h_fake_vs_hit.push_back( dbe_->book1D("fakerate_vs_hit","fake rate vs hit",nintHit,minHit,maxHit) );
  h_fake_vs_phi.push_back( dbe_->book1D("fakerate_vs_phi","fake vs phi",nintPhi,minPhi,maxPhi) );
  h_fake_vs_dxy.push_back( dbe_->book1D("fakerate_vs_dxy","fake rate vs dxy",nintDxy,minDxy,maxDxy) );
  h_fake_vs_dz.push_back( dbe_->book1D("fakerate_vs_dz","fake vs dz",nintDz,minDz,maxDz) );

  h_chi2meanhitsh.push_back( dbe_->bookProfile("chi2mean_vs_nhits","mean #chi^{2} vs nhits",25,0,25,100,0,10) );
  h_chi2meanh.push_back( dbe_->bookProfile("chi2mean","mean #chi^{2} vs #eta",nintEta,minEta,maxEta, 200, 0, 20) );
  h_chi2mean_vs_phi.push_back( dbe_->bookProfile("chi2mean_vs_phi","mean of #chi^{2} vs #phi",nintPhi,minPhi,maxPhi, 200, 0, 20) );

  h_hits_eta.push_back( dbe_->bookProfile("hits_eta","mean #hits vs eta",nintEta,minEta,maxEta,nintHit,minHit,maxHit) );
  h_PXBhits_eta.push_back( dbe_->bookProfile("PXBhits_eta","mean # PXB hits vs eta",nintEta,minEta,maxEta,nintHit,minHit,maxHit) );
  h_PXFhits_eta.push_back( dbe_->bookProfile("PXFhits_eta","mean # PXF hits vs eta",nintEta,minEta,maxEta,nintHit,minHit,maxHit) );
  h_TIBhits_eta.push_back( dbe_->bookProfile("TIBhits_eta","mean # TIB hits vs eta",nintEta,minEta,maxEta,nintHit,minHit,maxHit) );
  h_TIDhits_eta.push_back( dbe_->bookProfile("TIDhits_eta","mean # TID hits vs eta",nintEta,minEta,maxEta,nintHit,minHit,maxHit) );
  h_TOBhits_eta.push_back( dbe_->bookProfile("TOBhits_eta","mean # TOB hits vs eta",nintEta,minEta,maxEta,nintHit,minHit,maxHit) );
  h_TEChits_eta.push_back( dbe_->bookProfile("TEChits_eta","mean # TEC hits vs eta",nintEta,minEta,maxEta,nintHit,minHit,maxHit) );

  h_LayersWithMeas_eta.push_back(dbe_->bookProfile("LayersWithMeas_eta","mean # LayersWithMeas vs eta",
                           nintEta,minEta,maxEta,nintHit,minHit,maxHit) );
  h_PXLlayersWithMeas_eta.push_back(dbe_->bookProfile("PXLlayersWith2dMeas_eta","mean # PXLlayersWithMeas vs eta",
                              nintEta,minEta,maxEta,nintHit,minHit,maxHit) );
  h_STRIPlayersWithMeas_eta.push_back(dbe_->bookProfile("STRIPlayersWithMeas_eta","mean # STRIPlayersWithMeas vs eta",
                            nintEta,minEta,maxEta,nintHit,minHit,maxHit) );
  h_STRIPlayersWith1dMeas_eta.push_back(dbe_->bookProfile("STRIPlayersWith1dMeas_eta","mean # STRIPlayersWith1dMeas vs eta",
                              nintEta,minEta,maxEta,nintHit,minHit,maxHit) );
  h_STRIPlayersWith2dMeas_eta.push_back(dbe_->bookProfile("STRIPlayersWith2dMeas_eta","mean # STRIPlayersWith2dMeas vs eta",
                              nintEta,minEta,maxEta,nintHit,minHit,maxHit) );
  h_hits_phi.push_back( dbe_->bookProfile("hits_phi","mean #hits vs #phi",nintPhi,minPhi,maxPhi, nintHit,minHit,maxHit) );
  h_losthits_eta.push_back( dbe_->bookProfile("losthits_eta","losthits_eta",nintEta,minEta,maxEta,nintHit,minHit,maxHit) );

  h_ptrmsh.push_back( dbe_->book1D("ptres_vs_eta_Sigma","#sigma(#deltap_{t}/p_{t}) vs #eta",nintEta,minEta,maxEta) );
  h_ptmeanhPhi.push_back( dbe_->book1D("ptres_vs_phi_Mean","mean of p_{t} resolution vs #phi",nintPhi,minPhi,maxPhi));
  h_ptrmshPhi.push_back( dbe_->book1D("ptres_vs_phi_Sigma","#sigma(#deltap_{t}/p_{t}) vs #phi",nintPhi,minPhi,maxPhi) );
  h_ptmeanhPt.push_back( dbe_->book1D("ptres_vs_pt_Mean","mean of p_{t} resolution vs p_{t}",nintPt,minPt,maxPt));
  h_ptrmshPt.push_back( dbe_->book1D("ptres_vs_pt_Sigma","#sigma(#deltap_{t}/p_{t}) vs pT",nintPt,minPt,maxPt) );
  h_cotThetameanh.push_back( dbe_->book1D("cotThetares_vs_eta_Mean","#sigma(cot(#theta)) vs #eta Mean",nintEta,minEta,maxEta) );
  h_cotThetarmsh.push_back( dbe_->book1D("cotThetares_vs_eta_Sigma","#sigma(cot(#theta)) vs #eta Sigma",nintEta,minEta,maxEta) );
  h_cotThetameanhPt.push_back( dbe_->book1D("cotThetares_vs_pt_Mean","#sigma(cot(#theta)) vs pT Mean",nintPt,minPt,maxPt) );
  h_cotThetarmshPt.push_back( dbe_->book1D("cotThetares_vs_pt_Sigma","#sigma(cot(#theta)) vs pT Sigma",nintPt,minPt,maxPt) );
  h_phimeanh.push_back(dbe_->book1D("phires_vs_eta_Mean","mean of #phi res vs #eta",nintEta,minEta,maxEta));
  h_phirmsh.push_back( dbe_->book1D("phires_vs_eta_Sigma","#sigma(#delta#phi) vs #eta",nintEta,minEta,maxEta) );
  h_phimeanhPt.push_back(dbe_->book1D("phires_vs_pt_Mean","mean of #phi res vs pT",nintPt,minPt,maxPt));
  h_phirmshPt.push_back( dbe_->book1D("phires_vs_pt_Sigma","#sigma(#delta#phi) vs pT",nintPt,minPt,maxPt) );
  h_phimeanhPhi.push_back(dbe_->book1D("phires_vs_phi_Mean","mean of #phi res vs #phi",nintPhi,minPhi,maxPhi));
  h_phirmshPhi.push_back( dbe_->book1D("phires_vs_phi_Sigma","#sigma(#delta#phi) vs #phi",nintPhi,minPhi,maxPhi) );
  h_dxymeanh.push_back( dbe_->book1D("dxyres_vs_eta_Mean","mean of dxyres vs #eta",nintEta,minEta,maxEta) );
  h_dxyrmsh.push_back( dbe_->book1D("dxyres_vs_eta_Sigma","#sigma(#deltadxy) vs #eta",nintEta,minEta,maxEta) );
  h_dxymeanhPt.push_back( dbe_->book1D("dxyres_vs_pt_Mean","mean of dxyres vs pT",nintPt,minPt,maxPt) );
  h_dxyrmshPt.push_back( dbe_->book1D("dxyres_vs_pt_Sigma","#sigmadxy vs pT",nintPt,minPt,maxPt) );
  h_dzmeanh.push_back( dbe_->book1D("dzres_vs_eta_Mean","mean of dzres vs #eta",nintEta,minEta,maxEta) );
  h_dzrmsh.push_back( dbe_->book1D("dzres_vs_eta_Sigma","#sigma(#deltadz) vs #eta",nintEta,minEta,maxEta) );
  h_dzmeanhPt.push_back( dbe_->book1D("dzres_vs_pt_Mean","mean of dzres vs pT",nintPt,minPt,maxPt) );
  h_dzrmshPt.push_back( dbe_->book1D("dzres_vs_pt_Sigma","#sigma(#deltadz vs pT",nintPt,minPt,maxPt) );
  h_dxypulletamean.push_back( dbe_->book1D("h_dxypulleta_Mean","mean of dxy pull vs #eta",nintEta,minEta,maxEta) ); 
  h_ptpulletamean.push_back( dbe_->book1D("h_ptpulleta_Mean","mean of p_{t} pull vs #eta",nintEta,minEta,maxEta) ); 
  h_dzpulletamean.push_back( dbe_->book1D("h_dzpulleta_Mean","mean of dz pull vs #eta",nintEta,minEta,maxEta) ); 
  h_phipulletamean.push_back( dbe_->book1D("h_phipulleta_Mean","mean of #phi pull vs #eta",nintEta,minEta,maxEta) ); 
  h_thetapulletamean.push_back( dbe_->book1D("h_thetapulleta_Mean","mean of #theta pull vs #eta",nintEta,minEta,maxEta) );
  h_dxypulleta.push_back( dbe_->book1D("h_dxypulleta_Sigma","#sigma of dxy pull vs #eta",nintEta,minEta,maxEta) ); 
  h_ptpulleta.push_back( dbe_->book1D("h_ptpulleta_Sigma","#sigma of p_{t} pull vs #eta",nintEta,minEta,maxEta) ); 
  h_dzpulleta.push_back( dbe_->book1D("h_dzpulleta_Sigma","#sigma of dz pull vs #eta",nintEta,minEta,maxEta) ); 
  h_phipulleta.push_back( dbe_->book1D("h_phipulleta_Sigma","#sigma of #phi pull vs #eta",nintEta,minEta,maxEta) ); 
  h_thetapulleta.push_back( dbe_->book1D("h_thetapulleta_Sigma","#sigma of #theta pull vs #eta",nintEta,minEta,maxEta) );
  h_ptshifteta.push_back( dbe_->book1D("ptres_vs_eta_Mean","<#deltapT/pT>[%] vs #eta",nintEta,minEta,maxEta) ); 
  h_ptpullphimean.push_back( dbe_->book1D("h_ptpullphi_Mean","mean of p_{t} pull vs #phi",nintPhi,minPhi,maxPhi) ); 
  h_phipullphimean.push_back( dbe_->book1D("h_phipullphi_Mean","mean of #phi pull vs #phi",nintPhi,minPhi,maxPhi) );
  h_thetapullphimean.push_back( dbe_->book1D("h_thetapullphi_Mean","mean of #theta pull vs #phi",nintPhi,minPhi,maxPhi) );
  h_ptpullphi.push_back( dbe_->book1D("h_ptpullphi_Sigma","#sigma of p_{t} pull vs #phi",nintPhi,minPhi,maxPhi) ); 
  h_phipullphi.push_back( dbe_->book1D("h_phipullphi_Sigma","#sigma of #phi pull vs #phi",nintPhi,minPhi,maxPhi) );
  h_thetapullphi.push_back( dbe_->book1D("h_thetapullphi_Sigma","#sigma of #theta pull vs #phi",nintPhi,minPhi,maxPhi) );
  
  if(useLogPt){
    BinLogX(h_dzmeanhPt.back()->getTH1F());
    BinLogX(h_dzrmshPt.back()->getTH1F());
    BinLogX(h_dxymeanhPt.back()->getTH1F());
    BinLogX(h_dxyrmshPt.back()->getTH1F());
    BinLogX(h_phimeanhPt.back()->getTH1F());
    BinLogX(h_phirmshPt.back()->getTH1F());
    BinLogX(h_cotThetameanhPt.back()->getTH1F());
    BinLogX(h_cotThetarmshPt.back()->getTH1F());
    BinLogX(h_ptmeanhPt.back()->getTH1F());
    BinLogX(h_ptrmshPt.back()->getTH1F());
    BinLogX(h_efficPt.back()->getTH1F());
    BinLogX(h_fakeratePt.back()->getTH1F());
  }    
}

void MTVHistoProducerAlgoForTracker::fill_generic_simTrack_histos(int count,
								  ParticleBase::Vector momentumTP,
								  ParticleBase::Point vertexTP){
  h_ptSIM[count]->Fill(sqrt(momentumTP.perp2()));
  h_etaSIM[count]->Fill(momentumTP.eta());
  h_vertposSIM[count]->Fill(sqrt(vertexTP.perp2()));
}



// TO BE FIXED USING PLAIN HISTOGRAMS INSTEAD OF RE-IMPLEMENTATION OF HISTOGRAMS (i.d. vectors<int/double>)
void MTVHistoProducerAlgoForTracker::fill_recoAssociated_simTrack_histos(int count,
									 const TrackingParticle& tp,
									 ParticleBase::Vector momentumTP,
									 ParticleBase::Point vertexTP,
									 double dxySim, double dzSim, int nSimHits,
									 const reco::Track* track){
  bool isMatched = track;

  if((*TpSelectorForEfficiencyVsEta)(tp)){
    //effic vs hits
    int nSimHitsInBounds = std::min((int)nSimHits,int(maxHit-1));
    totSIM_hit[count][nSimHitsInBounds]++;
    if(isMatched) {
      totASS_hit[count][nSimHitsInBounds]++;
      nrecHit_vs_nsimHit_sim2rec[count]->Fill( track->numberOfValidHits(),nSimHits);
    }

    //effic vs eta
    for (unsigned int f=0; f<etaintervals[count].size()-1; f++){
      if (getEta(momentumTP.eta())>etaintervals[count][f]&&
	  getEta(momentumTP.eta())<etaintervals[count][f+1]) {
	totSIMeta[count][f]++;
	if (isMatched) {
	  totASSeta[count][f]++;
	}
      }
    } // END for (unsigned int f=0; f<etaintervals[w].size()-1; f++){
  }

  if((*TpSelectorForEfficiencyVsPhi)(tp)){
    for (unsigned int f=0; f<phiintervals[count].size()-1; f++){
      if (momentumTP.phi() > phiintervals[count][f]&&
	  momentumTP.phi() <phiintervals[count][f+1]) {
	totSIM_phi[count][f]++;
	if (isMatched) {
	  totASS_phi[count][f]++;
	}
      }
    } // END for (unsigned int f=0; f<phiintervals[count].size()-1; f++){
  }
	
  if((*TpSelectorForEfficiencyVsPt)(tp)){
    for (unsigned int f=0; f<pTintervals[count].size()-1; f++){
      if (getPt(sqrt(momentumTP.perp2()))>pTintervals[count][f]&&
	  getPt(sqrt(momentumTP.perp2()))<pTintervals[count][f+1]) {
	totSIMpT[count][f]++;
	if (isMatched) {
	  totASSpT[count][f]++;
	}
      }
    } // END for (unsigned int f=0; f<pTintervals[count].size()-1; f++){
  }	

  if((*TpSelectorForEfficiencyVsVTXR)(tp)){
    for (unsigned int f=0; f<dxyintervals[count].size()-1; f++){
      if (dxySim>dxyintervals[count][f]&&
	  dxySim<dxyintervals[count][f+1]) {
	totSIM_dxy[count][f]++;
	if (isMatched) {
	  totASS_dxy[count][f]++;
	}
      }
    } // END for (unsigned int f=0; f<dxyintervals[count].size()-1; f++){

    for (unsigned int f=0; f<vertposintervals[count].size()-1; f++){
      if (sqrt(vertexTP.perp2())>vertposintervals[count][f]&&
	  sqrt(vertexTP.perp2())<vertposintervals[count][f+1]) {
	totSIM_vertpos[count][f]++;
	if (isMatched) {
	  totASS_vertpos[count][f]++;
	}
      }
    } // END for (unsigned int f=0; f<vertposintervals[count].size()-1; f++){
  }

  if((*TpSelectorForEfficiencyVsVTXZ)(tp)){
    for (unsigned int f=0; f<dzintervals[count].size()-1; f++){
      if (dzSim>dzintervals[count][f]&&
	  dzSim<dzintervals[count][f+1]) {
	totSIM_dz[count][f]++;
	if (isMatched) {
	  totASS_dz[count][f]++;
	}
      }
    } // END for (unsigned int f=0; f<dzintervals[count].size()-1; f++){

  
    for (unsigned int f=0; f<zposintervals[count].size()-1; f++){
      if (vertexTP.z()>zposintervals[count][f]&&
	  vertexTP.z()<zposintervals[count][f+1]) {
	totSIM_zpos[count][f]++;
	if (isMatched) {
	  totASS_zpos[count][f]++;
	}
      }
    } // END for (unsigned int f=0; f<zposintervals[count].size()-1; f++){
  }

}

// dE/dx
void MTVHistoProducerAlgoForTracker::fill_dedx_recoTrack_histos(int count, edm::RefToBase<reco::Track>& trackref, std::vector< edm::ValueMap<reco::DeDxData> > v_dEdx) {
//void MTVHistoProducerAlgoForTracker::fill_dedx_recoTrack_histos(reco::TrackRef trackref, std::vector< edm::ValueMap<reco::DeDxData> > v_dEdx) {
  double dedx;
  int nom;
  int sat;
  edm::ValueMap<reco::DeDxData> dEdxTrack;
  for (unsigned int i=0; i<v_dEdx.size(); i++) {
    dEdxTrack = v_dEdx.at(i);
    dedx = dEdxTrack[trackref].dEdx(); 
    nom  = dEdxTrack[trackref].numberOfMeasurements();
    sat  = dEdxTrack[trackref].numberOfSaturatedMeasurements();
    if (i==0) {
      h_dedx_estim1[count]->Fill(dedx);
      h_dedx_nom1[count]->Fill(nom);
      h_dedx_sat1[count]->Fill(sat);
    } else if (i==1) {
      h_dedx_estim2[count]->Fill(dedx);
      h_dedx_nom2[count]->Fill(nom);
      h_dedx_sat2[count]->Fill(sat);
    }
  }
}


// TO BE FIXED USING PLAIN HISTOGRAMS INSTEAD OF RE-IMPLEMENTATION OF HISTOGRAMS (i.d. vectors<int/double>)
void MTVHistoProducerAlgoForTracker::fill_generic_recoTrack_histos(int count,
								   const reco::Track& track,
								   math::XYZPoint bsPosition,
								   bool isMatched){

  //Compute fake rate vs eta
  for (unsigned int f=0; f<etaintervals[count].size()-1; f++){
    if (getEta(track.momentum().eta())>etaintervals[count][f]&&
	getEta(track.momentum().eta())<etaintervals[count][f+1]) {
      totRECeta[count][f]++; 
      if (isMatched) {
	totASS2eta[count][f]++;
      }
    }
  } // End for (unsigned int f=0; f<etaintervals[count].size()-1; f++){

  for (unsigned int f=0; f<phiintervals[count].size()-1; f++){
    if (track.momentum().phi()>phiintervals[count][f]&&
	track.momentum().phi()<phiintervals[count][f+1]) {
      totREC_phi[count][f]++; 
      if (isMatched) {
	totASS2_phi[count][f]++;
      }		
    }
  } // End for (unsigned int f=0; f<phiintervals[count].size()-1; f++){

	
  for (unsigned int f=0; f<pTintervals[count].size()-1; f++){
    if (getPt(sqrt(track.momentum().perp2()))>pTintervals[count][f]&&
	getPt(sqrt(track.momentum().perp2()))<pTintervals[count][f+1]) {
      totRECpT[count][f]++; 
      if (isMatched) {
	totASS2pT[count][f]++;
      }	      
    }
  } // End for (unsigned int f=0; f<pTintervals[count].size()-1; f++){
  
  for (unsigned int f=0; f<dxyintervals[count].size()-1; f++){
    if (track.dxy(bsPosition)>dxyintervals[count][f]&&
	track.dxy(bsPosition)<dxyintervals[count][f+1]) {
      totREC_dxy[count][f]++; 
      if (isMatched) {
	totASS2_dxy[count][f]++;
      }	      
    }
  } // End for (unsigned int f=0; f<dxyintervals[count].size()-1; f++){
  
  for (unsigned int f=0; f<dzintervals[count].size()-1; f++){
    if (track.dz(bsPosition)>dzintervals[count][f]&&
	track.dz(bsPosition)<dzintervals[count][f+1]) {
      totREC_dz[count][f]++; 
      if (isMatched) {
	totASS2_dz[count][f]++;
      }	      
    }
  } // End for (unsigned int f=0; f<dzintervals[count].size()-1; f++){

  int tmp = std::min((int)track.found(),int(maxHit-1));
  totREC_hit[count][tmp]++;
  if (isMatched) totASS2_hit[count][tmp]++;

}


void MTVHistoProducerAlgoForTracker::fill_simAssociated_recoTrack_histos(int count,
									 const reco::Track& track){
    //nchi2 and hits global distributions
    h_nchi2[count]->Fill(track.normalizedChi2());
    h_nchi2_prob[count]->Fill(TMath::Prob(track.chi2(),(int)track.ndof()));
    h_hits[count]->Fill(track.numberOfValidHits());
    h_losthits[count]->Fill(track.numberOfLostHits());
    chi2_vs_nhits[count]->Fill(track.numberOfValidHits(),track.normalizedChi2());
    h_charge[count]->Fill( track.charge() );
    
    //chi2 and #hit vs eta: fill 2D histos
    chi2_vs_eta[count]->Fill(getEta(track.eta()),track.normalizedChi2());
    nhits_vs_eta[count]->Fill(getEta(track.eta()),track.numberOfValidHits());
    nPXBhits_vs_eta[count]->Fill(getEta(track.eta()),track.hitPattern().numberOfValidPixelBarrelHits());
    nPXFhits_vs_eta[count]->Fill(getEta(track.eta()),track.hitPattern().numberOfValidPixelEndcapHits());
    nTIBhits_vs_eta[count]->Fill(getEta(track.eta()),track.hitPattern().numberOfValidStripTIBHits());
    nTIDhits_vs_eta[count]->Fill(getEta(track.eta()),track.hitPattern().numberOfValidStripTIDHits());
    nTOBhits_vs_eta[count]->Fill(getEta(track.eta()),track.hitPattern().numberOfValidStripTOBHits());
    nTEChits_vs_eta[count]->Fill(getEta(track.eta()),track.hitPattern().numberOfValidStripTECHits());
    nLayersWithMeas_vs_eta[count]->Fill(getEta(track.eta()),track.hitPattern().trackerLayersWithMeasurement());
    nPXLlayersWithMeas_vs_eta[count]->Fill(getEta(track.eta()),track.hitPattern().pixelLayersWithMeasurement());
    int LayersAll = track.hitPattern().stripLayersWithMeasurement();
    int Layers2D = track.hitPattern().numberOfValidStripLayersWithMonoAndStereo(); 
    int Layers1D = LayersAll - Layers2D;	
    nSTRIPlayersWithMeas_vs_eta[count]->Fill(getEta(track.eta()),LayersAll);
    nSTRIPlayersWith1dMeas_vs_eta[count]->Fill(getEta(track.eta()),Layers1D);
    nSTRIPlayersWith2dMeas_vs_eta[count]->Fill(getEta(track.eta()),Layers2D);
	
    nlosthits_vs_eta[count]->Fill(getEta(track.eta()),track.numberOfLostHits());
}



void MTVHistoProducerAlgoForTracker::fill_ResoAndPull_recoTrack_histos(int count,
								       ParticleBase::Vector momentumTP,
								       ParticleBase::Point vertexTP,
								       int chargeTP,
								       const reco::Track& track,
								       math::XYZPoint bsPosition){

  // evaluation of TP parameters
  double qoverpSim = chargeTP/sqrt(momentumTP.x()*momentumTP.x()+momentumTP.y()*momentumTP.y()+momentumTP.z()*momentumTP.z());
  double lambdaSim = M_PI/2-momentumTP.theta();
  double phiSim    = momentumTP.phi();
  double dxySim    = (-vertexTP.x()*sin(momentumTP.phi())+vertexTP.y()*cos(momentumTP.phi()));
  double dzSim     = vertexTP.z() - (vertexTP.x()*momentumTP.x()+vertexTP.y()*momentumTP.y())/sqrt(momentumTP.perp2()) 
    * momentumTP.z()/sqrt(momentumTP.perp2());

	  
  reco::Track::ParameterVector rParameters = track.parameters();
  
  double qoverpRec(0);
  double qoverpErrorRec(0); 
  double ptRec(0);
  double ptErrorRec(0);
  double lambdaRec(0); 
  double lambdaErrorRec(0);
  double phiRec(0);
  double phiErrorRec(0);

  /* TO BE FIXED LATER  -----------
  //loop to decide whether to take gsfTrack (utilisation of mode-function) or common track
  const GsfTrack* gsfTrack(0);
  if(useGsf){
    gsfTrack = dynamic_cast<const GsfTrack*>(&(*track));
    if (gsfTrack==0) edm::LogInfo("TrackValidator") << "Trying to access mode for a non-GsfTrack";
  }
  
  if (gsfTrack) {
    // get values from mode
    getRecoMomentum(*gsfTrack, ptRec, ptErrorRec, qoverpRec, qoverpErrorRec, 
		    lambdaRec,lambdaErrorRec, phiRec, phiErrorRec); 
  }
	 
  else {
    // get values from track (without mode) 
    getRecoMomentum(*track, ptRec, ptErrorRec, qoverpRec, qoverpErrorRec, 
		    lambdaRec,lambdaErrorRec, phiRec, phiErrorRec); 
  }
  */
  getRecoMomentum(track, ptRec, ptErrorRec, qoverpRec, qoverpErrorRec, 
		  lambdaRec,lambdaErrorRec, phiRec, phiErrorRec); 
  // -------------

  double ptError = ptErrorRec;	
  double ptres=ptRec-sqrt(momentumTP.perp2()); 
  double etares=track.eta()-momentumTP.Eta();


  double dxyRec    = track.dxy(bsPosition);
  double dzRec     = track.dz(bsPosition);

  // eta residue; pt, k, theta, phi, dxy, dz pulls
  double qoverpPull=(qoverpRec-qoverpSim)/qoverpErrorRec;
  double thetaPull=(lambdaRec-lambdaSim)/lambdaErrorRec;
  double phiPull=(phiRec-phiSim)/phiErrorRec;
  double dxyPull=(dxyRec-dxySim)/track.dxyError();
  double dzPull=(dzRec-dzSim)/track.dzError();

  double contrib_Qoverp = ((qoverpRec-qoverpSim)/qoverpErrorRec)*
    ((qoverpRec-qoverpSim)/qoverpErrorRec)/5;
  double contrib_dxy = ((dxyRec-dxySim)/track.dxyError())*((dxyRec-dxySim)/track.dxyError())/5;
  double contrib_dz = ((dzRec-dzSim)/track.dzError())*((dzRec-dzSim)/track.dzError())/5;
  double contrib_theta = ((lambdaRec-lambdaSim)/lambdaErrorRec)*
    ((lambdaRec-lambdaSim)/lambdaErrorRec)/5;
  double contrib_phi = ((phiRec-phiSim)/phiErrorRec)*
    ((phiRec-phiSim)/phiErrorRec)/5;

  LogTrace("TrackValidatorTEST") 
    //<< "assocChi2=" << tp.begin()->second << "\n"
    << "" <<  "\n"
    << "ptREC=" << ptRec << "\n" << "etaREC=" << track.eta() << "\n" << "qoverpREC=" << qoverpRec << "\n"
    << "dxyREC=" << dxyRec << "\n" << "dzREC=" << dzRec << "\n"
    << "thetaREC=" << track.theta() << "\n" << "phiREC=" << phiRec << "\n"
    << "" <<  "\n"
    << "qoverpError()=" << qoverpErrorRec << "\n" << "dxyError()=" << track.dxyError() << "\n"<< "dzError()=" 
    << track.dzError() << "\n"
    << "thetaError()=" << lambdaErrorRec << "\n" << "phiError()=" << phiErrorRec << "\n"
    << "" <<  "\n"
    << "ptSIM=" << sqrt(momentumTP.perp2()) << "\n"<< "etaSIM=" << momentumTP.Eta() << "\n"<< "qoverpSIM=" << qoverpSim << "\n"
    << "dxySIM=" << dxySim << "\n"<< "dzSIM=" << dzSim << "\n" << "thetaSIM=" << M_PI/2-lambdaSim << "\n" 
    << "phiSIM=" << phiSim << "\n"
    << "" << "\n"
    << "contrib_Qoverp=" << contrib_Qoverp << "\n"<< "contrib_dxy=" << contrib_dxy << "\n"<< "contrib_dz=" << contrib_dz << "\n"
    << "contrib_theta=" << contrib_theta << "\n"<< "contrib_phi=" << contrib_phi << "\n"
    << "" << "\n"
    <<"chi2PULL="<<contrib_Qoverp+contrib_dxy+contrib_dz+contrib_theta+contrib_phi<<"\n";

  h_pullQoverp[count]->Fill(qoverpPull);
  h_pullTheta[count]->Fill(thetaPull);
  h_pullPhi[count]->Fill(phiPull);
  h_pullDxy[count]->Fill(dxyPull);
  h_pullDz[count]->Fill(dzPull);


  h_pt[count]->Fill(ptres/ptError);
  h_eta[count]->Fill(etares);
  etares_vs_eta[count]->Fill(getEta(track.eta()),etares);
  

	
  //resolution of track params: fill 2D histos
  dxyres_vs_eta[count]->Fill(getEta(track.eta()),dxyRec-dxySim);
  ptres_vs_eta[count]->Fill(getEta(track.eta()),(ptRec-sqrt(momentumTP.perp2()))/ptRec);
  dzres_vs_eta[count]->Fill(getEta(track.eta()),dzRec-dzSim);
  phires_vs_eta[count]->Fill(getEta(track.eta()),phiRec-phiSim);
  cotThetares_vs_eta[count]->Fill(getEta(track.eta()),1/tan(M_PI*0.5-lambdaRec)-1/tan(M_PI*0.5-lambdaSim));         
  
  //same as before but vs pT
  dxyres_vs_pt[count]->Fill(getPt(ptRec),dxyRec-dxySim);
  ptres_vs_pt[count]->Fill(getPt(ptRec),(ptRec-sqrt(momentumTP.perp2()))/ptRec);
  dzres_vs_pt[count]->Fill(getPt(ptRec),dzRec-dzSim);
  phires_vs_pt[count]->Fill(getPt(ptRec),phiRec-phiSim);
  cotThetares_vs_pt[count]->Fill(getPt(ptRec),1/tan(M_PI*0.5-lambdaRec)-1/tan(M_PI*0.5-lambdaSim));  	 
  	
  //pulls of track params vs eta: fill 2D histos
  dxypull_vs_eta[count]->Fill(getEta(track.eta()),dxyPull);
  ptpull_vs_eta[count]->Fill(getEta(track.eta()),ptres/ptError);
  dzpull_vs_eta[count]->Fill(getEta(track.eta()),dzPull);
  phipull_vs_eta[count]->Fill(getEta(track.eta()),phiPull);
  thetapull_vs_eta[count]->Fill(getEta(track.eta()),thetaPull);
	
  //plots vs phi
  nhits_vs_phi[count]->Fill(phiRec,track.numberOfValidHits());
  chi2_vs_phi[count]->Fill(phiRec,track.normalizedChi2());
  ptmean_vs_eta_phi[count]->Fill(phiRec,getEta(track.eta()),ptRec);
  phimean_vs_eta_phi[count]->Fill(phiRec,getEta(track.eta()),phiRec);
  ptres_vs_phi[count]->Fill(phiRec,(ptRec-sqrt(momentumTP.perp2()))/ptRec);
  phires_vs_phi[count]->Fill(phiRec,phiRec-phiSim); 
  ptpull_vs_phi[count]->Fill(phiRec,ptres/ptError);
  phipull_vs_phi[count]->Fill(phiRec,phiPull); 
  thetapull_vs_phi[count]->Fill(phiRec,thetaPull); 


}



void 
MTVHistoProducerAlgoForTracker::getRecoMomentum (const reco::Track& track, double& pt, double& ptError,
						 double& qoverp, double& qoverpError, double& lambda,double& lambdaError,
						 double& phi, double& phiError ) const {
  pt = track.pt();
  ptError = track.ptError();
  qoverp = track.qoverp();
  qoverpError = track.qoverpError();
  lambda = track.lambda();
  lambdaError = track.lambdaError(); 
  phi = track.phi(); 
  phiError = track.phiError();
  //   cout <<"test1" << endl; 
  


}

void 
MTVHistoProducerAlgoForTracker::getRecoMomentum (const reco::GsfTrack& gsfTrack, double& pt, double& ptError,
						 double& qoverp, double& qoverpError, double& lambda,double& lambdaError,
						 double& phi, double& phiError  ) const {

  pt = gsfTrack.ptMode();
  ptError = gsfTrack.ptModeError();
  qoverp = gsfTrack.qoverpMode();
  qoverpError = gsfTrack.qoverpModeError();
  lambda = gsfTrack.lambdaMode();
  lambdaError = gsfTrack.lambdaModeError(); 
  phi = gsfTrack.phiMode(); 
  phiError = gsfTrack.phiModeError();
  //   cout <<"test2" << endl;

}

double 
MTVHistoProducerAlgoForTracker::getEta(double eta) {
  if (useFabsEta) return fabs(eta);
  else return eta;
}
  
double 
MTVHistoProducerAlgoForTracker::getPt(double pt) {
  if (useInvPt && pt!=0) return 1/pt;
  else return pt;
}


void MTVHistoProducerAlgoForTracker::finalHistoFits(int counter){
  //resolution of track params: get sigma from 2D histos
  FitSlicesYTool fsyt_dxy(dxyres_vs_eta[counter]);
  fsyt_dxy.getFittedSigmaWithError(h_dxyrmsh[counter]);
  fsyt_dxy.getFittedMeanWithError(h_dxymeanh[counter]);
  FitSlicesYTool fsyt_dxyPt(dxyres_vs_pt[counter]);
  fsyt_dxyPt.getFittedSigmaWithError(h_dxyrmshPt[counter]);
  fsyt_dxyPt.getFittedMeanWithError(h_dxymeanhPt[counter]);
  FitSlicesYTool fsyt_pt(ptres_vs_eta[counter]);
  fsyt_pt.getFittedSigmaWithError(h_ptrmsh[counter]);
  fsyt_pt.getFittedMeanWithError(h_ptshifteta[counter]);      
  FitSlicesYTool fsyt_ptPt(ptres_vs_pt[counter]);
  fsyt_ptPt.getFittedSigmaWithError(h_ptrmshPt[counter]);
  fsyt_ptPt.getFittedMeanWithError(h_ptmeanhPt[counter]);
  FitSlicesYTool fsyt_ptPhi(ptres_vs_phi[counter]); 
  fsyt_ptPhi.getFittedSigmaWithError(h_ptrmshPhi[counter]);
  fsyt_ptPhi.getFittedMeanWithError(h_ptmeanhPhi[counter]);
  FitSlicesYTool fsyt_dz(dzres_vs_eta[counter]);
  fsyt_dz.getFittedSigmaWithError(h_dzrmsh[counter]);
  fsyt_dz.getFittedMeanWithError(h_dzmeanh[counter]);
  FitSlicesYTool fsyt_dzPt(dzres_vs_pt[counter]);
  fsyt_dzPt.getFittedSigmaWithError(h_dzrmshPt[counter]);
  fsyt_dzPt.getFittedMeanWithError(h_dzmeanhPt[counter]);
  FitSlicesYTool fsyt_phi(phires_vs_eta[counter]);
  fsyt_phi.getFittedSigmaWithError(h_phirmsh[counter]);
  fsyt_phi.getFittedMeanWithError(h_phimeanh[counter]);
  FitSlicesYTool fsyt_phiPt(phires_vs_pt[counter]);
  fsyt_phiPt.getFittedSigmaWithError(h_phirmshPt[counter]);
  fsyt_phiPt.getFittedMeanWithError(h_phimeanhPt[counter]);
  FitSlicesYTool fsyt_phiPhi(phires_vs_phi[counter]); 
  fsyt_phiPhi.getFittedSigmaWithError(h_phirmshPhi[counter]); 
  fsyt_phiPhi.getFittedMeanWithError(h_phimeanhPhi[counter]); 
  FitSlicesYTool fsyt_cotTheta(cotThetares_vs_eta[counter]);
  fsyt_cotTheta.getFittedSigmaWithError(h_cotThetarmsh[counter]);
  fsyt_cotTheta.getFittedMeanWithError(h_cotThetameanh[counter]);
  FitSlicesYTool fsyt_cotThetaPt(cotThetares_vs_pt[counter]);
  fsyt_cotThetaPt.getFittedSigmaWithError(h_cotThetarmshPt[counter]);
  fsyt_cotThetaPt.getFittedMeanWithError(h_cotThetameanhPt[counter]);

  //pulls of track params vs eta: get sigma from 2D histos
  FitSlicesYTool fsyt_dxyp(dxypull_vs_eta[counter]);
  fsyt_dxyp.getFittedSigmaWithError(h_dxypulleta[counter]);
  fsyt_dxyp.getFittedMeanWithError(h_dxypulletamean[counter]);
  FitSlicesYTool fsyt_ptp(ptpull_vs_eta[counter]);
  fsyt_ptp.getFittedSigmaWithError(h_ptpulleta[counter]);
  fsyt_ptp.getFittedMeanWithError(h_ptpulletamean[counter]);
  FitSlicesYTool fsyt_dzp(dzpull_vs_eta[counter]);
  fsyt_dzp.getFittedSigmaWithError(h_dzpulleta[counter]);
  fsyt_dzp.getFittedMeanWithError(h_dzpulletamean[counter]);
  FitSlicesYTool fsyt_phip(phipull_vs_eta[counter]);
  fsyt_phip.getFittedSigmaWithError(h_phipulleta[counter]);
  fsyt_phip.getFittedMeanWithError(h_phipulletamean[counter]);
  FitSlicesYTool fsyt_thetap(thetapull_vs_eta[counter]);
  fsyt_thetap.getFittedSigmaWithError(h_thetapulleta[counter]);
  fsyt_thetap.getFittedMeanWithError(h_thetapulletamean[counter]);
  //vs phi
  FitSlicesYTool fsyt_ptpPhi(ptpull_vs_phi[counter]);
  fsyt_ptpPhi.getFittedSigmaWithError(h_ptpullphi[counter]);
  fsyt_ptpPhi.getFittedMeanWithError(h_ptpullphimean[counter]);
  FitSlicesYTool fsyt_phipPhi(phipull_vs_phi[counter]);
  fsyt_phipPhi.getFittedSigmaWithError(h_phipullphi[counter]);
  fsyt_phipPhi.getFittedMeanWithError(h_phipullphimean[counter]);
  FitSlicesYTool fsyt_thetapPhi(thetapull_vs_phi[counter]);
  fsyt_thetapPhi.getFittedSigmaWithError(h_thetapullphi[counter]);
  fsyt_thetapPhi.getFittedMeanWithError(h_thetapullphimean[counter]);
  
  //effic&fake
  fillPlotFromVectors(h_effic[counter],totASSeta[counter],totSIMeta[counter],"effic");
  fillPlotFromVectors(h_fakerate[counter],totASS2eta[counter],totRECeta[counter],"fakerate");
  fillPlotFromVectors(h_efficPt[counter],totASSpT[counter],totSIMpT[counter],"effic");
  fillPlotFromVectors(h_fakeratePt[counter],totASS2pT[counter],totRECpT[counter],"fakerate");
  fillPlotFromVectors(h_effic_vs_hit[counter],totASS_hit[counter],totSIM_hit[counter],"effic");
  fillPlotFromVectors(h_fake_vs_hit[counter],totASS2_hit[counter],totREC_hit[counter],"fakerate");
  fillPlotFromVectors(h_effic_vs_phi[counter],totASS_phi[counter],totSIM_phi[counter],"effic");
  fillPlotFromVectors(h_fake_vs_phi[counter],totASS2_phi[counter],totREC_phi[counter],"fakerate");
  fillPlotFromVectors(h_effic_vs_dxy[counter],totASS_dxy[counter],totSIM_dxy[counter],"effic");
  fillPlotFromVectors(h_fake_vs_dxy[counter],totASS2_dxy[counter],totREC_dxy[counter],"fakerate");
  fillPlotFromVectors(h_effic_vs_dz[counter],totASS_dz[counter],totSIM_dz[counter],"effic");
  fillPlotFromVectors(h_fake_vs_dz[counter],totASS2_dz[counter],totREC_dz[counter],"fakerate");
  fillPlotFromVectors(h_effic_vs_vertpos[counter],totASS_vertpos[counter],totSIM_vertpos[counter],"effic");
  fillPlotFromVectors(h_effic_vs_zpos[counter],totASS_zpos[counter],totSIM_zpos[counter],"effic");

}


void MTVHistoProducerAlgoForTracker::fillProfileHistosFromVectors(int counter){
  //chi2 and #hit vs eta: get mean from 2D histos
  doProfileX(chi2_vs_eta[counter],h_chi2meanh[counter]);
  doProfileX(nhits_vs_eta[counter],h_hits_eta[counter]);
  doProfileX(nPXBhits_vs_eta[counter],h_PXBhits_eta[counter]);
  doProfileX(nPXFhits_vs_eta[counter],h_PXFhits_eta[counter]);
  doProfileX(nTIBhits_vs_eta[counter],h_TIBhits_eta[counter]);
  doProfileX(nTIDhits_vs_eta[counter],h_TIDhits_eta[counter]);
  doProfileX(nTOBhits_vs_eta[counter],h_TOBhits_eta[counter]);
  doProfileX(nTEChits_vs_eta[counter],h_TEChits_eta[counter]);

  doProfileX(nLayersWithMeas_vs_eta[counter],h_LayersWithMeas_eta[counter]);
  doProfileX(nPXLlayersWithMeas_vs_eta[counter],h_PXLlayersWithMeas_eta[counter]);    
  doProfileX(nSTRIPlayersWithMeas_vs_eta[counter],h_STRIPlayersWithMeas_eta[counter]);    
  doProfileX(nSTRIPlayersWith1dMeas_vs_eta[counter],h_STRIPlayersWith1dMeas_eta[counter]);
  doProfileX(nSTRIPlayersWith2dMeas_vs_eta[counter],h_STRIPlayersWith2dMeas_eta[counter]);



  doProfileX(nlosthits_vs_eta[counter],h_losthits_eta[counter]);
  //vs phi
  doProfileX(chi2_vs_nhits[counter],h_chi2meanhitsh[counter]); 
  //      doProfileX(ptres_vs_eta[counter],h_ptresmean_vs_eta[counter]);
  //      doProfileX(phires_vs_eta[counter],h_phiresmean_vs_eta[counter]);
  doProfileX(chi2_vs_phi[counter],h_chi2mean_vs_phi[counter]);
  doProfileX(nhits_vs_phi[counter],h_hits_phi[counter]);
  //       doProfileX(ptres_vs_phi[counter],h_ptresmean_vs_phi[counter]);
  //       doProfileX(phires_vs_phi[counter],h_phiresmean_vs_phi[counter]);
}

void MTVHistoProducerAlgoForTracker::fillHistosFromVectors(int counter){
  fillPlotFromVector(h_recoeta[counter],totRECeta[counter]);
  fillPlotFromVector(h_simuleta[counter],totSIMeta[counter]);
  fillPlotFromVector(h_assoceta[counter],totASSeta[counter]);
  fillPlotFromVector(h_assoc2eta[counter],totASS2eta[counter]);
  
  fillPlotFromVector(h_recopT[counter],totRECpT[counter]);
  fillPlotFromVector(h_simulpT[counter],totSIMpT[counter]);
  fillPlotFromVector(h_assocpT[counter],totASSpT[counter]);
  fillPlotFromVector(h_assoc2pT[counter],totASS2pT[counter]);
  
  fillPlotFromVector(h_recohit[counter],totREC_hit[counter]);
  fillPlotFromVector(h_simulhit[counter],totSIM_hit[counter]);
  fillPlotFromVector(h_assochit[counter],totASS_hit[counter]);
  fillPlotFromVector(h_assoc2hit[counter],totASS2_hit[counter]);
  
  fillPlotFromVector(h_recophi[counter],totREC_phi[counter]);
  fillPlotFromVector(h_simulphi[counter],totSIM_phi[counter]);
  fillPlotFromVector(h_assocphi[counter],totASS_phi[counter]);
  fillPlotFromVector(h_assoc2phi[counter],totASS2_phi[counter]);
  
  fillPlotFromVector(h_recodxy[counter],totREC_dxy[counter]);
  fillPlotFromVector(h_simuldxy[counter],totSIM_dxy[counter]);
  fillPlotFromVector(h_assocdxy[counter],totASS_dxy[counter]);
  fillPlotFromVector(h_assoc2dxy[counter],totASS2_dxy[counter]);

  fillPlotFromVector(h_recodz[counter],totREC_dz[counter]);
  fillPlotFromVector(h_simuldz[counter],totSIM_dz[counter]);
  fillPlotFromVector(h_assocdz[counter],totASS_dz[counter]);
  fillPlotFromVector(h_assoc2dz[counter],totASS2_dz[counter]);
  
  fillPlotFromVector(h_simulvertpos[counter],totSIM_vertpos[counter]);
  fillPlotFromVector(h_assocvertpos[counter],totASS_vertpos[counter]);
  
  fillPlotFromVector(h_simulzpos[counter],totSIM_zpos[counter]);
  fillPlotFromVector(h_assoczpos[counter],totASS_zpos[counter]);  
}
