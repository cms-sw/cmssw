#include "Validation/HcalHits/test/HcalHitValidation.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

#include "FWCore/Utilities/interface/Exception.h"

HcalHitValidation::HcalHitValidation(const edm::ParameterSet& ps) {

  g4Label  = ps.getUntrackedParameter<std::string>("moduleLabel","g4SimHits");
  hcalHits = ps.getUntrackedParameter<std::string>("HitCollection","HcalHits");
  layerInfo= ps.getUntrackedParameter<std::string>("LayerInfo","PHcalValidInfoLayer");
  nxNInfo  = ps.getUntrackedParameter<std::string>("NxNInfo","PHcalValidInfoNxN");
  jetsInfo = ps.getUntrackedParameter<std::string>("JetsInfo","PHcalValidInfoJets");
  outFile_ = ps.getUntrackedParameter<std::string>("outputFile", "hcValid.root");
  verbose_ = ps.getUntrackedParameter<bool>("Verbose", false);
  scheme_  = ps.getUntrackedParameter<bool>("TestNumbering", true);
  checkHit_= ps.getUntrackedParameter<bool>("CheckHits",  true);
  checkLay_= ps.getUntrackedParameter<bool>("CheckLayer", true);
  checkNxN_= ps.getUntrackedParameter<bool>("CheckNxN",   true);
  checkJet_= ps.getUntrackedParameter<bool>("CheckJets",  true);

  // register for data access
  tok_hh_ = consumes<edm::PCaloHitContainer>(edm::InputTag(g4Label,hcalHits));
  tok_iL_ = consumes<PHcalValidInfoLayer>(edm::InputTag(g4Label,layerInfo));
  tok_iN_ = consumes<PHcalValidInfoNxN>(edm::InputTag(g4Label,nxNInfo));
  tok_iJ_ = consumes<PHcalValidInfoJets>(edm::InputTag(g4Label,jetsInfo));

  edm::LogInfo("HcalHitValid") << "Module Label: " << g4Label << "   Hits: "
			       << hcalHits << " / "<< checkHit_ 
			       << "   LayerInfo: " << layerInfo << " / "
			       << checkLay_  << "  NxNInfo: " << nxNInfo 
			       << " / " << checkNxN_ << "  jetsInfo: "
			       << jetsInfo << " / " << checkJet_ 
			       << "   Output: " << outFile_ 
			       << "   Usage of TestNumberingScheme " <<scheme_;

}

HcalHitValidation::~HcalHitValidation() {
}

void HcalHitValidation::bookHistograms(DQMStore::IBooker & ibooker,
  edm::Run const &, edm::EventSetup const & ){

  ibooker.setCurrentFolder("HcalHitValidation");

  char title[60], name[20];
  double my_pi = 3.1415926;
  //Histograms for Hits
  if (checkHit_) {
    meAllNHit_   = ibooker.book1D("Hit01", "Number of Hits in HCal", 1000, 0., 5000.);
    meBadDetHit_ = ibooker.book1D("Hit02", "Hits with wrong Det",   100, 0., 100.);
    meBadSubHit_ = ibooker.book1D("Hit03", "Hits with wrong Subdet",100, 0., 100.);
    meBadIdHit_  = ibooker.book1D("Hit04", "Hits with wrong ID",    100, 0., 100.);
    meHBNHit_    = ibooker.book1D("Hit05", "Number of Hits in HB", 1000, 0., 5000.);
    meHENHit_    = ibooker.book1D("Hit06", "Number of Hits in HE", 1000, 0., 5000.);
    meHONHit_    = ibooker.book1D("Hit07", "Number of Hits in HO", 1000, 0., 5000.);
    meHFNHit_    = ibooker.book1D("Hit08", "Number of Hits in HF", 1000, 0., 5000.);
    meDetectHit_ = ibooker.book1D("Hit09", "Detector ID",           50, 0., 50.);
    meSubdetHit_ = ibooker.book1D("Hit10", "Subdetectors in HCal",  50, 0., 50.);
    meDepthHit_  = ibooker.book1D("Hit11", "Depths in HCal",        20, 0., 20.);
    meEtaHit_    = ibooker.book1D("Hit12", "Eta in HCal",          100, -50., 50.);
    mePhiHit_    = ibooker.book1D("Hit13", "Phi in HCal",          100, 0., 100.);
    meEnergyHit_ = ibooker.book1D("Hit14", "Energy in HCal",       100, 0., 1.);
    meTimeHit_   = ibooker.book1D("Hit15", "Time in HCal",         100, 0., 400.);
    meTimeWHit_  = ibooker.book1D("Hit16", "Time in HCal (E wtd)", 100, 0., 400.);
    meHBDepHit_  = ibooker.book1D("Hit17", "Depths in HB",          20, 0., 20.);
    meHEDepHit_  = ibooker.book1D("Hit18", "Depths in HE",          20, 0., 20.);
    meHODepHit_  = ibooker.book1D("Hit19", "Depths in HO",          20, 0., 20.);
    meHFDepHit_  = ibooker.book1D("Hit20", "Depths in HF",          20, 0., 20.);
    meHBEtaHit_  = ibooker.book1D("Hit21", "Eta in HB",            100, -50., 50.);
    meHEEtaHit_  = ibooker.book1D("Hit22", "Eta in HE",            100, -50., 50.);
    meHOEtaHit_  = ibooker.book1D("Hit23", "Eta in HO",            100, -50., 50.);
    meHFEtaHit_  = ibooker.book1D("Hit24", "Eta in HF",            100, -50., 50.);
    meHBPhiHit_  = ibooker.book1D("Hit25", "Phi in HB",            100, 0., 100.);
    meHEPhiHit_  = ibooker.book1D("Hit26", "Phi in HE",            100, 0., 100.);
    meHOPhiHit_  = ibooker.book1D("Hit27", "Phi in HO",            100, 0., 100.);
    meHFPhiHit_  = ibooker.book1D("Hit28", "Phi in HF",            100, 0., 100.);
    meHBEneHit_  = ibooker.book1D("Hit29", "Energy in HB",         100, 0., 1.);
    meHEEneHit_  = ibooker.book1D("Hit30", "Energy in HE",         100, 0., 1.);
    meHOEneHit_  = ibooker.book1D("Hit31", "Energy in HO",         100, 0., 1.);
    meHFEneHit_  = ibooker.book1D("Hit32", "Energy in HF",         100, 0., 100.);
    meHBTimHit_  = ibooker.book1D("Hit33", "Time in HB",           100, 0., 400.);
    meHETimHit_  = ibooker.book1D("Hit34", "Time in HE",           100, 0., 400.);
    meHOTimHit_  = ibooker.book1D("Hit35", "Time in HO",           100, 0., 400.);
    meHFTimHit_  = ibooker.book1D("Hit36", "Time in HF",           100, 0., 400.);
    mePMTHit_    = ibooker.book1D("Hit37", "Number of Hit in PMT",1000, 0., 1000.);
    mePMTDepHit_ = ibooker.book1D("Hit38", "Depths in HF PMT",      20, 0., 20.);
    mePMTEtaHit_ = ibooker.book1D("Hit39", "Eta in HF PMT",        100, -50., 50.);
    mePMTPhiHit_ = ibooker.book1D("Hit40", "Phi in HF PMT",        100, 0., 100.);
    mePMTEn1Hit_ = ibooker.book1D("Hit41", "Energy (Ceren) in PMT",100, 0., 100.);
    mePMTEn2Hit_ = ibooker.book1D("Hit42", "Energy (dE/dx) in PMT",100, 0., 100.);
    mePMTTimHit_ = ibooker.book1D("Hit43", "Time in HF PMT",       100, 0., 400.);
  }

  //Histograms for Layers
  if (checkLay_) {
    meLayerLay_ = ibooker.book1D("Lay01", "Layer # of the Hits", 20, 0., 20.);
    meEtaHLay_  = ibooker.book1D("Lay02", "Eta of the Hits", 100, -5., 5.);
    mePhiHLay_  = ibooker.book1D("Lay03", "Phi of the Hits", 100, -my_pi, my_pi);
    meEneHLay_  = ibooker.book1D("Lay04", "Energy of the Hits", 100, 0., 2.);
    meDepHlay_  = ibooker.book1D("Lay05", "Depth  of the Hits", 10, 0., 10.);
    meTimHLay_  = ibooker.book1D("Lay06", "Time of the Hits",   100, 0., 400.);
    meTimWLay_  = ibooker.book1D("Lay07", "Time (wtd) of Hits", 100, 0., 400.);
    meEtaPhi_   = ibooker.book2D("Lay08", "Phi%Eta of the Hits", 100, -5., 5.,
        100,-my_pi,my_pi);

    meHitELay_  = ibooker.book1D("Lay09", "Hit in Ecal", 1000, 0., 2000.);
    meHitHLay_  = ibooker.book1D("Lay10", "Hit in Hcal", 1000, 0., 2000.);
    meHitTLay_  = ibooker.book1D("Lay11", "Total Hits",  1000, 0., 2000.);
    meEneLLay_  = ibooker.book1D("Lay12", "Energy per layer", 100, 0., 1.);
    int nn=0;
    for (int i = 0; i < nLayersMAX; i++) {
      sprintf (name,  "Layl%d", nn); nn++;
      sprintf (title, "Energy deposit in Layer %d", i);
      meEneLay_[i] = ibooker.book1D(name, title, 100, 0., 0.4);
    }
    meLngLay_   = ibooker.book1D("Lay13", "Lonitudinal Shower Profile",20, 0, 20.);
    meEneDLay_  = ibooker.book1D("Lay14", "Energy per depth", 100, 0., 1.);
    for (int i = 0; i < nDepthsMAX; i++) {
      sprintf (name,  "Layl%d", nn); nn++;
      sprintf (title, "Energy deposit in Depth %d", i);
      meDepLay_[i] = ibooker.book1D(name, title, 100, 0., 2.);
    }

    meEtotLay_  = ibooker.book1D("Lay15", "Total Energy", 100, 0., 1.);
    meEHOLay_   = ibooker.book1D("Lay16", "Energy in HO", 100, 0., 2000.);
    meEHBHELay_ = ibooker.book1D("Lay17", "Energy in HB/HE", 100, 0., 2000.);
    meEFibLLay_ = ibooker.book1D("Lay18", "Energy in HF (Long)",  100, 0., 100.);
    meEFibSLay_ = ibooker.book1D("Lay19", "Energy in HF (Short)", 100, 0., 100.);
    meEHFEmLay_ = ibooker.book1D("Lay20", "EM   energy in HF", 100, 0., 200.);
    meEHFHdLay_ = ibooker.book1D("Lay21", "Had. energy in HF", 100, 0., 200.);
  }

  // Histograms for NxN analysis
  if (checkNxN_) {
    meEcalRNxN_ = ibooker.book1D("NxN01", "Energy in ECal (NxN)r", 100, 0., 200.);
    meHcalRNxN_ = ibooker.book1D("NxN02", "Energy in HCal (NxN)r", 100, 0., 200.);
    meHoRNxN_   = ibooker.book1D("NxN03", "Energy in HO (NxN)r",  100, 0., 200.);
    meEtotRNxN_ = ibooker.book1D("NxN04", "Energy Total (NxN)r",  100, 0., 200.);
    meEcalNxN_  = ibooker.book1D("NxN05", "Energy in ECal (NxN)", 100, 0., 200.);
    meHcalNxN_  = ibooker.book1D("NxN06", "Energy in HCal (NxN)", 100, 0., 200.);
    meHoNxN_    = ibooker.book1D("NxN07", "Energy in HO (NxN)",   100, 0., 200.);
    meEtotNxN_  = ibooker.book1D("NxN08", "Energy Total (NxN)",   100, 0., 200.);
    meEiNxN_    = ibooker.book1D("NxN09", "Energy of Hits in (NxN)", 100, 0., 1.);
    meTiNxN_    = ibooker.book1D("NxN10", "Time   of Hits in (NxN)", 100, 0., 400.);
    meTrNxN_    = ibooker.book1D("NxN11", "Dist.  of Hits in (NxN)", 100, 0., 1.);
  }

  //Histograms for Jets
  if (checkJet_) {
    meRJet_    = ibooker.book1D("Jet01", "R of Hits", 100, 0., 1.);
    meTJet_    = ibooker.book1D("Jet02", "T of Hits", 100, 0., 200.);
    meEJet_    = ibooker.book1D("Jet03", "E of Hits", 100, 0., 200.);
    meEcalJet_ = ibooker.book1D("Jet04", "Ecal Energy (First Jet)", 100, 0., 200.);
    meHcalJet_ = ibooker.book1D("Jet05", "Hcal Energy (First Jet)", 100, 0., 200.);
    meHoJet_   = ibooker.book1D("Jet06", "Ho   Energy (First Jet)", 100, 0., 200.);
    meEtotJet_ = ibooker.book1D("Jet07", "Total Energy(First Jet)", 100, 0., 200.);
    meEcHcJet_ = ibooker.book2D("Jet08", "Energy in Hcal% Ecal", 100,0., 200., 100,
        0., 200.);

    meDetaJet_ = ibooker.book1D("Jet09", "Delta Eta", 100, 0., 2.);
    meDphiJet_ = ibooker.book1D("Jet10", "Delta Phi", 100, 0., 1.);
    meDrJet_   = ibooker.book1D("Jet11", "Delta R",   100, 0., 2.);
    meMassJet_ = ibooker.book1D("Jet12", "Di-jet mass", 100, 0., 200.);
    meEneJet_  = ibooker.book1D("Jet13", "Jet Energy", 100, 0., 200.);
    meEtaJet_  = ibooker.book1D("Jet14", "Jet Eta", 100, -5., 5.);
    mePhiJet_  = ibooker.book1D("Jet15", "Jet Phi", 100, -my_pi, my_pi);
  }
}

void HcalHitValidation::analyze(const edm::Event& e, const edm::EventSetup& ) {

  LogDebug("HcalHitValid") << "Run = " << e.id().run() << " Event = " 
			   << e.id().event();

  std::vector<PCaloHit>               caloHits;
  edm::Handle<edm::PCaloHitContainer> hitsHcal;
  edm::Handle<PHcalValidInfoLayer>    infoLayer;
  edm::Handle<PHcalValidInfoNxN>      infoNxN;
  edm::Handle<PHcalValidInfoJets>     infoJets;

  bool getHits = false;
  if (checkHit_) {
    e.getByToken(tok_hh_,hitsHcal); 
    if (hitsHcal.isValid()) getHits = true;
  }

  bool getLayer = false;
  if (checkLay_) {
    e.getByToken(tok_iL_,infoLayer);
    if (infoLayer.isValid()) getLayer = true;
  }

  bool getNxN = false;
  if (checkNxN_) {
    e.getByToken(tok_iN_,infoNxN);
    if (infoNxN.isValid()) getNxN = true;
  }

  bool getJets = false;
  if (checkJet_) {
    e.getByToken(tok_iJ_,infoJets);
    if (infoJets.isValid()) getJets = true;
  }

  LogDebug("HcalHitValid") << "HcalValidation: Input flags Hits " << getHits 
			   << ", Layer " << getLayer << ", NxN " << getNxN
			   << ", Jets " << getJets;

  if (getHits) {
    caloHits.insert(caloHits.end(),hitsHcal->begin(),hitsHcal->end());
    LogDebug("HcalHitValid") << "HcalValidation: Hit buffer " 
			     << caloHits.size(); 
    analyzeHits (caloHits);
  }

  if (getLayer) analyzeLayer (infoLayer);
  if (getNxN)   analyzeNxN   (infoNxN);
  if (getJets)  analyzeJets  (infoJets);
}

void HcalHitValidation::analyzeHits (std::vector<PCaloHit>& hits) {

  int nHit = hits.size();
  int nHB=0, nHE=0, nHO=0, nHF=0, nPMT=0, nBad1=0, nBad2=0, nBad=0;
  for (int i=0; i<nHit; i++) {
    double energy    = hits[i].energy();
    double time      = hits[i].time();
    unsigned int id_ = hits[i].id();
    int det, subdet, depth, eta, phi;
    if (scheme_) {
      det            = 4;
      subdet         = (id_>>28)&15;
      depth          = (id_>>26)&3; depth++;
      int zside      = ((id_&0x100000)?(1):(-1)); 
      eta            = zside*((id_>>10)&1023);
      phi            = (id_&1023);
    } else {
      HcalDetId id   = HcalDetId(id_);
      det            = id.det();
      subdet         = id.subdet();
      depth          = id.depth();
      eta            = id.ieta();
      phi            = id.iphi();
    }
    uint16_t depth_  = hits[i].depth();
    double energyEM  = hits[i].energyEM();
    double energyHad = hits[i].energyHad();
    LogDebug("HcalHitValid") << "Hit[" << i << "] ID " << std::hex << id_ 
	         << std::dec << " Det " << det << " Sub "
	         << subdet << " depth " << depth << " " << depth_
	         << " Eta " << eta << " Phi " << phi << " E "
	         << energy << "(EM " << energyEM << ", Had "
	         << energyHad << ") time " << time;
    if (det ==  4) { // Check DetId.h
      if (subdet == static_cast<int>(HcalBarrel))
      {
	      nHB++;
      } else if (subdet == static_cast<int>(HcalEndcap))
      {
	      nHE++;
      } else if (subdet == static_cast<int>(HcalOuter))
      {
	      nHO++;
      } else if (subdet == static_cast<int>(HcalForward))
      {
	      if (depth_ == 0) nHF++;
	      else             nPMT++;
      } else
      {
	      nBad++;
        nBad2++;
      }
    } else
    {
      nBad++;
      nBad1++;
    }

    meDetectHit_->Fill(double(det));
    if (det ==  4 && depth_ == 0) {
	    meSubdetHit_->Fill(double(subdet));
	    meDepthHit_->Fill(double(depth));
	    meEtaHit_->Fill(double(eta));
	    mePhiHit_->Fill(double(phi));
	    meEnergyHit_->Fill(energy);
	    meTimeHit_->Fill(time);
	    meTimeWHit_->Fill(double(time),energy);
	    if (subdet == static_cast<int>(HcalBarrel)) {
	      meHBDepHit_->Fill(double(depth));
	      meHBEtaHit_->Fill(double(eta));
	      meHBPhiHit_->Fill(double(phi));
	      meHBEneHit_->Fill(energy);
	      meHBTimHit_->Fill(time);
	    } else if (subdet == static_cast<int>(HcalEndcap)) {
	      meHEDepHit_->Fill(double(depth));
	      meHEEtaHit_->Fill(double(eta));
	      meHEPhiHit_->Fill(double(phi));
	      meHEEneHit_->Fill(energy);
	      meHETimHit_->Fill(time);
	    } else if (subdet == static_cast<int>(HcalOuter)) {
	      meHODepHit_->Fill(double(depth));
	      meHOEtaHit_->Fill(double(eta));
	      meHOPhiHit_->Fill(double(phi));
	      meHOEneHit_->Fill(energy);
	      meHOTimHit_->Fill(time);
	    } else if (subdet == static_cast<int>(HcalForward)) {
	      meHFDepHit_->Fill(double(depth));
	      meHFEtaHit_->Fill(double(eta));
	      meHFPhiHit_->Fill(double(phi));
	      meHFEneHit_->Fill(energy);
	      meHFTimHit_->Fill(time);
	    }
    }
    else if (det == 0 && subdet == static_cast<int>(HcalForward))
    {
	    mePMTDepHit_->Fill(double(depth));
	    mePMTEtaHit_->Fill(double(eta));
	    mePMTPhiHit_->Fill(double(phi));
	    mePMTEn1Hit_->Fill(energyEM);
	    mePMTEn2Hit_->Fill(energyHad);
	    mePMTTimHit_->Fill(time);
    }
  }
  meAllNHit_->Fill(double(nHit));
  meBadDetHit_->Fill(double(nBad1));
  meBadSubHit_->Fill(double(nBad2));
  meBadIdHit_->Fill(double(nBad));
  meHBNHit_->Fill(double(nHB));
  meHENHit_->Fill(double(nHE));
  meHONHit_->Fill(double(nHO));
  meHFNHit_->Fill(double(nHF));
  mePMTHit_->Fill(double(nPMT));

  LogDebug("HcalHitValid") << "HcalHitValidation::analyzeHits: HB " << nHB
      << " HE " << nHE << " HO " << nHO << " HF " << nHF
      << " PMT " << nPMT << " Bad " << nBad << " All "
      << nHit;

}

void HcalHitValidation::analyzeLayer (edm::Handle<PHcalValidInfoLayer>& infoLayer) {

  // CaloHits from PHcalValidInfoLayer  
  int                    nHits = infoLayer->nHit();
  std::vector<float>    idHits = infoLayer->idHit();
  std::vector<float>   phiHits = infoLayer->phiHit();
  std::vector<float>   etaHits = infoLayer->etaHit();
  std::vector<float> layerHits = infoLayer->layerHit();
  std::vector<float>     eHits = infoLayer->eHit();
  std::vector<float>     tHits = infoLayer->tHit();

  int ne = 0, nh = 0; 
  for (int j = 0; j < nHits; j++) {
    int layer = (int)(layerHits[j])-1;
    int id    = (int)(idHits[j]);
    
    if(id >= 10) {ne++;}
    else {nh++;}

    LogDebug("HcalHitValid") << "HcalHitValidation::analyzeLayer:Hit "
			     << "subdet = " << id  << "  lay = " << layer;

    meLayerLay_->Fill(double(layer));
    meEtaHLay_->Fill(etaHits[j]);
    mePhiHLay_->Fill(phiHits[j]);
    meEneHLay_->Fill(eHits[j]);
    meDepHlay_->Fill(idHits[j]);
    meTimHLay_->Fill(tHits[j]);
    meTimWLay_->Fill(tHits[j],eHits[j]);
    if (id < 6) // HCAL only. Depth is needed, not layer !!!
    {
	    meEtaPhi_->Fill(etaHits[j],phiHits[j]);
    }
  }
  
  meHitELay_->Fill(double(ne));
  meHitHLay_->Fill(double(nh));
  meHitTLay_->Fill(double(nHits));

  // Layers and depths PHcalValidInfoLayer
  std::vector<float> eLayer = infoLayer->elayer();
  std::vector<float> eDepth = infoLayer->edepth();
  float eTot = 0.;

  for (int j = 0; j < nLayersMAX ; j++)
  {
    eTot += eLayer[j];
    meEneLLay_->Fill(eLayer[j]);
    meEneLay_[j]->Fill(eLayer[j]);
    meLngLay_->Fill((double)(j),eLayer[j]);  // HCAL SimHits only
  }
  for (int j = 0; j < nDepthsMAX; j++)
  {
    meEneDLay_->Fill(eDepth[j]);
    meDepLay_[j]->Fill(eDepth[j]);
  }
  meEtotLay_->Fill(eTot);
       
  // The rest  PHcalValidInfoLayer
  double eHO      =  infoLayer->eho(); 
  double eHBHE    =  infoLayer->ehbhe(); 
  double elongHF  =  infoLayer->elonghf(); 
  double eshortHF =  infoLayer->eshorthf(); 
  double eEcalHF  =  infoLayer->eecalhf(); 
  double eHcalHF  =  infoLayer->ehcalhf(); 

  meEHOLay_->Fill(eHO);
  meEHBHELay_->Fill(eHBHE);
  meEFibLLay_->Fill(elongHF);
  meEFibSLay_->Fill(eshortHF);
  meEHFEmLay_->Fill(eEcalHF);
  meEHFHdLay_->Fill(eHcalHF);

  LogDebug("HcalHitValid") << "HcalHitValidation::analyzeLayer: eHO " << eHO
      << "  eHBHE = " << eHBHE  << " elongHF = "
      << elongHF << " eshortHF = " << eshortHF
      << "  eEcalHF = " << eEcalHF << "  eHcalHF = "
      << eHcalHF;
}

void HcalHitValidation::analyzeNxN (edm::Handle<PHcalValidInfoNxN>& infoNxN) {

  // NxN quantities
  double ecalNxNr = infoNxN->ecalnxnr();
  double hcalNxNr = infoNxN->hcalnxnr();
  double   hoNxNr = infoNxN->honxnr();
  double etotNxNr = infoNxN->etotnxnr();
  
  double ecalNxN  = infoNxN->ecalnxn();
  double hcalNxN  = infoNxN->hcalnxn();
  double   hoNxN  = infoNxN->honxn();
  double etotNxN  = infoNxN->etotnxn();

  meEcalRNxN_->Fill(ecalNxNr);
  meHcalRNxN_->Fill(hcalNxNr);
  meHoRNxN_->Fill(hoNxNr);
  meEtotRNxN_->Fill(etotNxNr);
   
  meEcalNxN_->Fill(ecalNxN);
  meHcalNxN_->Fill(hcalNxN);
  meHoNxN_->Fill(hoNxN);
  meEtotNxN_->Fill(etotNxN);
   
  int                    nIxI = infoNxN->nnxn();
  std::vector<float>    idIxI = infoNxN->idnxn();
  std::vector<float>     eIxI = infoNxN->enxn();
  std::vector<float>     tIxI = infoNxN->tnxn();
 
  for (int j = 0; j < nIxI ; j++) // NB !!! j < nIxI
  {
    meEiNxN_->Fill(eIxI[j]);
    meTiNxN_->Fill(tIxI[j]);
    meTrNxN_->Fill(idIxI[j],eIxI[j]);  // transverse profile
  }

  LogDebug("HcalHitValid") << "HcalHitValidation::analyzeNxN: " << nIxI
      << " hits in NxN analysis; Total Energy " << etotNxN
      << "/" << etotNxNr;
}

void HcalHitValidation::analyzeJets (edm::Handle<PHcalValidInfoJets>& infoJets) {

  // -- Leading Jet
  int nJetHits =  infoJets->njethit();

  std::vector<float> rJetHits = infoJets->jethitr();
  std::vector<float> tJetHits = infoJets->jethitt();
  std::vector<float> eJetHits = infoJets->jethite();

  double ecalJet = infoJets->ecaljet();
  double hcalJet = infoJets->hcaljet();
  double   hoJet = infoJets->hojet();
  double etotJet = infoJets->etotjet();

  double detaJet = infoJets->detajet();
  double dphiJet = infoJets->dphijet();
  double   drJet = infoJets->drjet();
  double  dijetM = infoJets->dijetm();

  for (int j = 0; j < nJetHits; j++)
  {
    meRJet_->Fill(rJetHits[j]);
    meTJet_->Fill(tJetHits[j]);
    meEJet_->Fill(eJetHits[j]);
  }

  meEcalJet_->Fill(ecalJet);
  meHcalJet_->Fill(hcalJet);
  meHoJet_->Fill(hoJet);
  meEtotJet_->Fill(etotJet);
  meEcHcJet_->Fill(ecalJet,hcalJet);

  meDetaJet_->Fill(detaJet);
  meDphiJet_->Fill(dphiJet);
  meDrJet_->Fill(drJet);
  meMassJet_->Fill(dijetM);

  // All Jets 
  int                nJets  = infoJets->njet();
  std::vector<float> jetE   = infoJets->jete();
  std::vector<float> jetEta = infoJets->jeteta();
  std::vector<float> jetPhi = infoJets->jetphi();
  
  for (int j = 0; j < nJets; j++) {
    meEneJet_->Fill(jetE[j]);
    meEtaJet_->Fill(jetEta[j]);
    mePhiJet_->Fill(jetPhi[j]);
  }
  LogDebug("HcalHitValid") << "HcalHitValidation::analyzeJets: " << nJets
      << " jets with "  << nJetHits << " hits in the "
      << "leading jet\n" << "   d(Eta) = " << detaJet
      << "  d(Phi) = " << dphiJet << "  d(R) = " << drJet
      << "  diJet Mass = " << dijetM;
}

//define this as a plug-in
DEFINE_FWK_MODULE(HcalHitValidation);
