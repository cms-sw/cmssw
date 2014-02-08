/*Class MBUEandQCDValidation
 *  
 *  Class to fill dqm monitor elements from existing EDM file
 *
 */
 
#include "Validation/EventGenerator/interface/MBUEandQCDValidation.h"
#include "Validation/EventGenerator/interface/HepMCValidationHelper.h"

#include "CLHEP/Units/defs.h"
#include "CLHEP/Units/PhysicalConstants.h"
#include "CLHEP/Units/SystemOfUnits.h"

#include "fastjet/JetDefinition.hh"
#include "fastjet/ClusterSequence.hh"

using namespace edm;

MBUEandQCDValidation::MBUEandQCDValidation(const edm::ParameterSet& iPSet): 
  wmanager_(iPSet,consumesCollector()),
  hepmcCollection_(iPSet.getParameter<edm::InputTag>("hepmcCollection")),
  genchjetCollection_(iPSet.getParameter<edm::InputTag>("genChjetsCollection")),
  genjetCollection_(iPSet.getParameter<edm::InputTag>("genjetsCollection")),
  verbosity_(iPSet.getUntrackedParameter<unsigned int>("verbosity",0))
{    

  hepmcGPCollection.reserve(initSize);
  hepmcCharge.reserve(initSize);

  theCalo= new CaloCellManager(verbosity_);

  eneInCell.resize(CaloCellManager::nCaloCell);

  hepmcCollectionToken_=consumes<HepMCProduct>(hepmcCollection_);
  genjetCollectionToken_=consumes<reco::GenJetCollection>(genjetCollection_);
  genchjetCollectionToken_=consumes<reco::GenJetCollection>(genchjetCollection_);

}

MBUEandQCDValidation::~MBUEandQCDValidation() {

  delete theCalo;
  
}

void MBUEandQCDValidation::dqmBeginRun(const edm::Run& r, const edm::EventSetup& c) {
  c.getData( fPDGTable );
}


void MBUEandQCDValidation::bookHistograms(DQMStore::IBooker &i, edm::Run const &, edm::EventSetup const &){
	///Setting the DQM top directories
	i.setCurrentFolder("Generator/MBUEandQCD");
	
	///Booking the ME's
    
    // Number of analyzed events
    nEvt = i.book1D("nEvt", "n analyzed Events", 1, 0., 1.);

    // Number of events with no forward trigger
    nNoFwdTrig = i.book1D("nNoFwdTrig", "n Events no forward trigger", 1, 0., 1.);
	
    // Number of events with a single arm forward trigger
    nSaFwdTrig = i.book1D("nSaFwdTrig", "n Events single arm forward trigger", 1, 0., 1.);

    // Number of events with b quark
    nbquark = i.book1D("nbquark", "n Events with b quark", 1, 0., 1.);
    
    // Number of events with c and b quark
    ncandbquark = i.book1D("ncandbquark", "n Events with c and b quark", 1, 0., 1.);
    
    // Number of events with c and no b quark
    ncnobquark = i.book1D("ncnobquark", "n Events with c and no b quark", 1, 0., 1.);
    

    // Number of selected events for QCD-09-010
    nEvt1 = i.book1D("nEvt1", "n Events QCD-09-010", 1, 0., 1.);
    // dNchdpt QCD-09-010
	dNchdpt1 = i.book1D("dNchdpt1", "dNchdpt QCD-09-010", 30, 0., 6.); 
    // dNchdeta QCD-09-010
    dNchdeta1 = i.book1D("dNchdeta1", "dNchdeta QCD-09-010", 10, -2.5, 2.5);
    // Number of selected events for QCD-10-001

    nEvt2 = i.book1D("nEvt2", "n Events QCD-10-001", 1, 0., 1.);
    // Leading track pt QCD-10-001
    leadTrackpt = i.book1D("leadTrackpt", "leading track pt QCD-10-001", 200, 0., 100.);
    // Leading track eta QCD-10-001
    leadTracketa = i.book1D("leadTracketa", "leading track eta QCD-10-001", 50., -2.5,2.5);
    // transverse charged particle density vs leading track pt
    nChaDenLpt = i.bookProfile("nChaDenLpt", "charged density vs leading pt", 200, 0., 100., 0., 100., " ");
    // transverse charged particle density vs leading track pt
    sptDenLpt = i.bookProfile("sptDenLpt", "sum pt density vs leading pt", 200, 0., 100., 0., 300., " ");
    // dNchdpt QCD-10-001 transverse
	dNchdpt2 = i.book1D("dNchdpt2", "dNchdpt QCD-10-001", 200, 0., 100.); 
    // dNchdeta QCD-10-001 transverse
    dNchdeta2 = i.book1D("dNchdeta2", "dNchdeta QCD-10-001", 50, -2.5, 2.5);
    // nCha QCD-10-001 transverse
    nCha = i.book1D("nCha", "n charged QCD-10-001", 100, 0., 100.);
    // dNchdSpt transverse
    dNchdSpt = i.book1D("dNchdSpt", "dNchdSpt QCD-10-001", 300, 0., 300.);
    // dNchdphi
    dNchdphi = i.bookProfile("dNchdphi", "dNchdphi QCD-10-001", nphiBin, -180., 180., 0., 30., " ");
    // dSptdphi
    dSptdphi = i.bookProfile("dSptdphi", "dSptdphi QCD-10-001", nphiBin, -180., 180., 0., 30., " ");

    // number of charged jets QCD-10-001
    nChj = i.book1D("nChj", "n charged jets QCD-10-001", 30, 0, 30.);
    // dNchjdeta QCD-10-001
    dNchjdeta = i.book1D("dNchjdeta", "dNchjdeta QCD-10-001", 50, -2.5, 2.5);
    // dNchjdpt QCD-10-001
    dNchjdpt = i.book1D("dNchjdpt", "dNchjdpt QCD-10-001", 100, 0., 100.);
    // leading charged jet pt QCD-10-001
    leadChjpt = i.book1D("leadChjpt", "leadChjpt QCD-10-001", 100, 0., 100.);
    // leading charged jet eta QCD-10-001
    leadChjeta = i.book1D("leadChjeta", "leadChjeta QCD-10-001", 50, -2.5, 2.5);
    // (pt1+pt2)/ptot
    pt1pt2optotch = i.bookProfile("pt1pt2optotch", "sum 2 leading jets over ptot", 50, 0., 100., 0., 1., " ");

    // particle rates in tracker acceptance
    nPPbar = i.book1D("nPPbar", "nPPbar QCD-10-001", 30, 0., 30.);
    nKpm = i.book1D("nKpm", "nKpm QCD-10-001", 30, 0., 30.);
    nK0s = i.book1D("nK0s", "nK0s QCD-10-001", 30, 0., 30.);
    nL0 = i.book1D("nL0", "nL0 QCD-10-001", 30, 0., 30.);
    nXim = i.book1D("nXim", "nXim QCD-10-001", 30, 0., 30.);
    nOmega = i.book1D("nOmega", "nOmega QCD-10-001", 30, 0., 30.);

    pPPbar = i.book1D("pPPbar", "Log10(pt) PPbar QCD-10-001", 25, -2., 3.);
    pKpm = i.book1D("pKpm", "Log10(pt) Kpm QCD-10-001", 25, -2., 3.);
    pK0s = i.book1D("pK0s", "Log10(pt) K0s QCD-10-001", 25, -2., 3.);
    pL0 = i.book1D("pL0", "Log10(pt) L0 QCD-10-001", 25, -2., 3.);
    pXim = i.book1D("pXim", "Log10(pt) Xim QCD-10-001", 25, -2., 3.);
    pOmega = i.book1D("pOmega", "Log10(pt) Omega QCD-10-001", 25, -2., 3.);

    // neutral rate in the barrel + HF acceptance
    nNNbar = i.book1D("nNNbar", "nNNbar QCD-10-001", 30, 0., 30.);
    nGamma = i.book1D("nGamma", "nGamma QCD-10-001", 50, 0., 200.);

    pNNbar = i.book1D("pNNbar", "Log10(pt) NNbar QCD-10-001", 25, -2., 3.);
    pGamma = i.book1D("pGamma", "Log10(pt) Gamma QCD-10-001", 25, -2., 3.);

    // highest pt electron spectrum
    elePt = i.book1D("elePt", "highest pt electron Log10(pt)", 30, -2., 4.);

    // highest pt muon spectrum
    muoPt = i.book1D("muoPt", "highest pt muon Log10(pt)", 30, -2., 4.);


    // number of selected di-jet events
    nDijet = i.book1D("nDijet", "n Dijet Events", 1, 0., 1.);
    // number of jets 
    nj = i.book1D("nj", "n jets ", 30, 0, 30.);
    // dNjdeta 
    dNjdeta = i.book1D("dNjdeta", "dNjdeta ", 50, -5., 5.);
    // dNjdpt 
    dNjdpt = i.book1D("dNjdpt", "dNjdpt ", 60, 0., 300.);
    // (pt1+pt2)/ptot
    pt1pt2optot = i.bookProfile("pt1pt2optot", "sum 2 leading jets over Et tot ", 60, 0., 300., 0., 1., " ");
    // pt1-pt2
    pt1pt2balance = i.book1D("pt1pt2balance", "2 leading jets pt difference ", 10, 0., 1.);
    // pt1 pt2 Delta phi
    pt1pt2Dphi = i.book1D("pt1pt2Dphi", "pt1 pt2 delta phi ", nphiBin, 0., 180.);
    // pt1 pt2 invariant mass
    pt1pt2InvM = i.book1D("pt1pt2InvM", "pt1 pt2 invariant mass ", 60, 0., 600.);
    // pt3 fraction
    pt3Frac = i.book1D("pt3Frac", "2 pt3 over pt1+pt2 ", 30, 0., 1.);
    // sum of jets Et
    sumJEt = i.book1D("sumJEt", "sum Jet Et ", 60, 0., 300.);
    // fraction of missing Et over sum of jets Et
    missEtosumJEt = i.book1D("missEtosumJEt", "missing Et over sumJet Et ", 30, 0., 1.);
    // sum of final state particle Pt
    sumPt = i.book1D("sumPt", "sum particle Pt ", 60, 0., 600.);
    // sum of final state charged particle Pt
    sumChPt = i.book1D("sumChPt", "sum charged particle Pt ", 60, 0., 300.);

    //Number of selected events for the HF energy flux analysis
    nHFflow = i.book1D("nHFflow", "n HF flow events", 1, 0., 1.);
    //Forward energy flow for MinBias BSC selection
    dEdetaHFmb = i.bookProfile("dEdetaHFmb", "dEdeta HF MinBias", (int)CaloCellManager::nForwardEta, 0, (double)CaloCellManager::nForwardEta, 0., 300., " ");
    //Forward energy flow for QCD dijet selection
    dEdetaHFdj = i.bookProfile("dEdetaHFdj", "dEdeta HF QCD dijet", (int)CaloCellManager::nForwardEta, 0, (double)CaloCellManager::nForwardEta, 0., 300., " ");

    // FWD-10-001 like diffraction analysis
    nHFSD = i.book1D("nHFSD","n single diffraction in HF", 1, 0., 1.);
    // E-pz HF-
    EmpzHFm = i.book1D("EmpzHFm", "E-pz HF- SD", 40, 0., 200.);
    // Number of cells above threshold
    ntHFm = i.book1D("ntHFm", "number of HF- tower SD", 20, 0., 20.);
    // Energy in HF-
    eneHFmSel = i.book1D("eneHFmSel", "energy in HF-", 40, 0., 200.);

    // number of jets accepted in the 'Jet-Multiplicity' analysis
    _JM25njets = i.book1D("JM25njets", "n jets", 15, 0, 15.);    
    _JM25ht = i.book1D("JM25ht", "HT", 80, 0, 800.);    
    _JM25pt1 = i.book1D("JM25pt1", "pt", 40, 0, 200.);    
    _JM25pt2 = i.book1D("JM25pt2", "pt", 40, 0, 200.);    
    _JM25pt3 = i.book1D("JM25pt3", "pt", 40, 0, 200.);    
    _JM25pt4 = i.book1D("JM25pt4", "pt", 40, 0, 200.);    

    _JM80njets = i.book1D("JM80njets", "n jets", 15, 0, 15.);    
    _JM80ht = i.book1D("JM80ht", "HT", 80, 300, 1100.);    
    _JM80pt1 = i.book1D("JM80pt1", "pt", 40, 60, 260.);    
    _JM80pt2 = i.book1D("JM80pt2", "pt", 40, 60, 260.);    
    _JM80pt3 = i.book1D("JM80pt3", "pt", 40, 60, 260.);    
    _JM80pt4 = i.book1D("JM80pt4", "pt", 40, 60, 260.);    


    // differential jet rates
    djr10 = i.book1D("djr10", "Differential Jet Rate 1#rightarrow0", 60, -1., 5.);
    djr21 = i.book1D("djr21", "Differential Jet Rate 2#rightarrow1", 60, -1., 5.);
    djr32 = i.book1D("djr32", "Differential Jet Rate 3#rightarrow2", 60, -1., 5.);
    djr43 = i.book1D("djr43", "Differential Jet Rate 4#rightarrow3", 60, -1., 5.);

    // sumET analysis
    _sumEt = i.book1D("sumET", "Sum of stable particles Et", 150, 0, 600.);
    _sumEt1 = i.book1D("sumET1", "Sum of stable particles Et (eta<0.5)", 150, 0, 200.);
    _sumEt2 = i.book1D("sumET2", "Sum of stable particles Et (0.5<eta<1.0)", 150, 0, 200.);
    _sumEt3 = i.book1D("sumET3", "Sum of stable particles Et (1.0<eta<1.5)", 150, 0, 200.);
    _sumEt4 = i.book1D("sumET4", "Sum of stable particles Et (1.5<eta<2.0)", 150, 0, 200.);
    _sumEt5 = i.book1D("sumET5", "Sum of stable particles Et (2.0<eta<5.0)", 150, 0, 200.);
    

  return;
}

void MBUEandQCDValidation::analyze(const edm::Event& iEvent,const edm::EventSetup& iSetup)
{ 

  ///Gathering the HepMCProduct information
  edm::Handle<HepMCProduct> evt;
  iEvent.getByToken(hepmcCollectionToken_, evt);

  //Get HepMC EVENT
  HepMC::GenEvent *myGenEvent = new HepMC::GenEvent(*(evt->GetEvent()));

  double weight = wmanager_.weight(iEvent);


  if ( verbosity_ > 0 ) { myGenEvent->print(); }

  double binW = 1.;
  
  hepmcGPCollection.clear();
  hepmcCharge.clear();
  for (unsigned int i = 0; i < eneInCell.size(); i++) { eneInCell[i] = 0.; }

  nEvt->Fill(0.5,weight);
  
  //Looping through HepMC::GenParticle collection to search for status 1 particles
  double charge = 0.;
  unsigned int nb = 0;
  unsigned int nc = 0;
  for (HepMC::GenEvent::particle_const_iterator iter = myGenEvent->particles_begin(); iter != myGenEvent->particles_end(); ++iter){
    if ( std::fabs((*iter)->pdg_id()) == 4 ) { nc++; }
    if ( std::fabs((*iter)->pdg_id()) == 5 ) { nb++; }
    if ( (*iter)->status() == 1) {
      hepmcGPCollection.push_back(*iter);
      const HepPDT::ParticleData* PData = fPDGTable->particle(HepPDT::ParticleID((*iter)->pdg_id()));
      if(PData==0) { charge = -999.; }
      else
        charge = PData->charge();

      hepmcCharge.push_back(charge);
      
      if ( verbosity_ > 0 ) {
        std::cout << "HepMC " << std::setw(14) << std::fixed << (*iter)->barcode() 
                  << std::setw(14) << std::fixed << (*iter)->pdg_id() 
                  << std::setw(14) << std::fixed << (*iter)->momentum().perp() 
                  << std::setw(14) << std::fixed << (*iter)->momentum().eta() 
                  << std::setw(14) << std::fixed << (*iter)->momentum().phi() << std::endl;
      }

    }
  }
  
  int nBSCp = 0; int nBSCm = 0; double eneHFp = 0.; double eneHFm = 0.; int nChapt05 = 0; int nChaVtx = 0;
  for (unsigned int i = 0; i < hepmcGPCollection.size(); i++ ){
    if ( !isNeutrino(i) ) {

      // BSC trigger

      if ( hepmcGPCollection[i]->momentum().eta() > 3.23 && hepmcGPCollection[i]->momentum().eta() < 4.65 ) { nBSCp++; }
      if ( hepmcGPCollection[i]->momentum().eta() < -3.23 && hepmcGPCollection[i]->momentum().eta() > -4.65 ) { nBSCm++; }

      // number of charged particles in different selections

      if ( std::fabs(hepmcGPCollection[i]->momentum().eta()) < 2.5 && hepmcGPCollection[i]->momentum().perp() > 0.5 && isCharged(i) ) { nChapt05++; }
      if ( std::fabs(hepmcGPCollection[i]->momentum().eta()) < 2.5 && hepmcGPCollection[i]->momentum().perp() > 0.1 && isCharged(i) ) { nChaVtx++; }
      unsigned int theIndex = theCalo->getCellIndexFromAngle(hepmcGPCollection[i]->momentum().eta(),hepmcGPCollection[i]->momentum().phi());
      if ( theIndex < CaloCellManager::nCaloCell ) eneInCell[theIndex] += hepmcGPCollection[i]->momentum().rho();
    }
  }

  // Forward calorimeters energy

  for (unsigned int icell = CaloCellManager::nBarrelCell+CaloCellManager::nEndcapCell; icell < CaloCellManager::nCaloCell; icell++ ) {
    if ( theCalo->getCellFromIndex(icell)->getEtaMin() < 0. ) { eneHFm += eneInCell[icell]; }
    else { eneHFp += eneInCell[icell]; }
  }

  // QCD-09-010 selection
  bool sel1 = false;
  if ( (nBSCp > 0 || nBSCm > 0) && eneHFp >= 3. && eneHFm >= 3. ) { sel1 = true; }

  // QCD-10-001 selection
  bool sel2 = false;
  if ( (nBSCp >0 || nBSCm > 0) && nChaVtx >= 3 && nChapt05 > 1 ) { sel2 = true; }
  
  // no forward trigger selection
  bool sel3 = false;
  if ( nBSCp == 0 && nBSCm == 0  ) { sel3 = true; }
  
  // single arm forward trigger selection
  bool sel4 = false;
  if ( ( nBSCp>0 && nBSCm == 0 ) || ( nBSCm>0 && nBSCp == 0 ) ) { sel4 = true; }
  
  // BSC selection
  bool sel5 = false;
  if ( nBSCp > 0 && nBSCm > 0 ) { sel5 = true; }
  
  // basic JME-10-001, FWD-10-002 and Jet-Multiplicity selection
  bool sel6 = false;
  if ( sel5 && nChaVtx > 3 ) { sel6 = true; }

  // FWD-10-001 selection
  bool sel7 = false;
  if ( nChaVtx >= 3 && nBSCm > 0 && eneHFp < 8. ) { sel7 = true; }

  // Fill selection histograms
  if ( sel1 ) nEvt1->Fill(0.5,weight);
  if ( sel2 ) nEvt2->Fill(0.5,weight);
  if ( sel3 ) nNoFwdTrig->Fill(0.5,weight);
  if ( sel4 ) nSaFwdTrig->Fill(0.5,weight);
  if ( sel6 ) nHFflow->Fill(0.5,weight);
  if ( sel7 ) nHFSD->Fill(0.5,weight);
  
  if ( nb > 0 ) nbquark->Fill(0.5,weight);
  if ( nb > 0 && nc > 0 ) ncandbquark->Fill(0.5,weight);
  if ( nb == 0 && nc > 0 ) ncnobquark->Fill(0.5,weight);

  // track analyses 
  double ptMax = 0.;
  unsigned int iMax = 0;
  double ptot = 0.;
  unsigned int ppbar = 0; unsigned int nnbar = 0; unsigned int kpm = 0; unsigned int k0s = 0; unsigned int l0 = 0; unsigned int gamma = 0; 
  unsigned int xim = 0; unsigned int omega = 0;
  unsigned int ele = 0; unsigned int muo = 0;
  unsigned int eleMax = 0;
  unsigned int muoMax = 0;

  std::vector<double> hfMB (CaloCellManager::nForwardEta,0);
  std::vector<double> hfDJ (CaloCellManager::nForwardEta,0);

  for (unsigned int i = 0; i < hepmcGPCollection.size(); i++ ){
    double eta = hepmcGPCollection[i]->momentum().eta();
    double pt = hepmcGPCollection[i]->momentum().perp();
    int pdgId = hepmcGPCollection[i]->pdg_id();
    if ( isCharged(i) && std::fabs(eta) < 2.5 ) {
      if ( sel1 ) {
        // QCD-09-010 
        binW = dNchdpt1->getTH1()->GetBinWidth(1);
        dNchdpt1->Fill(pt,1./binW); // weight to account for the pt bin width
        binW = dNchdeta1->getTH1()->GetBinWidth(1);
        dNchdeta1->Fill(eta,1./binW); // weight to account for the eta bin width
      }
      // search for the leading track QCD-10-001
      if ( sel2 ) {
        if ( pt > ptMax ) { ptMax = pt; iMax = i; }
        ptot += pt;
        
        // identified charged particle
        if (std::abs(pdgId) == 2212) {
          ppbar++;
          pPPbar->Fill(std::log10(pt),weight);
        }
        else if (std::abs(pdgId) == 321) {
          kpm++;
          pKpm->Fill(std::log10(pt),weight);
        }
        else if (std::abs(pdgId) == 3312) {
          xim++;
          pXim->Fill(std::log10(pt),weight);
        }
        else if (std::abs(pdgId) == 3334) {
          omega++;
          pOmega->Fill(std::log10(pt),weight);
        }
        else if (std::abs(pdgId) == 11) {
          ele++;
          eleMax = i;
        }
        else if (std::abs(pdgId) == 13) {
          muo++;
          muoMax = i;
        }
      }
    }
    else if ( sel2 && isNeutral(i) && std::fabs(eta) < 2.5 ) {
      if (std::abs(pdgId) == 310) {
        k0s++;
        pK0s->Fill(std::log10(pt),weight);
      }
      else if (std::abs(pdgId) == 3122) {
        l0++;
        pL0->Fill(std::log10(pt),weight);
      }
    }
    else if ( sel2 && isNeutral(i) && std::fabs(eta) < 5.19 ) {
      if (std::abs(pdgId) == 2112) {
        nnbar++;
        pNNbar->Fill(std::log10(pt),weight);
      }
      else if (std::abs(pdgId) == 22) {
        gamma++;
        pGamma->Fill(std::log10(pt),weight);
      }
    }
    unsigned int iBin = getHFbin(eta);
    if ( sel6 && !isNeutrino(i) &&  iBin < CaloCellManager::nForwardEta ) {
      hfMB[iBin] += hepmcGPCollection[i]->momentum().rho();
    }
  }
  nPPbar->Fill(ppbar,weight);
  nNNbar->Fill(nnbar,weight);
  nKpm->Fill(kpm,weight);
  nK0s->Fill(k0s,weight);
  nL0->Fill(l0,weight);
  nXim->Fill(xim,weight);
  nOmega->Fill(omega,weight);
  nGamma->Fill(gamma,weight);

  if ( ele > 0 ) elePt->Fill(std::log10(hepmcGPCollection[eleMax]->momentum().perp()),weight);
  if ( muo > 0 ) muoPt->Fill(std::log10(hepmcGPCollection[muoMax]->momentum().perp()),weight);

  leadTrackpt->Fill(hepmcGPCollection[iMax]->momentum().perp(),weight); 
  leadTracketa->Fill(hepmcGPCollection[iMax]->momentum().eta(),weight); 

  std::vector<double> theEtaRanges(theCalo->getEtaRanges());

  for (unsigned int i = 0; i < CaloCellManager::nForwardEta; i++ ) {
    binW = theEtaRanges[CaloCellManager::nBarrelEta+CaloCellManager::nEndcapEta+i+1]-theEtaRanges[CaloCellManager::nBarrelEta+CaloCellManager::nEndcapEta+i];
    dEdetaHFmb->Fill(i+0.5,hfMB[i]/binW);
  }

  // FWD-10-001

  if ( sel7 ) {

    double empz = 0.;
    unsigned int nCellOvTh = 0;
    double threshold = 0.;

    for (unsigned int icell = 0; icell < eneInCell.size(); icell++ ) {

      if ( theCalo->getCellFromIndex(icell)->getSubSys() != CaloCellId::Forward ) { threshold = 3.; }
      else { threshold = 4.; }

      if ( eneInCell[icell] > threshold ) {
        if ( theCalo->getCellFromIndex(icell)->getSubSys() == CaloCellId::Forward ) { nCellOvTh++; } 
        empz += eneInCell[icell]*(1.-std::cos(theCalo->getCellFromIndex(icell)->getThetaCell()));
      }

    }
    
    EmpzHFm->Fill(empz,weight);
    ntHFm->Fill(nCellOvTh,weight);
    eneHFmSel->Fill(eneHFm,weight);

  }
  
  // QCD-10-001
  double phiMax = hepmcGPCollection[iMax]->momentum().phi();
  std::vector<unsigned int> nchvsphi (nphiBin,0);
  std::vector<double> sptvsphi (nphiBin,0.);
  unsigned int nChaTra = 0;
  double sptTra = 0.;
  
  double binPhiW = 360./nphiBin;
  if ( sel2 ) {
    for (unsigned int i = 0; i < hepmcGPCollection.size(); i++ ){
      if ( isCharged(i) && std::fabs(hepmcGPCollection[i]->momentum().eta()) < 2. ) {
        double thePhi = (hepmcGPCollection[i]->momentum().phi()-phiMax)/CLHEP::degree;
        if ( thePhi < -180. ) { thePhi += 360.; }
        else if ( thePhi > 180. ) { thePhi -= 360.; }
        unsigned int thePhiBin = (int)((thePhi+180.)/binPhiW);
        if ( thePhiBin == nphiBin ) { thePhiBin -= 1; }
        nchvsphi[thePhiBin]++;
        sptvsphi[thePhiBin] += hepmcGPCollection[i]->momentum().perp();
        // analysis in the transverse region
        if ( std::fabs(thePhi) > 60. && std::fabs(thePhi) < 120. ) {
          nChaTra++;
          sptTra += hepmcGPCollection[i]->momentum().perp();
          binW = dNchdpt2->getTH1()->GetBinWidth(1);
          dNchdpt2->Fill(hepmcGPCollection[i]->momentum().perp(),1./binW); // weight to account for the pt bin width
          binW = dNchdeta2->getTH1()->GetBinWidth(1);
          dNchdeta2->Fill(hepmcGPCollection[i]->momentum().eta(),1./binW);  // weight to account for the eta bin width
        }
      }
    }
    nCha->Fill(nChaTra,weight);
    binW = dNchdSpt->getTH1()->GetBinWidth(1);
    dNchdSpt->Fill(sptTra,1.);
    //how do one apply weights to a profile? MonitorElement doesn't allow to 
    nChaDenLpt->Fill(hepmcGPCollection[iMax]->momentum().perp(),nChaTra/4./CLHEP::twopi);
    sptDenLpt->Fill(hepmcGPCollection[iMax]->momentum().perp(),sptTra/4./CLHEP::twopi);
    for ( unsigned int i = 0; i < nphiBin; i++ ) {
      double thisPhi = -180.+(i+0.5)*binPhiW;
      dNchdphi->Fill(thisPhi,nchvsphi[i]/binPhiW/4.); // density in phi and eta
      dSptdphi->Fill(thisPhi,sptvsphi[i]/binPhiW/4.); // density in phi and eta
    }
  }
  
  // Gather information in the charged GenJet collection
  edm::Handle<reco::GenJetCollection> genChJets;
  iEvent.getByToken(genchjetCollectionToken_, genChJets );
  
  unsigned int nJets = 0;
  double pt1 = 0.; 
  double pt2 = 0.;
  reco::GenJetCollection::const_iterator ij1 = genChJets->begin();
  reco::GenJetCollection::const_iterator ij2 = genChJets->begin();
  if ( sel2 ) {
    for (reco::GenJetCollection::const_iterator iter=genChJets->begin();iter!=genChJets->end();++iter){
      double eta = (*iter).eta();
      double pt = (*iter).pt();
      if ( verbosity_ > 0 ) { 
        std::cout << "GenJet " << std::setw(14) << std::fixed << (*iter).pt() 
                  << std::setw(14) << std::fixed << (*iter).eta() 
                  << std::setw(14) << std::fixed << (*iter).phi() << std::endl;
      }
      if ( std::fabs(eta) < 2. ) {
        nJets++;
        binW = dNchjdeta->getTH1()->GetBinWidth(1);
        dNchjdeta->Fill(eta,1./binW);
        binW = dNchjdpt->getTH1()->GetBinWidth(1);
        dNchjdpt->Fill(pt,1./binW);
        if ( pt >= pt1 ) { pt1 = pt; ij1 = iter; }
        if ( pt < pt1 && pt >= pt2 ) { pt2 = pt; ij2 = iter; }
      }
    }
    
    nChj->Fill(nJets,weight);
    if ( nJets > 0 && ij1 != genChJets->end() ) {
      leadChjpt->Fill(pt1,weight);
      leadChjeta->Fill((*ij1).eta(),weight);
      if ( nJets > 1 && ij2 != genChJets->end() ) {
        pt1pt2optotch->Fill(pt1+pt2,(pt1+pt2)/ptot);
      }
    }
  }
    
  
  // Gather information in the GenJet collection
  edm::Handle<reco::GenJetCollection> genJets;
  iEvent.getByToken(genjetCollectionToken_, genJets );
  
  nJets = 0;
  pt1 = 0.; 
  pt2 = 0.;
  double pt3 = 0.;


  // needed for Jet-Multiplicity Analysis
  int jm25njets  = 0;
  double jm25HT  = 0.;
  double jm25pt1 = 0.;
  double jm25pt2 = 0.;
  double jm25pt3 = 0.;
  double jm25pt4 = 0.;

  int jm80njets  = 0;
  double jm80HT  = 0.;
  double jm80pt1 = 0.;
  double jm80pt2 = 0.;
  double jm80pt3 = 0.;
  double jm80pt4 = 0.;



  reco::GenJetCollection::const_iterator ij3 = genJets->begin();
  if ( sel6 ) {
    for (reco::GenJetCollection::const_iterator iter=genJets->begin();iter!=genJets->end();++iter){
      double eta = (*iter).eta();
      double pt = (*iter).pt();
      if ( verbosity_ > 0 ) {
        std::cout << "GenJet " << std::setw(14) << std::fixed << (*iter).pt() 
                  << std::setw(14) << std::fixed << (*iter).eta() 
                  << std::setw(14) << std::fixed << (*iter).phi() << std::endl;
      }
      if ( std::fabs(eta) < 5. ) {
        nJets++;
        if ( pt >= pt1 ) { pt1 = pt; ij1 = iter; }
        if ( pt < pt1 && pt >= pt2 ) { pt2 = pt; ij2 = iter; }
        if ( pt < pt2 && pt >= pt3 ) { pt3 = pt; ij3 = iter; }
      }

      // find variables for Jet-Multiplicity Analysis
      if(fabs(iter->eta()) < 3. && iter->pt()>25.) {
        jm25njets++;
        jm25HT += iter->pt();
        if(iter->pt()>jm25pt1) {
          jm25pt4 = jm25pt3;
          jm25pt3 = jm25pt2;
          jm25pt2 = jm25pt1;
          jm25pt1 = iter->pt();
        } else if(iter->pt()>jm25pt2) {
          jm25pt4 = jm25pt3;
          jm25pt3 = jm25pt2;
          jm25pt2 = iter->pt();
        } else if(iter->pt()>jm25pt3) {
          jm25pt4 = jm25pt3;
          jm25pt3 = iter->pt();
        } else if(iter->pt()>jm25pt4) {
          jm25pt4 = iter->pt();
        }
        // even harder jets...
        if(iter->pt()>80.) {
          jm80njets++;
          jm80HT += iter->pt();
          if(iter->pt()>jm80pt1) {
            jm80pt4 = jm80pt3;
            jm80pt3 = jm80pt2;
            jm80pt2 = jm80pt1;
            jm80pt1 = iter->pt();
          } else if(iter->pt()>jm80pt2) {
            jm80pt4 = jm80pt3;
            jm80pt3 = jm80pt2;
            jm80pt2 = iter->pt();
          } else if(iter->pt()>jm80pt3) {
            jm80pt4 = jm80pt3;
            jm80pt3 = iter->pt();
          } else if(iter->pt()>jm80pt4) {
            jm80pt4 = iter->pt();
          }
        }
	
      }

      if(jm25njets>3) {
        _JM25njets ->Fill(jm25njets,weight);
        _JM25ht    ->Fill(jm25HT,weight);
        _JM25pt1   ->Fill(jm25pt1,weight);
        _JM25pt2   ->Fill(jm25pt2,weight);
        _JM25pt3   ->Fill(jm25pt3,weight);
        _JM25pt4   ->Fill(jm25pt4,weight);
      }
      if(jm80njets>3) {
        _JM80njets ->Fill(jm80njets,weight);
        _JM80ht    ->Fill(jm80HT,weight);
        _JM80pt1   ->Fill(jm80pt1,weight);
        _JM80pt2   ->Fill(jm80pt2,weight);
        _JM80pt3   ->Fill(jm80pt3,weight);
        _JM80pt4   ->Fill(jm80pt4,weight);
      }
    }
    
    // select a di-jet event JME-10-001 variant
    double sumJetEt = 0; double sumPartPt = 0.; double sumChPartPt = 0.;
    double jpx = 0; double jpy = 0;
    if ( nJets >= 2 && ij1 != genJets->end() && ij2 != genJets->end() ) {
      if ( (*ij1).pt() > 25. && (*ij1).pt() > 25. ) {
        double deltaPhi = std::fabs((*ij1).phi()-(*ij2).phi())/CLHEP::degree;
        if ( deltaPhi > 180. ) deltaPhi = 360.-deltaPhi;
        pt1pt2Dphi->Fill(deltaPhi,weight);
        if ( std::fabs(deltaPhi) > 2.5*CLHEP::degree ) {

          nDijet->Fill(0.5,weight);

          for (unsigned int i = 0; i < hepmcGPCollection.size(); i++ ){
            double eta = hepmcGPCollection[i]->momentum().eta();
            unsigned int iBin = getHFbin(eta);
            if ( !isNeutrino(i) &&  iBin < CaloCellManager::nForwardEta ) {
              hfDJ[iBin] += hepmcGPCollection[i]->momentum().rho();
            }
            if ( !isNeutrino(i) && std::fabs(eta) < 5. ) {
              sumPartPt += hepmcGPCollection[i]->momentum().perp();
              if ( isCharged(i) ) {
                sumChPartPt += hepmcGPCollection[i]->momentum().perp();
              }
            }
          }
          for (unsigned int i = 0; i < CaloCellManager::nForwardEta; i++ ) {
            binW = theEtaRanges[CaloCellManager::nBarrelEta+CaloCellManager::nEndcapEta+i+1]-theEtaRanges[CaloCellManager::nBarrelEta+CaloCellManager::nEndcapEta+i];
            dEdetaHFdj->Fill(i+0.5,hfDJ[i]/binW);
          }

          double invMass = (*ij1).energy()*(*ij2).energy()-(*ij1).px()*(*ij2).px()-(*ij1).py()*(*ij2).py()-(*ij1).pz()*(*ij2).pz();
          invMass = std::sqrt(invMass);
          pt1pt2InvM->Fill(invMass,weight);

          sumPt->Fill(sumPartPt,weight);
          sumChPt->Fill(sumChPartPt,weight);

          unsigned int nSelJets = 0;
          for (reco::GenJetCollection::const_iterator iter=genJets->begin();iter!=genJets->end();++iter){
            double pt = (*iter).pt();
            double eta = (*iter).eta();
            if ( std::fabs(eta) < 5. ) { 
              nSelJets++; 
              binW = dNjdeta->getTH1()->GetBinWidth(1);
              dNjdeta->Fill(eta,1./binW*weight);
              binW = dNjdpt->getTH1()->GetBinWidth(1);
              dNjdpt->Fill(pt,1./binW*weight);
              sumJetEt += (*iter).pt();
              jpx += (*iter).px();
              jpy += (*iter).py();
            }
          }

          nj->Fill(nSelJets,weight);
          double mEt = std::sqrt(jpx*jpx+jpy*jpy);
          sumJEt->Fill(sumJetEt,weight);
          missEtosumJEt->Fill(mEt/sumJetEt,weight);

          if ( nSelJets >= 3 ) { pt3Frac->Fill((*ij3).pt()/(pt1+pt2),weight); }

          pt1pt2optot->Fill(pt1+pt2,(pt1+pt2)/sumJetEt);
          pt1pt2balance->Fill((pt1-pt2)/(pt1+pt2),weight);
        }
      }      
    }
  }
    
 
  //compute differential jet rates
  std::vector<const HepMC::GenParticle*> qcdActivity;
  HepMCValidationHelper::removeIsolatedLeptons(myGenEvent, 0.2, 3., qcdActivity);
  //HepMCValidationHelper::allStatus1(myGenEvent, qcdActivity);
  //fill PseudoJets to use fastjet
  std::vector<fastjet::PseudoJet> vecs;
  int counterUser = 1;
  std::vector<const HepMC::GenParticle*>::const_iterator iqcdact;
  for (iqcdact = qcdActivity.begin(); iqcdact != qcdActivity.end(); ++iqcdact){
    const HepMC::FourVector& fmom = (*iqcdact)->momentum(); 
    fastjet::PseudoJet pseudoJet(fmom.px(), fmom.py(), fmom.pz(), fmom.e());
    pseudoJet.set_user_index(counterUser);
    vecs.push_back(pseudoJet);
    ++counterUser;
  }
  //compute jets
  fastjet::ClusterSequence cseq(vecs, fastjet::JetDefinition(fastjet::kt_algorithm, 1., fastjet::E_scheme)); 
  //access the cluster sequence and get the relevant info
  djr10->Fill(std::log10(sqrt(cseq.exclusive_dmerge(0))),weight);
  djr21->Fill(std::log10(sqrt(cseq.exclusive_dmerge(1))),weight);
  djr32->Fill(std::log10(sqrt(cseq.exclusive_dmerge(2))),weight);
  djr43->Fill(std::log10(sqrt(cseq.exclusive_dmerge(3))),weight);
  

  // compute sumEt for all stable particles
  std::vector<const HepMC::GenParticle*> allStable;
  HepMCValidationHelper::allStatus1(myGenEvent, allStable);
  
  double sumEt  = 0.;
  double sumEt1 = 0.;
  double sumEt2 = 0.;
  double sumEt3 = 0.;
  double sumEt4 = 0.;
  double sumEt5 = 0.;

  for(std::vector<const HepMC::GenParticle*>::const_iterator iter=allStable.begin();
      iter != allStable.end(); ++iter) {

    double thisEta=fabs((*iter)->momentum().eta());
    
    if(thisEta < 5.) {
      const HepMC::FourVector mom=(*iter)->momentum();
      double px=mom.px();
      double py=mom.py();
      double pz=mom.pz();
      double E=mom.e();
      double thisSumEt = (
                          sqrt(px*px + py*py)*E /
                          sqrt(px*px + py*py + pz*pz)
                          );
      sumEt += thisSumEt;
      if(thisEta<1.0) sumEt1 += thisSumEt;
      else if(thisEta<2.0) sumEt2 += thisSumEt;
      else if(thisEta<3.0) sumEt3 += thisSumEt;
      else if(thisEta<4.0) sumEt4 += thisSumEt;
      else sumEt5 += thisSumEt;
      
    }
  }
  
  if(sumEt>0.)
    _sumEt->Fill(sumEt,weight);
  if(sumEt1>0.)
    _sumEt1->Fill(sumEt1,weight);
  if(sumEt2>0.)
    _sumEt2->Fill(sumEt2,weight);
  if(sumEt3>0.)
    _sumEt3->Fill(sumEt3,weight);
  if(sumEt4>0.)
    _sumEt4->Fill(sumEt4,weight);
  if(sumEt5>0.)
    _sumEt5->Fill(sumEt5,weight);
  
  delete myGenEvent;
}//analyze

bool MBUEandQCDValidation::isCharged(unsigned int i){
  
  bool status = false;
  if ( hepmcGPCollection.size() < i+1 ) { return status; }
  else { status = (hepmcCharge[i] != 0. && hepmcCharge[i] != -999.); }
  return status;

}

bool MBUEandQCDValidation::isNeutral(unsigned int i){
  
  bool status = false;
  int pdgId = std::abs(hepmcGPCollection[i]->pdg_id());
  if ( hepmcGPCollection.size() < i+1 ) { return status; }
  else { status = (hepmcCharge[i] == 0. && pdgId != 12 && pdgId != 14 && pdgId != 16) ; }
  return status;

}

bool MBUEandQCDValidation::isNeutrino(unsigned int i){
  
  bool status = false;
  int pdgId = std::abs(hepmcGPCollection[i]->pdg_id());
  if ( hepmcGPCollection.size() < i+1 ) { return status; }
  else { status = (pdgId == 12 || pdgId == 14 || pdgId == 16) ; }
  return status;

}

unsigned int MBUEandQCDValidation::getHFbin(double eta) {

  unsigned int iBin = 999;

  std::vector<double> theEtaRanges(theCalo->getEtaRanges());

  for (unsigned int i = CaloCellManager::nBarrelEta+CaloCellManager::nEndcapEta; 
       i < CaloCellManager::nBarrelEta+CaloCellManager::nEndcapEta+CaloCellManager::nForwardEta; i++ ){
    if ( std::fabs(eta) >= theEtaRanges[i] && std::fabs(eta) < theEtaRanges[i+1] ) 
      { iBin = i-CaloCellManager::nBarrelEta-CaloCellManager::nEndcapEta; }
  }

  return iBin;

}
