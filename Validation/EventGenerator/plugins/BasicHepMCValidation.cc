/*class BasicHepMCValidation
 *  
 *  Class to fill dqm monitor elements from existing EDM file
 *
 */
 
#include "Validation/EventGenerator/interface/BasicHepMCValidation.h"

#include "CLHEP/Units/defs.h"
#include "CLHEP/Units/PhysicalConstants.h"
#include "Validation/EventGenerator/interface/DQMHelper.h"
using namespace edm;

BasicHepMCValidation::BasicHepMCValidation(const edm::ParameterSet& iPSet): 
  wmanager_(iPSet,consumesCollector()),
  hepmcCollection_(iPSet.getParameter<edm::InputTag>("hepmcCollection"))
{    
  hepmcCollectionToken_=consumes<HepMCProduct>(hepmcCollection_);
}

BasicHepMCValidation::~BasicHepMCValidation() {}

void BasicHepMCValidation::dqmBeginRun(const edm::Run& r, const edm::EventSetup& c) {
  c.getData( fPDGTable );
}

void BasicHepMCValidation::bookHistograms(DQMStore::IBooker &i, edm::Run const &, edm::EventSetup const &){

	///Setting the DQM top directories
	DQMHelper dqm(&i); i.setCurrentFolder("Generator/Particles");
    
    // Number of analyzed events
    nEvt = dqm.book1dHisto("nEvt", "n analyzed Events", 1, 0., 1.);
	
	///Booking the ME's
	///multiplicity
	uNumber = dqm.book1dHisto("uNumber", "No. u", 20, 0, 20);
	dNumber = dqm.book1dHisto("dNumber", "No. d", 20, 0, 20);
	sNumber = dqm.book1dHisto("sNumber", "No. s", 20, 0, 20);
    cNumber = dqm.book1dHisto("cNumber", "No. c", 20, 0, 20);
	bNumber = dqm.book1dHisto("bNumber", "No. b", 20, 0, 20);
	tNumber = dqm.book1dHisto("tNumber", "No. t", 20, 0, 20);
	//
	ubarNumber = dqm.book1dHisto("ubarNumber", "No. ubar", 20, 0, 20);
	dbarNumber = dqm.book1dHisto("dbarNumber", "No. dbar", 20, 0, 20);
	sbarNumber = dqm.book1dHisto("sbarNumber", "No. sbar", 20, 0, 20);
        cbarNumber = dqm.book1dHisto("cbarNumber", "No. cbar", 20, 0, 20);
	bbarNumber = dqm.book1dHisto("bbarNumber", "No. bbar", 20, 0, 20);
	tbarNumber = dqm.book1dHisto("tbarNumber", "No. tbar", 20, 0, 20);
	//
	eminusNumber = dqm.book1dHisto("eminusNumber", "No. e-", 20, 0, 20);
	nueNumber = dqm.book1dHisto("nueNumber", "No. nu_e", 20, 0, 20);
	muminusNumber = dqm.book1dHisto("muminusNumber", "No. mu-", 20, 0, 20);
	numuNumber = dqm.book1dHisto("numuNumber", "No. nu_mu", 20, 0, 20);
	tauminusNumber = dqm.book1dHisto("tauminusNumber", "No. tau-", 20, 0, 20);
	nutauNumber = dqm.book1dHisto("nutauNumber", "No. nu_tau", 20, 0, 20);
	//
	eplusNumber = dqm.book1dHisto("eplusNumber", "No. e+", 20, 0, 20);
	nuebarNumber = dqm.book1dHisto("nuebarNumber", "No. nu_e_bar", 20, 0, 20);
	muplusNumber = dqm.book1dHisto("muplusNumber", "No. mu+", 20, 0, 20);
	numubarNumber = dqm.book1dHisto("numubarNumber", "No. nu_mu_bar", 20, 0, 20);
	tauplusNumber = dqm.book1dHisto("tauplusNumber", "No. tau+", 20, 0, 20);
	nutaubarNumber = dqm.book1dHisto("nutaubarNumber", "No. nu_tau_bar", 20, 0, 20);
	//
	WplusNumber = dqm.book1dHisto("WplusNumber", "No. W+", 20, 0, 20);
	WminusNumber = dqm.book1dHisto("WminusNumber", "No. W-", 20, 0, 20);
	ZNumber = dqm.book1dHisto("ZNumber", "No. Z", 20, 0, 20);
	gammaNumber = dqm.book1dHisto("gammaNumber", "Log10(No. gamma)", 60, -1, 5); //Log
	gluNumber = dqm.book1dHisto("gluonNumber", "Log10(No. gluons)", 60, -1, 5); //Log
	//
	piplusNumber = dqm.book1dHisto("piplusNumber", "Log10(No. pi+)", 60, -1, 5); //Log
	piminusNumber = dqm.book1dHisto("piminusNumber", "Log10(No. pi-)", 60, -1, 5); //Log
	pizeroNumber = dqm.book1dHisto("pizeroNumber", "Log10(No. pi_0)", 60, -1, 5); //Log
	KplusNumber = dqm.book1dHisto("KplusNumber", "No. K+", 100, 0, 100);
	KminusNumber = dqm.book1dHisto("KminusNumber", "No. K-", 100, 0, 100);
	KlzeroNumber = dqm.book1dHisto("KlzeroNumber", "No. K_l^0", 100, 0, 100);
	KszeroNumber = dqm.book1dHisto("KszeroNumber", "No. K_s^0", 100, 0, 100);
	//
	pNumber = dqm.book1dHisto("pNumber", "No. p", 100, 0, 100);
	pbarNumber = dqm.book1dHisto("pbarNumber", "No. pbar", 100, 0, 100);
	nNumber = dqm.book1dHisto("nNumber", "No. n", 100, 0, 100);
	nbarNumber = dqm.book1dHisto("nbarNumber", "No. nbar", 100, 0, 100);
	l0Number = dqm.book1dHisto("l0Number", "No. Lambda0", 100, 0, 100);
	l0barNumber = dqm.book1dHisto("l0barNumber", "No. Lambda0bar", 100, 0, 100);
	//
	DplusNumber = dqm.book1dHisto("DplusNumber", "No. D+", 20, 0, 20);
	DminusNumber = dqm.book1dHisto("DminusNumber", "No. D-", 20, 0, 20);
	DzeroNumber = dqm.book1dHisto("DzeroNumber", "No. D^0", 20, 0, 20);
	//
	BplusNumber = dqm.book1dHisto("BplusNumber", "No. B+", 20, 0, 20);
	BminusNumber = dqm.book1dHisto("BminusNumber", "No. B-", 20, 0, 20);
	BzeroNumber = dqm.book1dHisto("BzeroNumber", "No. B^0", 20, 0, 20);
	BszeroNumber = dqm.book1dHisto("BszeroNumber", "No. B^0_s", 20, 0, 20);
	//
	otherPtclNumber = dqm.book1dHisto("otherPtclNumber", "Log10(No. other ptcls)", 60, -1, 5); //Log

	//Momentum 
	uMomentum = dqm.book1dHisto("uMomentum", "Log10(p) u", 60, -2, 4);
	dMomentum = dqm.book1dHisto("dMomentum", "Log10(p) d", 60, -2, 4);
	sMomentum = dqm.book1dHisto("sMomentum", "Log10(p) s", 60, -2, 4);
    cMomentum = dqm.book1dHisto("cMomentum", "Log10(p) c", 60, -2, 4);
	bMomentum = dqm.book1dHisto("bMomentum", "Log10(p) b", 60, -2, 4);
	tMomentum = dqm.book1dHisto("tMomentum", "Log10(p) t", 60, -2, 4);
	//
	ubarMomentum = dqm.book1dHisto("ubarMomentum", "Log10(p) ubar", 60, -2, 4);
	dbarMomentum = dqm.book1dHisto("dbarMomentum", "Log10(p) dbar", 60, -2, 4);
	sbarMomentum = dqm.book1dHisto("sbarMomentum", "Log10(p) sbar", 60, -2, 4);
    cbarMomentum = dqm.book1dHisto("cbarMomentum", "Log10(p) cbar", 60, -2, 4);
	bbarMomentum = dqm.book1dHisto("bbarMomentum", "Log10(p) bbar", 60, -2, 4);
	tbarMomentum = dqm.book1dHisto("tbarMomentum", "Log10(p) tbar", 60, -2, 4);
	//
	eminusMomentum = dqm.book1dHisto("eminusMomentum", "Log10(p) e-", 60, -2, 4);
	nueMomentum = dqm.book1dHisto("nueMomentum", "Log10(Momentum) nue", 60, -2, 4);
	muminusMomentum = dqm.book1dHisto("muminusMomentum", "Log10(p) mu-", 60, -2, 4);
	numuMomentum = dqm.book1dHisto("numuMomentum", "Log10(p) numu", 60, -2, 4);
	tauminusMomentum = dqm.book1dHisto("tauminusMomentum", "Log10(p) tau-", 60, -2, 4);
	nutauMomentum = dqm.book1dHisto("nutauMomentum", "Log10(p) nutau", 60, -2, 4);
	//
	eplusMomentum = dqm.book1dHisto("eplusMomentum", "Log10(p) e+", 60, -2, 4);
	nuebarMomentum = dqm.book1dHisto("nuebarMomentum", "Log10(p) nuebar", 60, -2, 4);
	muplusMomentum = dqm.book1dHisto("muplusMomentum", "Log10(p) mu+", 60, -2, 4);
	numubarMomentum = dqm.book1dHisto("numubarMomentum", "Log10(p) numubar", 60, -2, 4);
	tauplusMomentum = dqm.book1dHisto("tauplusMomentum", "Log10(p) tau+", 60, -2, 4);
	nutaubarMomentum = dqm.book1dHisto("nutaubarMomentum", "Log10(p) nutaubar", 60, -2, 4);
	//
	gluMomentum = dqm.book1dHisto("gluonMomentum", "Log10(p) gluons", 70, -3, 4);
	WplusMomentum = dqm.book1dHisto("WplusMomentum", "Log10(p) W+", 60, -2, 4);
	WminusMomentum = dqm.book1dHisto("WminusMomentum", "Log10(p) W-", 60, -2, 4);
	ZMomentum = dqm.book1dHisto("ZMomentum", "Log10(p) Z", 60, -2, 4);
	gammaMomentum = dqm.book1dHisto("gammaMomentum", "Log10(p) gamma", 70, -3, 4);
	//
	piplusMomentum = dqm.book1dHisto("piplusMomentum", "Log10(p) pi+", 60, -2, 4);
	piminusMomentum = dqm.book1dHisto("piminusMomentum", "Log10(p) pi-", 60, -2, 4);
	pizeroMomentum = dqm.book1dHisto("pizeroMomentum", "Log10(p) pi_0", 60, -2, 4);
	KplusMomentum = dqm.book1dHisto("KplusMomentum", "Log10(p) K+", 60, -2, 4);
	KminusMomentum = dqm.book1dHisto("KminusMomentum", "Log10(p) K-", 60, -2, 4);
	KlzeroMomentum = dqm.book1dHisto("KlzeroMomentum", "Log10(p) K_l^0", 60, -2, 4);
	KszeroMomentum = dqm.book1dHisto("KszeroMomentum", "Log10(p) K_s^0", 60, -2, 4);
	//
	pMomentum = dqm.book1dHisto("pMomentum", "Log10(p) p", 60, -2, 4);
	pbarMomentum = dqm.book1dHisto("pbarMomentum", "Log10(p) pbar", 60, -2, 4);
	nMomentum = dqm.book1dHisto("nMomentum", "Log10(p) n", 60, -2, 4);
	nbarMomentum = dqm.book1dHisto("nbarMomentum", "Log10(p) nbar", 60, -2, 4);
	l0Momentum = dqm.book1dHisto("l0Momentum", "Log10(p) Lambda0", 60, -2, 4);
	l0barMomentum = dqm.book1dHisto("l0barMomentum", "Log10(p) Lambda0bar", 60, -2, 4);
	//
	DplusMomentum = dqm.book1dHisto("DplusMomentum", "Log10(p) D+", 60, -2, 4);
	DminusMomentum = dqm.book1dHisto("DminusMomentum", "Log10(p) D-", 60, -2, 4);
	DzeroMomentum = dqm.book1dHisto("DzeroMomentum", "Log10(p) D^0", 60, -2, 4);
	//
	BplusMomentum = dqm.book1dHisto("BplusMomentum", "Log10(p) B+", 60, -2, 4);
	BminusMomentum = dqm.book1dHisto("BminusMomentum", "Log10(p) B-", 60, -2, 4);
	BzeroMomentum = dqm.book1dHisto("BzeroMomentum", "Log10(p) B^0", 60, -2, 4);
	BszeroMomentum = dqm.book1dHisto("BszeroMomentum", "Log10(p) B^0_s", 60, -2, 4);
	//
	otherPtclMomentum = dqm.book1dHisto("otherPtclMomentum", "Log10(p) other ptcls", 60, -2, 4);

	///other
	genPtclNumber = dqm.book1dHisto("genPtclNumber", "Log10(No. all particles)", 60, -1, 5); //Log
	genVrtxNumber = dqm.book1dHisto("genVrtxNumber", "Log10(No. all vertexs)", 60, -1, 5); //Log
	//
	stablePtclNumber= dqm.book1dHisto("stablePtclNumber", "Log10(No. stable particles)", 50, 0, 5); //Log
	stablePtclPhi = dqm.book1dHisto("stablePtclPhi", "stable Ptcl Phi", 360, -180, 180);
	stablePtclEta = dqm.book1dHisto("stablePtclEta", "stable Ptcl Eta (pseudo rapidity)", 220, -11, 11);
	stablePtclCharge = dqm.book1dHisto("stablePtclCharge", "stablePtclCharge", 5, -2, 2);
	stableChaNumber= dqm.book1dHisto("stableChaNumber", "Log10(No. stable charged particles)", 50, 0, 5); //Log
	stablePtclp = dqm.book1dHisto("stablePtclp", "Log10(p) stable ptcl p", 80, -4, 4); //Log
	stablePtclpT = dqm.book1dHisto("stablePtclpT", "Log10(pT) stable ptcl pT", 80, -4, 4); //Log
        partonNumber = dqm.book1dHisto("partonNumber", "number of partons", 100, 0, 100);
	partonpT = dqm.book1dHisto("partonpT", "Log10(pT) parton pT", 80, -4, 4); //Log
	outVrtxStablePtclNumber = dqm.book1dHisto("outVrtxStablePtclNumber", "No. outgoing stable ptcls from vrtx", 10, 0, 10); 
	//
	outVrtxPtclNumber = dqm.book1dHisto("outVrtxPtclNumber", "No. outgoing ptcls from vrtx", 30, 0, 30);
	vrtxZ = dqm.book1dHisto("VrtxZ", "VrtxZ", 50 , -250, 250);
	vrtxRadius = dqm.book1dHisto("vrtxRadius", "vrtxRadius", 50, 0, 50);
	//
	unknownPDTNumber = dqm.book1dHisto("unknownPDTNumber", "Log10(No. unknown ptcls PDT)", 60, -1, 5); //Log
    genPtclStatus = dqm.book1dHisto("genPtclStatus", "Status of genParticle", 200,0,200.);
	//
    Bjorken_x = dqm.book1dHisto("Bjorken_x", "Bjorken_x", 1000, 0.0, 1.0);
    //
    status1ShortLived = dqm.book1dHisto("status1ShortLived","Status 1 short lived", 11, 0, 11);
    status1ShortLived->setBinLabel(1,"d/dbar");
    status1ShortLived->setBinLabel(2,"u/ubar");
    status1ShortLived->setBinLabel(3,"s/sbar");
    status1ShortLived->setBinLabel(4,"c/cbar");
    status1ShortLived->setBinLabel(5,"b/bbar");
    status1ShortLived->setBinLabel(6,"t/tbar");
    status1ShortLived->setBinLabel(7,"g");
    status1ShortLived->setBinLabel(8,"tau-/tau+");
    status1ShortLived->setBinLabel(9,"Z0");
    status1ShortLived->setBinLabel(10,"W-/W+");
    status1ShortLived->setBinLabel(11,"PDG = 7,8,17,25-99");

    DeltaEcms = dqm.book1dHisto("DeltaEcms1","deviation from nominal Ecms", 200,-1., 1.);
    DeltaPx = dqm.book1dHisto("DeltaPx1","deviation from nominal Px", 200,-1., 1.);
    DeltaPy = dqm.book1dHisto("DeltaPy1","deviation from nominal Py", 200,-1., 1.);
    DeltaPz = dqm.book1dHisto("DeltaPz1","deviation from nominal Pz", 200,-1., 1.);

  return;
}

void BasicHepMCValidation::analyze(const edm::Event& iEvent,const edm::EventSetup& iSetup)
{ 
  ///counters to zero for every event
  int uNum = 0; int dNum = 0; int sNum = 0; int cNum = 0; int bNum = 0; int tNum = 0;
  int ubarNum = 0; int dbarNum = 0; int sbarNum = 0; int cbarNum = 0; int bbarNum = 0; int tbarNum = 0;
  int partonNum = 0;
  //
  int eminusNum = 0; int nueNum = 0; int muminusNum = 0; int numuNum = 0; int tauminusNum = 0; int nutauNum = 0;
  int eplusNum = 0; int nuebarNum = 0; int muplusNum = 0; int numubarNum = 0; int tauplusNum = 0; int nutaubarNum = 0;
  //
  int gluNum = 0; int WplusNum = 0; int WminusNum = 0; int ZNum = 0; int gammaNum = 0;
  //
  int piplusNum = 0; int piminusNum = 0; int pizeroNum = 0; int KplusNum = 0; int KminusNum = 0; int KlzeroNum = 0; int KszeroNum = 0;  
  //
  int pNum = 0; int pbarNum = 0; int nNum = 0; int nbarNum = 0; int l0Num = 0; int l0barNum = 0;
  //
  int DplusNum = 0; int DminusNum = 0; int DzeroNum = 0; int BplusNum = 0; int BminusNum = 0; int BzeroNum = 0; int BszeroNum = 0;
  //
  int outVrtxStablePtclNum = 0; int stablePtclNum = 0; int otherPtclNum = 0; int unknownPDTNum = 0; int stableChaNum = 0;
  //
  double bjorken = 0.;
  //
  double etotal = 0. ; double pxtotal = 0.; double pytotal = 0.; double pztotal = 0.;

  ///Gathering the HepMCProduct information
  edm::Handle<HepMCProduct> evt;
  iEvent.getByToken(hepmcCollectionToken_, evt);

  //Get EVENT
  HepMC::GenEvent *myGenEvent = new HepMC::GenEvent(*(evt->GetEvent()));

  double weight = wmanager_.weight(iEvent);

  nEvt->Fill(0.5,weight);

  genPtclNumber->Fill(log10(myGenEvent->particles_size()),weight);     
  genVrtxNumber->Fill(log10(myGenEvent->vertices_size()),weight);

  ///Bjorken variable from PDF
  HepMC::PdfInfo *pdf = myGenEvent->pdf_info();    
  if(pdf){
    bjorken = ((pdf->x1())/((pdf->x1())+(pdf->x2())));
  }
  Bjorken_x->Fill(bjorken,weight);

  //Looping through the VERTICES in the event
  HepMC::GenEvent::vertex_const_iterator vrtxBegin = myGenEvent->vertices_begin();
  HepMC::GenEvent::vertex_const_iterator vrtxEnd = myGenEvent->vertices_end();
  for(HepMC::GenEvent::vertex_const_iterator vrtxIt = vrtxBegin; vrtxIt!=vrtxEnd; ++vrtxIt)
    {
      ///Vertices
      HepMC::GenVertex *vrtx = *vrtxIt;
      outVrtxPtclNumber->Fill(vrtx->particles_out_size(),weight); //std::cout << "all " << vrtx->particles_out_size() << '\n';
      vrtxZ->Fill(vrtx->point3d().z(),weight);
      vrtxRadius->Fill(vrtx->point3d().perp(),weight);
	
      ///loop on vertex particles
      HepMC::GenVertex::particles_out_const_iterator vrtxPtclBegin = vrtx->particles_out_const_begin();
      HepMC::GenVertex::particles_out_const_iterator vrtxPtclEnd = vrtx->particles_out_const_end();
      outVrtxStablePtclNum = 0;
      for(HepMC::GenVertex::particles_out_const_iterator vrtxPtclIt = vrtxPtclBegin; vrtxPtclIt != vrtxPtclEnd; ++vrtxPtclIt)
        {
          HepMC::GenParticle *vrtxPtcl = *vrtxPtclIt;
          if (vrtxPtcl->status() == 1){
            ++outVrtxStablePtclNum; //std::cout << "stable " << outVrtxStablePtclNum << '\n';
          }
        }
      outVrtxStablePtclNumber->Fill(outVrtxStablePtclNum,weight);
    }//vertices

    
  ///Looping through the PARTICLES in the event
  HepMC::GenEvent::particle_const_iterator ptclBegin = myGenEvent->particles_begin();
  HepMC::GenEvent::particle_const_iterator ptclEnd = myGenEvent->particles_end();
  for(HepMC::GenEvent::particle_const_iterator ptclIt = ptclBegin; ptclIt!=ptclEnd; ++ptclIt)
    {
    
      ///Particles
      HepMC::GenParticle *ptcl = *ptclIt;
      int Id = ptcl->pdg_id(); // std::cout << Id << '\n'; 
      float Log_p = log10( ptcl->momentum().rho() );
      double charge = 999.;	// for the charge it's needed a HepPDT method
      int status = ptcl->status();
      const HepPDT::ParticleData* PData = fPDGTable->particle(HepPDT::ParticleID(Id));
      if(PData==0) {
        //	    std::cout << "Unknown id = " << Id << '\n';
	    ++unknownPDTNum;
      }
      else
	    charge = PData->charge();

      ///Status statistics
      genPtclStatus->Fill((float)status,weight);

      ///Stable particles
      if(ptcl->status() == 1){
	    ++stablePtclNum;
	    stablePtclPhi->Fill(ptcl->momentum().phi()/CLHEP::degree,weight); //std::cout << ptcl->polarization().phi() << '\n';
	    stablePtclEta->Fill(ptcl->momentum().pseudoRapidity(),weight);
	    stablePtclCharge->Fill(charge,weight); // std::cout << ptclData.charge() << '\n';
	    stablePtclp->Fill(Log_p,weight);
	    stablePtclpT->Fill(log10(ptcl->momentum().perp()),weight);
        if (charge != 0. && charge != 999.) ++stableChaNum;
        if ( std::abs(Id) == 1 ) status1ShortLived->Fill(1,weight);
        if ( std::abs(Id) == 2 ) status1ShortLived->Fill(2,weight);
        if ( std::abs(Id) == 3 ) status1ShortLived->Fill(3,weight);
        if ( std::abs(Id) == 4 ) status1ShortLived->Fill(4,weight);
        if ( std::abs(Id) == 5 ) status1ShortLived->Fill(5,weight);
        if ( std::abs(Id) == 6 ) status1ShortLived->Fill(6,weight);
        if ( Id == 21 ) status1ShortLived->Fill(7,weight);
        if ( std::abs(Id) == 15 ) status1ShortLived->Fill(8,weight);
        if ( Id == 23 ) status1ShortLived->Fill(9,weight);
        if ( std::abs(Id) == 24 ) status1ShortLived->Fill(10,weight);
        if ( std::abs(Id) == 7 || std::abs(Id) == 8 || std::abs(Id) == 17 || (std::abs(Id) >= 25 && std::abs(Id) <= 99) ) status1ShortLived->Fill(11,weight);
        etotal += ptcl->momentum().e(); 
        pxtotal += ptcl->momentum().px(); 
        pytotal += ptcl->momentum().py(); 
        pztotal += ptcl->momentum().pz(); 
      }

      if (abs(Id) < 6 || abs(Id) == 22){
        ++partonNum; partonpT->Fill(Log_p,weight);
      }

      ///counting multiplicities and filling momentum distributions
      switch(abs(Id)){

      case 1 : {
		if(Id > 0) {
          ++dNum; dMomentum->Fill(Log_p,weight);}
		else{
          ++dbarNum; dbarMomentum->Fill(Log_p,weight);}
      }
		break;
		//
      case 2 : {
		if(Id > 0) {
          ++uNum; uMomentum->Fill(Log_p,weight);}
		else{
          ++ubarNum; ubarMomentum->Fill(Log_p,weight);}
      }
		break;
		//
      case 3 :  {
		if(Id > 0) {
          ++sNum; sMomentum->Fill(Log_p,weight);}
		else{
          ++sbarNum; sbarMomentum->Fill(Log_p,weight);}
      }
		break;
		//
      case 4 : {
		if(Id > 0) {
          ++cNum; cMomentum->Fill(Log_p,weight);}
		else{
          ++cbarNum; cbarMomentum->Fill(Log_p,weight);}
      }
		break;
		//
      case 5 : {
		if(Id > 0) {
          ++bNum; bMomentum->Fill(Log_p,weight);}
		else{
          ++bbarNum; bbarMomentum->Fill(Log_p,weight);}
      }
		break;
		//
      case 6 : {
		if(Id > 0) {
          ++tNum; tMomentum->Fill(Log_p,weight);}
		else{
          ++tbarNum; tbarMomentum->Fill(Log_p,weight);}
      }
		break;
		//
      case 11 : {
		if(Id > 0) {
          ++eminusNum; eminusMomentum->Fill(Log_p,weight);}
		else{
          ++eplusNum; eplusMomentum->Fill(Log_p,weight);}
      }
		break;
		//
      case 12 : {
		if(Id > 0) {
          ++nueNum; nueMomentum->Fill(Log_p, weight);}
		else{
          ++nuebarNum; nuebarMomentum->Fill(Log_p,weight);}
      }
		break;
		//
      case 13 : {
		if(Id > 0) {
          ++muminusNum; muminusMomentum->Fill(Log_p,weight);}
		else{
          ++muplusNum; muplusMomentum->Fill(Log_p,weight);}
      }
		break;
		//
      case 14 : {
		if(Id > 0) {
          ++numuNum; numuMomentum->Fill(Log_p,weight);}
		else{
          ++numubarNum; numubarMomentum->Fill(Log_p,weight);}
      }
		break;
		//
      case 15 : {
		if(Id > 0) {
          ++tauminusNum; tauminusMomentum->Fill(Log_p,weight);}
		else{
          ++tauplusNum; tauplusMomentum->Fill(Log_p,weight);}
      }
		break;
		//
      case 16 : {
		if(Id > 0) {
          ++nutauNum; nutauMomentum->Fill(Log_p,weight);}
		else{
          ++nutaubarNum; nutaubarMomentum->Fill(Log_p,weight);}
      }
		break;
		//
		//
      case 21 : {
		++gluNum; gluMomentum->Fill(Log_p,weight); 
      }
		break;
		//
      case 22 : {
		++gammaNum; gammaMomentum->Fill(Log_p,weight);
      }
		break;
		//
      case 23 : {
		++ZNum; ZMomentum->Fill(Log_p,weight);
      }
		break;
      case 24 : {
		if(Id > 0) {
          ++WplusNum; WplusMomentum->Fill(Log_p,weight);}
		else{
          ++WminusNum; WminusMomentum->Fill(Log_p,weight);}
      }
		break;
		//
		//
      case 211 : {
		if(Id > 0) {
          ++piplusNum; piplusMomentum->Fill(Log_p,weight);}
		else{
          ++piminusNum; piminusMomentum->Fill(Log_p,weight);}
      }
		break;
		//
      case 111 : {
		++pizeroNum; pizeroMomentum->Fill(Log_p,weight);
      }
		break;
		//
      case 321 : {
		if(Id > 0) {
          ++KplusNum; KplusMomentum->Fill(Log_p,weight);}
		else{
          ++KminusNum; KminusMomentum->Fill(Log_p,weight);}
      }
		break;
		//
      case 130 : {
        ++KlzeroNum; KlzeroMomentum->Fill(Log_p,weight);
      }
		break;
		//
      case 310 : {
		++KszeroNum; KszeroMomentum->Fill(Log_p,weight);
      }
		break;
		//
		//
      case 2212 : {
		if(Id > 0) {
          ++pNum; pMomentum->Fill(Log_p,weight);}
		else{
          ++pbarNum; pbarMomentum->Fill(Log_p,weight);}
      }
		break;
		//
      case 2112 : {
		if(Id > 0) {
          ++nNum; nMomentum->Fill(Log_p,weight);}
		else{
          ++nbarNum; nbarMomentum->Fill(Log_p,weight);}
      }
		break;
		//
		//
      case 3122 : {
		if(Id > 0) {
          ++l0Num; l0Momentum->Fill(Log_p,weight);}
		else{
          ++l0barNum; l0barMomentum->Fill(Log_p,weight);}
      }
        break;
        //
        //
      case 411 : {
		if(Id > 0) {
          ++DplusNum; DplusMomentum->Fill(Log_p,weight);}
		else{
          ++DminusNum; DminusMomentum->Fill(Log_p,weight);}
      }
		break;
		//
      case 421 : {
		++DzeroNum; DzeroMomentum->Fill(Log_p,weight);
      }
		break;
		//
      case 521 : {
		if(Id > 0) {
          ++BplusNum; BplusMomentum->Fill(Log_p,weight);}
		else{
          ++BminusNum; BminusMomentum->Fill(Log_p,weight);}
      }
		break;
		//
      case 511 : {
        ++BzeroNum; BzeroMomentum->Fill(Log_p,weight);
      }
		break;
		//
      case 531 : {
		++BszeroNum; BszeroMomentum->Fill(Log_p,weight);
      }
		break;
		//
      default : {
		++otherPtclNum; otherPtclMomentum->Fill(Log_p,weight);
      }
      }//switch
      //	if( 0 < Id && 100 > Id) ++part_counter[Id];
    }//event particles


  // set a default sqrt(s) and then check in the event
  double ecms = 7000.;
  if ( myGenEvent->valid_beam_particles() ) {
    ecms = myGenEvent->beam_particles().first->momentum().e()+myGenEvent->beam_particles().second->momentum().e();
  }
  DeltaEcms->Fill(etotal-ecms,weight);
  DeltaPx->Fill(pxtotal,weight);
  DeltaPy->Fill(pytotal,weight);
  DeltaPz->Fill(pztotal,weight);

 
  ///filling multiplicity ME's
  stablePtclNumber->Fill(log10(stablePtclNum+0.1),weight); 
  stableChaNumber->Fill(log10(stableChaNum+0.1),weight); 
  otherPtclNumber->Fill(log10(otherPtclNum+0.1),weight);
  unknownPDTNumber->Fill(log10(unknownPDTNum+0.1),weight);
  //
  dNumber->Fill(dNum,weight); uNumber->Fill(uNum,weight); sNumber->Fill(sNum,weight); cNumber->Fill(cNum,weight); bNumber->Fill(bNum,weight); tNumber->Fill(tNum,weight);  
  dbarNumber->Fill(dbarNum,weight); ubarNumber->Fill(ubarNum,weight); sbarNumber->Fill(sbarNum,weight); cbarNumber->Fill(cbarNum,weight); bbarNumber->Fill(bbarNum,weight); tbarNumber->Fill(tbarNum,weight); 
  partonNumber->Fill(partonNum,weight);
  //
  eminusNumber->Fill(eminusNum,weight); nueNumber->Fill(nueNum,weight); muminusNumber->Fill(muminusNum,weight); numuNumber->Fill(numuNum,weight); tauminusNumber->Fill(tauminusNum,weight); nutauNumber->Fill(nutauNum,weight);  
  eplusNumber->Fill(eplusNum,weight); nuebarNumber->Fill(nuebarNum,weight); muplusNumber->Fill(muplusNum,weight); numubarNumber->Fill(numubarNum,weight); tauplusNumber->Fill(tauplusNum,weight); nutaubarNumber->Fill(nutaubarNum,weight);  
  //
  ZNumber->Fill(ZNum,weight); WminusNumber->Fill(WminusNum,weight); WplusNumber->Fill(WplusNum,weight); 
  gammaNumber->Fill(log10(gammaNum+0.1),weight);
  gluNumber->Fill(log10(gluNum+0.1),weight);
  //
  piplusNumber->Fill(log10(piplusNum+0.1),weight);
  piminusNumber->Fill(log10(piminusNum+0.1),weight);
  pizeroNumber->Fill(log10(pizeroNum+0.1),weight);
  KplusNumber->Fill(KplusNum,weight); KminusNumber->Fill(KminusNum,weight); KlzeroNumber->Fill(KlzeroNum,weight); KszeroNumber->Fill(KszeroNum,weight); 
  //
  pNumber->Fill(pNum,weight); pbarNumber->Fill(pbarNum,weight); nNumber->Fill(nNum,weight); nbarNumber->Fill(nbarNum,weight); l0Number->Fill(l0Num); l0barNumber->Fill(l0barNum,weight);    
  //
  DplusNumber->Fill(DplusNum,weight); DminusNumber->Fill(DminusNum,weight); DzeroNumber->Fill(DzeroNum,weight); BplusNumber->Fill(BplusNum,weight); BminusNumber->Fill(BminusNum,weight); BzeroNumber->Fill(BzeroNum,weight); BszeroNumber->Fill(BszeroNum,weight); 

  delete myGenEvent;
}//analyze
