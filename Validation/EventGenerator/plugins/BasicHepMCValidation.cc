/*class BasicHepMCValidation
 *  
 *  Class to fill dqm monitor elements from existing EDM file
 *
 *  $Date: 2010/07/02 13:34:23 $
 *  $Revision: 1.2 $
 */
 
#include "Validation/EventGenerator/interface/BasicHepMCValidation.h"

#include "CLHEP/Units/defs.h"
#include "CLHEP/Units/PhysicalConstants.h"

using namespace edm;

BasicHepMCValidation::BasicHepMCValidation(const edm::ParameterSet& iPSet):  
  hepmcCollection_(iPSet.getParameter<edm::InputTag>("hepmcCollection"))
{    
  dbe = 0;
  dbe = edm::Service<DQMStore>().operator->();
}

BasicHepMCValidation::~BasicHepMCValidation() {}

void BasicHepMCValidation::beginJob()
{
  if(dbe){
	///Setting the DQM top directories
	dbe->setCurrentFolder("Generator/Particles");
    
    // Number of analyzed events
    nEvt = dbe->book1D("nEvt", "n analyzed Events", 1, 0., 1.);
	
	///Booking the ME's
	///multiplicity
	uNumber = dbe->book1D("uNumber", "No. u", 20, 0, 20);
	dNumber = dbe->book1D("dNumber", "No. d", 20, 0, 20);
	sNumber = dbe->book1D("sNumber", "No. s", 20, 0, 20);
    cNumber = dbe->book1D("cNumber", "No. c", 20, 0, 20);
	bNumber = dbe->book1D("bNumber", "No. b", 20, 0, 20);
	tNumber = dbe->book1D("tNumber", "No. t", 20, 0, 20);
	//
	ubarNumber = dbe->book1D("ubarNumber", "No. ubar", 20, 0, 20);
	dbarNumber = dbe->book1D("dbarNumber", "No. dbar", 20, 0, 20);
	sbarNumber = dbe->book1D("sbarNumber", "No. sbar", 20, 0, 20);
    cbarNumber = dbe->book1D("cbarNumber", "No. cbar", 20, 0, 20);
	bbarNumber = dbe->book1D("bbarNumber", "No. bbar", 20, 0, 20);
	tbarNumber = dbe->book1D("tbarNumber", "No. tbar", 20, 0, 20);
	//
	eminusNumber = dbe->book1D("eminusNumber", "No. e-", 20, 0, 20);
	nueNumber = dbe->book1D("nueNumber", "No. nu_e", 20, 0, 20);
	muminusNumber = dbe->book1D("muminusNumber", "No. mu-", 20, 0, 20);
	numuNumber = dbe->book1D("numuNumber", "No. nu_mu", 20, 0, 20);
	tauminusNumber = dbe->book1D("tauminusNumber", "No. tau-", 20, 0, 20);
	nutauNumber = dbe->book1D("nutauNumber", "No. nu_tau", 20, 0, 20);
	//
	eplusNumber = dbe->book1D("eplusNumber", "No. e+", 20, 0, 20);
	nuebarNumber = dbe->book1D("nuebarNumber", "No. nu_e_bar", 20, 0, 20);
	muplusNumber = dbe->book1D("muplusNumber", "No. mu+", 20, 0, 20);
	numubarNumber = dbe->book1D("numuNumber", "No. nu_mu_bar", 20, 0, 20);
	tauplusNumber = dbe->book1D("tauplusNumber", "No. tau+", 20, 0, 20);
	nutaubarNumber = dbe->book1D("nutauNumber", "No. nu_tau_bar", 20, 0, 20);
	//
	WplusNumber = dbe->book1D("WplusNumber", "No. W+", 20, 0, 20);
	WminusNumber = dbe->book1D("WminusNumber", "No. W-", 20, 0, 20);
	ZNumber = dbe->book1D("ZNumber", "No. Z", 20, 0, 20);
	gammaNumber = dbe->book1D("gammaNumber", "Log10(No. gamma)", 60, -1, 5); //Log
	gluNumber = dbe->book1D("gluonNumber", "Log10(No. gluons)", 60, -1, 5); //Log
	//
	piplusNumber = dbe->book1D("piplusNumber", "Log10(No. pi+)", 60, -1, 5); //Log
	piminusNumber = dbe->book1D("piminusNumber", "Log10(No. pi-)", 60, -1, 5); //Log
	pizeroNumber = dbe->book1D("pizeroNumber", "Log10(No. pi_0)", 60, -1, 5); //Log
	KplusNumber = dbe->book1D("KplusNumber", "No. K+", 100, 0, 100);
	KminusNumber = dbe->book1D("KminusNumber", "No. K-", 100, 0, 100);
	KlzeroNumber = dbe->book1D("KlzeroNumber", "No. K_l^0", 100, 0, 100);
	KszeroNumber = dbe->book1D("KszeroNumber", "No. K_s^0", 100, 0, 100);
	//
	pNumber = dbe->book1D("pNumber", "No. p", 100, 0, 100);
	pbarNumber = dbe->book1D("pbarNumber", "No. pbar", 100, 0, 100);
	nNumber = dbe->book1D("nNumber", "No. n", 100, 0, 100);
	nbarNumber = dbe->book1D("nbarNumber", "No. nbar", 100, 0, 100);
	l0Number = dbe->book1D("l0Number", "No. Lambda0", 100, 0, 100);
	l0barNumber = dbe->book1D("l0barNumber", "No. Lambda0bar", 100, 0, 100);
	//
	DplusNumber = dbe->book1D("DplusNumber", "No. D+", 20, 0, 20);
	DminusNumber = dbe->book1D("DminusNumber", "No. D-", 20, 0, 20);
	DzeroNumber = dbe->book1D("DzeroNumber", "No. D^0", 20, 0, 20);
	//
	BplusNumber = dbe->book1D("BplusNumber", "No. B+", 20, 0, 20);
	BminusNumber = dbe->book1D("BminusNumber", "No. B-", 20, 0, 20);
	BzeroNumber = dbe->book1D("BzeroNumber", "No. B^0", 20, 0, 20);
	BszeroNumber = dbe->book1D("BszeroNumber", "No. B^0_s", 20, 0, 20);
	//
	otherPtclNumber = dbe->book1D("otherPtclNumber", "Log10(No. other ptcls)", 60, -1, 5); //Log

	//Momentum 
	uMomentum = dbe->book1D("uMomentum", "Log10(p) u", 60, -2, 4);
	dMomentum = dbe->book1D("dMomentum", "Log10(p) d", 60, -2, 4);
	sMomentum = dbe->book1D("sMomentum", "Log10(p) s", 60, -2, 4);
    cMomentum = dbe->book1D("cMomentum", "Log10(p) c", 60, -2, 4);
	bMomentum = dbe->book1D("bMomentum", "Log10(p) b", 60, -2, 4);
	tMomentum = dbe->book1D("tMomentum", "Log10(p) t", 60, -2, 4);
	//
	ubarMomentum = dbe->book1D("ubarMomentum", "Log10(p) ubar", 60, -2, 4);
	dbarMomentum = dbe->book1D("dbarMomentum", "Log10(p) dbar", 60, -2, 4);
	sbarMomentum = dbe->book1D("sbarMomentum", "Log10(p) sbar", 60, -2, 4);
    cbarMomentum = dbe->book1D("cbarMomentum", "Log10(p) cbar", 60, -2, 4);
	bbarMomentum = dbe->book1D("bbarMomentum", "Log10(p) bbar", 60, -2, 4);
	tbarMomentum = dbe->book1D("tbarMomentum", "Log10(p) tbar", 60, -2, 4);
	//
	eminusMomentum = dbe->book1D("eminusMomentum", "Log10(p) e-", 60, -2, 4);
	nueMomentum = dbe->book1D("nueMomentum", "Log10(Momentum) nue", 60, -2, 4);
	muminusMomentum = dbe->book1D("muminusMomentum", "Log10(p) mu-", 60, -2, 4);
	numuMomentum = dbe->book1D("numuMomentum", "Log10(p) numu", 60, -2, 4);
	tauminusMomentum = dbe->book1D("tauminusMomentum", "Log10(p) tau-", 60, -2, 4);
	nutauMomentum = dbe->book1D("nutauMomentum", "Log10(p) nutau", 60, -2, 4);
	//
	eplusMomentum = dbe->book1D("eplusMomentum", "Log10(p) e+", 60, -2, 4);
	nuebarMomentum = dbe->book1D("nuebarMomentum", "Log10(p) nuebar", 60, -2, 4);
	muplusMomentum = dbe->book1D("muplusMomentum", "Log10(p) mu+", 60, -2, 4);
	numubarMomentum = dbe->book1D("numuMomentum", "Log10(p) numubar", 60, -2, 4);
	tauplusMomentum = dbe->book1D("tauplusMomentum", "Log10(p) tau+", 60, -2, 4);
	nutaubarMomentum = dbe->book1D("nutauMomentum", "Log10(p) nutaubar", 60, -2, 4);
	//
	gluMomentum = dbe->book1D("gluonMomentum", "Log10(p) gluons", 70, -3, 4);
	WplusMomentum = dbe->book1D("WplusMomentum", "Log10(p) W+", 60, -2, 4);
	WminusMomentum = dbe->book1D("WminusMomentum", "Log10(p) W-", 60, -2, 4);
	ZMomentum = dbe->book1D("ZMomentum", "Log10(p) Z", 60, -2, 4);
	gammaMomentum = dbe->book1D("gammaMomentum", "Log10(p) gamma", 70, -3, 4);
	//
	piplusMomentum = dbe->book1D("piplusMomentum", "Log10(p) pi+", 60, -2, 4);
	piminusMomentum = dbe->book1D("piminusMomentum", "Log10(p) pi-", 60, -2, 4);
	pizeroMomentum = dbe->book1D("pizeroMomentum", "Log10(p) pi_0", 60, -2, 4);
	KplusMomentum = dbe->book1D("KplusMomentum", "Log10(p) K+", 60, -2, 4);
	KminusMomentum = dbe->book1D("KminusMomentum", "Log10(p) K-", 60, -2, 4);
	KlzeroMomentum = dbe->book1D("KlzeroMomentum", "Log10(p) K_l^0", 60, -2, 4);
	KszeroMomentum = dbe->book1D("KszeroMomentum", "Log10(p) K_s^0", 60, -2, 4);
	//
	pMomentum = dbe->book1D("pMomentum", "Log10(p) p", 60, -2, 4);
	pbarMomentum = dbe->book1D("pbarMomentum", "Log10(p) pbar", 60, -2, 4);
	nMomentum = dbe->book1D("nMomentum", "Log10(p) n", 60, -2, 4);
	nbarMomentum = dbe->book1D("nbarMomentum", "Log10(p) nbar", 60, -2, 4);
	l0Momentum = dbe->book1D("l0Momentum", "Log10(p) Lambda0", 60, -2, 4);
	l0barMomentum = dbe->book1D("l0barMomentum", "Log10(p) Lambda0bar", 60, -2, 4);
	//
	DplusMomentum = dbe->book1D("DplusMomentum", "Log10(p) D+", 60, -2, 4);
	DminusMomentum = dbe->book1D("DminusMomentum", "Log10(p) D-", 60, -2, 4);
	DzeroMomentum = dbe->book1D("DzeroMomentum", "Log10(p) D^0", 60, -2, 4);
	//
	BplusMomentum = dbe->book1D("BplusMomentum", "Log10(p) B+", 60, -2, 4);
	BminusMomentum = dbe->book1D("BminusMomentum", "Log10(p) B-", 60, -2, 4);
	BzeroMomentum = dbe->book1D("BzeroMomentum", "Log10(p) B^0", 60, -2, 4);
	BszeroMomentum = dbe->book1D("BszeroMomentum", "Log10(p) B^0_s", 60, -2, 4);
	//
	otherPtclMomentum = dbe->book1D("otherPtclMomentum", "Log10(p) other ptcls", 60, -2, 4);

	///other
	genPtclNumber = dbe->book1D("genPtclNumber", "Log10(No. all particles)", 60, -1, 5); //Log
	genVrtxNumber = dbe->book1D("genVrtxNumber", "Log10(No. all vertexs)", 60, -1, 5); //Log
	//
	stablePtclNumber= dbe->book1D("stablePtclNumber", "Log10(No. stable particles)", 50, 0, 5); //Log
	stablePtclPhi = dbe->book1D("stablePtclPhi", "stable Ptcl Phi", 360, -180, 180);
	stablePtclEta = dbe->book1D("stablePtclEta", "stable Ptcl Eta (pseudo rapidity)", 220, -11, 11);
	stablePtclCharge = dbe->book1D("stablePtclCharge", "stablePtclCharge", 5, -2, 2);
	stableChaNumber= dbe->book1D("stableChaNumber", "Log10(No. stable charged particles)", 50, 0, 5); //Log
	stablePtclp = dbe->book1D("stablePtclp", "Log10(p) stable ptcl p", 80, -4, 4); //Log
	stablePtclpT = dbe->book1D("stablePtclpT", "Log10(pT) stable ptcl pT", 80, -4, 4); //Log
	outVrtxStablePtclNumber = dbe->book1D("outVrtxStablePtclNumber", "No. outgoing stable ptcls from vrtx", 10, 0, 10); 
	//
	outVrtxPtclNumber = dbe->book1D("outVrtxPtclNumber", "No. outgoing ptcls from vrtx", 30, 0, 30);
	vrtxZ = dbe->book1D("VrtxZ", "VrtxZ", 50 , -250, 250);
	vrtxRadius = dbe->book1D("vrtxRadius", "vrtxRadius", 50, 0, 50);
	//
	unknownPDTNumber = dbe->book1D("unknownPDTNumber", "Log10(No. unknown ptcls PDT)", 60, -1, 5); //Log
    genPtclStatus = dbe->book1D("genPtclStatus", "Status of genParticle", 200,0,200.);
	//
    Bjorken_x = dbe->book1D("Bjorken_x", "Bjorken_x", 1000, 0.0, 1.0);
    //
    status1ShortLived = dbe->book1D("status1ShortLived","Status 1 short lived", 11, 0, 11);
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
  }
  return;
}

void BasicHepMCValidation::endJob(){return;}
void BasicHepMCValidation::beginRun(const edm::Run& iRun,const edm::EventSetup& iSetup)
{
  ///Get PDT Table
  iSetup.getData( fPDGTable );
  return;
}
void BasicHepMCValidation::endRun(const edm::Run& iRun,const edm::EventSetup& iSetup){return;}
void BasicHepMCValidation::analyze(const edm::Event& iEvent,const edm::EventSetup& iSetup)
{ 
  ///counters to zero for every event
  int uNum = 0; int dNum = 0; int sNum = 0; int cNum = 0; int bNum = 0; int tNum = 0;
  int ubarNum = 0; int dbarNum = 0; int sbarNum = 0; int cbarNum = 0; int bbarNum = 0; int tbarNum = 0;
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

  ///Gathering the HepMCProduct information
  edm::Handle<HepMCProduct> evt;
  iEvent.getByLabel(hepmcCollection_, evt);

  //Get EVENT
  HepMC::GenEvent *myGenEvent = new HepMC::GenEvent(*(evt->GetEvent()));

  nEvt->Fill(0.5);

  genPtclNumber->Fill(log10(myGenEvent->particles_size()));     
  genVrtxNumber->Fill(log10(myGenEvent->vertices_size()));

  ///Bjorken variable from PDF
  HepMC::PdfInfo *pdf = myGenEvent->pdf_info();    
  if(pdf){
    bjorken = ((pdf->x1())/((pdf->x1())+(pdf->x2())));
  }
  Bjorken_x->Fill(bjorken);

  //Looping through the VERTICES in the event
  HepMC::GenEvent::vertex_const_iterator vrtxBegin = myGenEvent->vertices_begin();
  HepMC::GenEvent::vertex_const_iterator vrtxEnd = myGenEvent->vertices_end();
  for(HepMC::GenEvent::vertex_const_iterator vrtxIt = vrtxBegin; vrtxIt!=vrtxEnd; ++vrtxIt)
    {
      ///Vertices
      HepMC::GenVertex *vrtx = *vrtxIt;
      outVrtxPtclNumber->Fill(vrtx->particles_out_size()); //std::cout << "all " << vrtx->particles_out_size() << '\n';
      vrtxZ->Fill(vrtx->point3d().z());
      vrtxRadius->Fill(vrtx->point3d().perp());
	
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
      outVrtxStablePtclNumber->Fill(outVrtxStablePtclNum);
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
      genPtclStatus->Fill((float)status);

      ///Stable particles
      if(ptcl->status() == 1){
	    ++stablePtclNum;
	    stablePtclPhi->Fill(ptcl->momentum().phi()/CLHEP::degree); //std::cout << ptcl->polarization().phi() << '\n';
	    stablePtclEta->Fill(ptcl->momentum().pseudoRapidity());
	    stablePtclCharge->Fill(charge); // std::cout << ptclData.charge() << '\n';
	    stablePtclp->Fill(Log_p);
	    stablePtclpT->Fill(log10(ptcl->momentum().perp()));
        if (charge != 0. && charge != 999.) ++stableChaNum;
        if ( std::abs(Id) == 1 ) status1ShortLived->Fill(1);
        if ( std::abs(Id) == 2 ) status1ShortLived->Fill(2);
        if ( std::abs(Id) == 3 ) status1ShortLived->Fill(3);
        if ( std::abs(Id) == 4 ) status1ShortLived->Fill(4);
        if ( std::abs(Id) == 5 ) status1ShortLived->Fill(5);
        if ( std::abs(Id) == 6 ) status1ShortLived->Fill(6);
        if ( Id == 21 ) status1ShortLived->Fill(7);
        if ( std::abs(Id) == 15 ) status1ShortLived->Fill(8);
        if ( Id == 23 ) status1ShortLived->Fill(9);
        if ( std::abs(Id) == 24 ) status1ShortLived->Fill(10);
        if ( std::abs(Id) == 7 || std::abs(Id) == 8 || std::abs(Id) == 17 || (std::abs(Id) >= 25 && std::abs(Id) <= 99) ) status1ShortLived->Fill(11);
      }
	
      ///counting multiplicities and filling momentum distributions
      switch(abs(Id)){

      case 1 : {
		if(Id > 0) {
          ++dNum; dMomentum->Fill(Log_p);}
		else{
          ++dbarNum; dbarMomentum->Fill(Log_p);}
      }
		break;
		//
      case 2 : {
		if(Id > 0) {
          ++uNum; uMomentum->Fill(Log_p);}
		else{
          ++ubarNum; ubarMomentum->Fill(Log_p);}
      }
		break;
		//
      case 3 :  {
		if(Id > 0) {
          ++sNum; sMomentum->Fill(Log_p);}
		else{
          ++sbarNum; sbarMomentum->Fill(Log_p);}
      }
		break;
		//
      case 4 : {
		if(Id > 0) {
          ++cNum; cMomentum->Fill(Log_p);}
		else{
          ++cbarNum; cbarMomentum->Fill(Log_p);}
      }
		break;
		//
      case 5 : {
		if(Id > 0) {
          ++bNum; bMomentum->Fill(Log_p);}
		else{
          ++bbarNum; bbarMomentum->Fill(Log_p);}
      }
		break;
		//
      case 6 : {
		if(Id > 0) {
          ++tNum; tMomentum->Fill(Log_p);}
		else{
          ++tbarNum; tbarMomentum->Fill(Log_p);}
      }
		break;
		//
      case 11 : {
		if(Id > 0) {
          ++eminusNum; eminusMomentum->Fill(Log_p);}
		else{
          ++eplusNum; eplusMomentum->Fill(Log_p);}
      }
		break;
		//
      case 12 : {
		if(Id > 0) {
          ++nueNum; nueMomentum->Fill(Log_p);}
		else{
          ++nuebarNum; nuebarMomentum->Fill(Log_p);}
      }
		break;
		//
      case 13 : {
		if(Id > 0) {
          ++muminusNum; muminusMomentum->Fill(Log_p);}
		else{
          ++muplusNum; muplusMomentum->Fill(Log_p);}
      }
		break;
		//
      case 14 : {
		if(Id > 0) {
          ++numuNum; numuMomentum->Fill(Log_p);}
		else{
          ++numubarNum; numubarMomentum->Fill(Log_p);}
      }
		break;
		//
      case 15 : {
		if(Id > 0) {
          ++tauminusNum; tauminusMomentum->Fill(Log_p);}
		else{
          ++tauplusNum; tauplusMomentum->Fill(Log_p);}
      }
		break;
		//
      case 16 : {
		if(Id > 0) {
          ++nutauNum; nutauMomentum->Fill(Log_p);}
		else{
          ++nutaubarNum; nutaubarMomentum->Fill(Log_p);}
      }
		break;
		//
		//
      case 21 : {
		++gluNum; gluMomentum->Fill(Log_p); 
      }
		break;
		//
      case 22 : {
		++gammaNum; gammaMomentum->Fill(Log_p);
      }
		break;
		//
      case 23 : {
		++ZNum; ZMomentum->Fill(Log_p);
      }
		break;
      case 24 : {
		if(Id > 0) {
          ++WplusNum; WplusMomentum->Fill(Log_p);}
		else{
          ++WminusNum; WminusMomentum->Fill(Log_p);}
      }
		break;
		//
		//
      case 211 : {
		if(Id > 0) {
          ++piplusNum; piplusMomentum->Fill(Log_p);}
		else{
          ++piminusNum; piminusMomentum->Fill(Log_p);}
      }
		break;
		//
      case 111 : {
		++pizeroNum; pizeroMomentum->Fill(Log_p);
      }
		break;
		//
      case 321 : {
		if(Id > 0) {
          ++KplusNum; KplusMomentum->Fill(Log_p);}
		else{
          ++KminusNum; KminusMomentum->Fill(Log_p);}
      }
		break;
		//
      case 130 : {
        ++KlzeroNum; KlzeroMomentum->Fill(Log_p);
      }
		break;
		//
      case 310 : {
		++KszeroNum; KszeroMomentum->Fill(Log_p);
      }
		break;
		//
		//
      case 2212 : {
		if(Id > 0) {
          ++pNum; pMomentum->Fill(Log_p);}
		else{
          ++pbarNum; pbarMomentum->Fill(Log_p);}
      }
		break;
		//
      case 2112 : {
		if(Id > 0) {
          ++nNum; nMomentum->Fill(Log_p);}
		else{
          ++nbarNum; nbarMomentum->Fill(Log_p);}
      }
		break;
		//
		//
      case 3122 : {
		if(Id > 0) {
          ++l0Num; l0Momentum->Fill(Log_p);}
		else{
          ++l0barNum; l0barMomentum->Fill(Log_p);}
      }
        break;
        //
        //
      case 411 : {
		if(Id > 0) {
          ++DplusNum; DplusMomentum->Fill(Log_p);}
		else{
          ++DminusNum; DminusMomentum->Fill(Log_p);}
      }
		break;
		//
      case 421 : {
		++DzeroNum; DzeroMomentum->Fill(Log_p);
      }
		break;
		//
      case 521 : {
		if(Id > 0) {
          ++BplusNum; BplusMomentum->Fill(Log_p);}
		else{
          ++BminusNum; BminusMomentum->Fill(Log_p);}
      }
		break;
		//
      case 511 : {
        ++BzeroNum; BzeroMomentum->Fill(Log_p);
      }
		break;
		//
      case 531 : {
		++BszeroNum; BszeroMomentum->Fill(Log_p);
      }
		break;
		//
      default : {
		++otherPtclNum; otherPtclMomentum->Fill(Log_p);
      }
      }//switch
      //	if( 0 < Id && 100 > Id) ++part_counter[Id];
    }//event particles


  ///filling multiplicity ME's
  stablePtclNumber->Fill(log10(stablePtclNum+0.1)); 
  stableChaNumber->Fill(log10(stableChaNum+0.1)); 
  otherPtclNumber->Fill(log10(otherPtclNum+0.1));
  unknownPDTNumber->Fill(log10(unknownPDTNum+0.1));
  //
  dNumber->Fill(dNum); uNumber->Fill(uNum); sNumber->Fill(sNum); cNumber->Fill(cNum); bNumber->Fill(bNum); tNumber->Fill(tNum);  
  dbarNumber->Fill(dbarNum); ubarNumber->Fill(ubarNum); sbarNumber->Fill(sbarNum); cbarNumber->Fill(cbarNum); bbarNumber->Fill(bbarNum); tbarNumber->Fill(tbarNum);  
  //
  eminusNumber->Fill(eminusNum); nueNumber->Fill(nueNum); muminusNumber->Fill(muminusNum); numuNumber->Fill(numuNum); tauminusNumber->Fill(tauminusNum); nutauNumber->Fill(nutauNum);  
  eplusNumber->Fill(eplusNum); nuebarNumber->Fill(nuebarNum); muplusNumber->Fill(muplusNum); numubarNumber->Fill(numubarNum); tauplusNumber->Fill(tauplusNum); nutaubarNumber->Fill(nutaubarNum);  
  //
  ZNumber->Fill(ZNum); WminusNumber->Fill(WminusNum); WplusNumber->Fill(WplusNum); 
  gammaNumber->Fill(log10(gammaNum+0.1));
  gluNumber->Fill(log10(gluNum+0.1));
  //
  piplusNumber->Fill(log10(piplusNum+0.1));
  piminusNumber->Fill(log10(piminusNum+0.1));
  pizeroNumber->Fill(log10(pizeroNum+0.1));
  KplusNumber->Fill(KplusNum); KminusNumber->Fill(KminusNum); KlzeroNumber->Fill(KlzeroNum); KszeroNumber->Fill(KszeroNum); 
  //
  pNumber->Fill(pNum); pbarNumber->Fill(pbarNum); nNumber->Fill(nNum); nbarNumber->Fill(nbarNum); l0Number->Fill(l0Num); l0barNumber->Fill(l0barNum);    
  //
  DplusNumber->Fill(DplusNum); DminusNumber->Fill(DminusNum); DzeroNumber->Fill(DzeroNum); BplusNumber->Fill(BplusNum); BminusNumber->Fill(BminusNum); BzeroNumber->Fill(BzeroNum); BszeroNumber->Fill(BszeroNum); 

  delete myGenEvent;
}//analyze
