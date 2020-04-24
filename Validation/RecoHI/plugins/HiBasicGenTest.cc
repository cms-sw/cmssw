#include "HepMC/GenEvent.h"
#include "HepMC/GenParticle.h"
#include "Validation/RecoHI/plugins/HiBasicGenTest.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "HepMC/GenEvent.h"
#include "HepMC/HeavyIon.h"

#include <TString.h>
#include <TMath.h>

using namespace edm;
using namespace HepMC;

HiBasicGenTest::HiBasicGenTest(const edm::ParameterSet& iPSet)
{
  generatorToken_ = consumes<edm::HepMCProduct> (
      iPSet.getParameter<edm::InputTag>("generatorLabel"));
}

HiBasicGenTest::~HiBasicGenTest() {}

void HiBasicGenTest::bookHistograms(DQMStore::IBooker & ibooker,
  edm::Run const &, edm::EventSetup const & ){

  ///Setting the DQM top directories
  ibooker.setCurrentFolder("Generator/Particles");

  ///Booking the ME's
  for (int ibin = 0; ibin < 3; ibin++) {
    dnchdeta[ibin] = ibooker.book1D(Form("dnchdeta%d", ibin), ";#eta;dN^{ch}/d#eta",
        100, -6.0, 6.0);

    dnchdpt[ibin] = ibooker.book1D(Form("dnchdpt%d", ibin), ";p_{T};dN^{ch}/dp_{T}",
        200, 0.0, 100.0);

    b[ibin] = ibooker.book1D(Form("b%d",ibin),";b[fm];events", 100, 0.0, 20.0);
    dnchdphi[ibin] = ibooker.book1D(Form("dnchdphi%d", ibin), ";#phi;dN^{ch}/d#phi",
        100, -3.2, 3.2);

    ibooker.tag(dnchdeta[ibin], 1+ibin*4);
    ibooker.tag(dnchdpt[ibin], 2+ibin*4);
    ibooker.tag(b[ibin], 3+ibin*4);
    ibooker.tag(dnchdphi[ibin], 4+ibin*4);
  }

  rp = ibooker.book1D("phi0", ";#phi_{RP};events", 100, -3.2, 3.2);
  ibooker.tag(rp, 13);
}

void HiBasicGenTest::dqmBeginRun(const edm::Run& iRun,const edm::EventSetup& iSetup)
{
  iSetup.getData(pdt);
}


void HiBasicGenTest::analyze(const edm::Event& iEvent,const edm::EventSetup& iSetup)
{

  Handle<HepMCProduct> mc;
  iEvent.getByToken(generatorToken_, mc);
  const HepMC::GenEvent *evt = mc->GetEvent();
  const HepMC::HeavyIon *hi = evt->heavy_ion();

  int cbin = 0;
  double phi0 =0.;

  if(hi){

    double ip = hi->impact_parameter();
    phi0 = hi->event_plane_angle();
    
    // fill reaction plane distribution
    rp->Fill(phi0);
    
    // if the event is in one of the centrality bins of interest fill hists
    int cbin=-1;
    if(ip < 5.045) cbin=0;
    else if (ip < 7.145 && ip > 5.045) cbin=1;
    else if (ip < 15.202 && ip > 14.283) cbin=2;
    if(cbin<0) return;
    
    // fill impact parameter distributions
    b[cbin]->Fill(ip);
  }

  // loop over particles
  HepMC::GenEvent::particle_const_iterator begin = evt->particles_begin();
  HepMC::GenEvent::particle_const_iterator end = evt->particles_end();
  for(HepMC::GenEvent::particle_const_iterator it = begin; it != end; ++it){

    // only fill hists for status=1 particles
    if((*it)->status() != 1) continue;

    // only fill hists for charged particles
    int pdg_id = (*it)->pdg_id();
    const ParticleData * part = pdt->particle(pdg_id);
    int charge = static_cast<int>(part->charge());
    if(charge==0) continue;

    float eta = (*it)->momentum().eta();
    float phi = (*it)->momentum().phi();
    float pt = (*it)->momentum().perp();

    dnchdeta[cbin]->Fill(eta);
    dnchdpt[cbin]->Fill(pt);

    double pi = TMath::Pi();
    double p = phi-phi0;
    if(p > pi) p = p - 2*pi;
    if(p < -1*pi) p = p + 2*pi;
    dnchdphi[cbin]->Fill(p);

  }

  return;

}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HiBasicGenTest);


