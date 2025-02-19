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
  dbe = 0;
  dbe = edm::Service<DQMStore>().operator->();
}

HiBasicGenTest::~HiBasicGenTest() {}

void HiBasicGenTest::beginJob()
{
  if(dbe){
    ///Setting the DQM top directories
    dbe->setCurrentFolder("Generator/Particles");
    
    ///Booking the ME's
    for(int ibin=0; ibin<3; ibin++) {
      dnchdeta[ibin] = dbe->book1D(Form("dnchdeta%d",ibin), ";#eta;dN^{ch}/d#eta", 100, -6.0, 6.0);
      dnchdpt[ibin] = dbe->book1D(Form("dnchdpt%d",ibin), ";p_{T};dN^{ch}/dp_{T}", 200, 0.0, 100.0);
      b[ibin] = dbe->book1D(Form("b%d",ibin),";b[fm];events",100, 0.0, 20.0);
      dnchdphi[ibin] = dbe->book1D(Form("dnchdphi%d",ibin),";#phi;dN^{ch}/d#phi",100, -3.2, 3.2);

      dbe->tag(dnchdeta[ibin]->getFullname(),1+ibin*4);
      dbe->tag(dnchdpt[ibin]->getFullname(),2+ibin*4);
      dbe->tag(b[ibin]->getFullname(),3+ibin*4);
      dbe->tag(dnchdphi[ibin]->getFullname(),4+ibin*4);
    }

    rp = dbe->book1D("phi0",";#phi_{RP};events",100,-3.2,3.2);
    dbe->tag(rp->getFullname(),13);


 }
  return;
}

void HiBasicGenTest::endJob(){
  // normalization of histograms can be done here (or in post-processor)
  return;
}

void HiBasicGenTest::beginRun(const edm::Run& iRun,const edm::EventSetup& iSetup)
{
  iSetup.getData(pdt);
  return;
}

void HiBasicGenTest::endRun(const edm::Run& iRun,const edm::EventSetup& iSetup){return;}

void HiBasicGenTest::analyze(const edm::Event& iEvent,const edm::EventSetup& iSetup)
{ 

  Handle<HepMCProduct> mc;
  iEvent.getByLabel("generator",mc);
  const HepMC::GenEvent *evt = mc->GetEvent();
  const HepMC::HeavyIon *hi = evt->heavy_ion();
  
  double ip = hi->impact_parameter();
  double phi0 = hi->event_plane_angle();

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


