#include "TFile.h"
#include "TTree.h"

#include "HepMC/GenEvent.h"
#include <cassert>

int main() {
  //Write

  constexpr int eventNumber = 10;
  {
    TFile oFile("rootio_t.root", "RECREATE");
    TTree events("Events","Events");

    HepMC::GenEvent event;

    auto vtx = new HepMC::GenVertex( HepMC::FourVector(0.,0.,0.));

    constexpr double kMass = 1.;
    constexpr double kMom = 1.;
    const double kEnergy = sqrt( kMass*kMass+kMom*kMom);
    constexpr int id = 11;

    auto p1 = new HepMC::GenParticle(HepMC::FourVector{kMom,0.,0.,kEnergy}, id, 1);

    p1->suggest_barcode(1);
    vtx->add_particle_out(p1);


    auto p2 = new HepMC::GenParticle(HepMC::FourVector{-kMom,0.,0.,kEnergy}, -id, 1);

    p2->suggest_barcode(2);
    vtx->add_particle_out(p2);

    auto decayVtx = new HepMC::GenVertex(HepMC::FourVector(1.,0.,0.));
    decayVtx->add_particle_in(p1);

    event.add_vertex(vtx);
    event.add_vertex(decayVtx);

    event.set_event_number(eventNumber);
    event.set_signal_process_id(20);

    HepMC::GenEvent* pEv = &event;

    events.Branch("GenEvent",&pEv);

    events.Fill();
    
    oFile.Write();
    oFile.Close();
  }

  //Read
  {
    TFile iFile("rootio_t.root");

    TTree* events = dynamic_cast<TTree*>( iFile.Get("Events") );

    HepMC::GenEvent event;
    HepMC::GenEvent* pEv = &event;
    events->SetBranchAddress("GenEvent",&pEv);

    events->GetEntry(0);

    assert(event.event_number() == eventNumber);
    assert(event.particles_size() == 2);
    assert(event.vertices_size() == 2);
    
    int barcode = 1;
    for(auto it = event.particles_begin(); it != event.particles_end(); ++it) {
      if((*it)->production_vertex()) {
        auto vtx = (*it)->production_vertex();
        assert( std::find(vtx->particles_out_const_begin(), vtx->particles_out_const_end(), *it) != vtx->particles_out_const_end());
      }
      if((*it)->end_vertex()) {
        auto vtx = (*it)->end_vertex();
        assert( std::find(vtx->particles_in_const_begin(), vtx->particles_in_const_end(), *it) != vtx->particles_in_const_end());
      }

      assert( (*it)->barcode() == barcode);
      assert( *it == event.barcode_to_particle( (*it)->barcode() ) );
      ++barcode;
    }

    
  }

}
