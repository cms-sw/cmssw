/*
#include "TauAnalysis/MCEmbeddingTools/plugins/CheckHepMCEvent.h"

CheckHepMCEvent::CheckHepMCEvent(const edm::ParameterSet& pset)
{
	HepMCSource_= pset.getUntrackedParameter<string>("HepMCSource","newSource");
}

CheckHepMCEvent::~CheckHepMCEvent()
{

}
void CheckHepMCEvent::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
	using namespace HepMC;

	Handle<edm::HepMCProduct> HepMCHandle;
	if (!iEvent.getByLabel(HepMCSource_,HepMCHandle))
	{
		LogError("CheckHepMCEvent") << "[EEE] \t Could not read event!\n";
		return;
	}

	HepMC::GenEvent * evt = new HepMC::GenEvent(*(HepMCHandle->GetEvent()));

// 	evt->print(std::cout);

	int total_cnt_vertices = evt->vertices_size();
	int total_cnt_particles = evt->particles_size();

	LogInfo("CheckHepMCEvent") << "[III] Event contains...\n";
	LogInfo("CheckHepMCEvent") << "[III] \t"<< total_cnt_particles << " particles\n";
	LogInfo("CheckHepMCEvent") << "[III] \t"<< total_cnt_vertices << " vertices\n";

	for ( GenEvent::vertex_iterator vert = evt->vertices_begin(); vert != evt->vertices_end(); vert++ ) 
	{
// 		std::cout << "[III] Vertex...\n";
// 		(*vert)->print(std::cout);
		if ((*vert)->particles_in_size() <= 0)
			LogError("CheckHepMCEvent") << "[EEE] \t vertex has no incoming particles\n";

		if ((*vert)->particles_out_size() <= 0)
			LogError("CheckHepMCEvent") << "[EEE] \t vertex has no outgoing particles\n";

		if (!(*vert)->check_momentum_conservation())
			LogWarning("CheckHepMCEvent") << "[WWW] \t particles in vertex do not obey momentum conservation\n";

		for (GenVertex::particles_in_const_iterator it = (*vert)->particles_in_const_begin(); it!=(*vert)->particles_in_const_end(); it++)
			checkParticle(*it);

		for (GenVertex::particles_out_const_iterator it = (*vert)->particles_out_const_begin(); it!=(*vert)->particles_out_const_end(); it++)
		{
			checkParticle(*it);
			if ((*it)->production_vertex() && (*it)->production_vertex()!=(*vert))
				LogError("CheckHepMCEvent") << "[EEE] \t production vertex of outgoing particles is not identical to this vertex\n";
		}
	}

	delete evt;
}

void CheckHepMCEvent::checkParticle(HepMC::GenParticle * part)
{
// 	std::cout << "[III] Particle... " << part->pdg_id() << " " << part->status()<< "\n";
	if (!part->parent_event())
		LogError("CheckHepMCEvent") << "[EEE] \t particle without parent event\n";

	if (!part->production_vertex())
		LogWarning("CheckHepMCEvent") << "[WWW] \t particle without production vertexz\n";
		
	if (part->status()<=0)
		LogError("CheckHepMCEvent") << "[EEE] \t invalidstatus\n";

	if (part->status()==1 && (abs(part->pdg_id())<=9 || abs(part->pdg_id())==21))
	{
		LogError("CheckHepMCEvent") << "[EEE] \t gluon or quark in final state\n";
		
	}
	if (part->status()!=1 && !part->end_vertex())
	{
		LogError("CheckHepMCEvent") << "[EEE] \t decaying particle without end_vertex\n";
	}
}

void CheckHepMCEvent::beginJob(const edm::EventSetup& )
{

}

void CheckHepMCEvent::endJob()
{

}
*/
