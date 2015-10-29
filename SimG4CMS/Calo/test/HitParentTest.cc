#include "SimG4CMS/Calo/test/HitParentTest.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "SimG4CMS/Calo/interface/CaloHitID.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include <boost/format.hpp>
#include <algorithm>

HitParentTest::HitParentTest(const edm::ParameterSet& ps) {

  sourceLabel = ps.getUntrackedParameter<std::string>("SourceLabel","VtxSmeared");
  g4Label = ps.getUntrackedParameter<std::string>("ModuleLabel","g4SimHits");
  hitLabEB= ps.getUntrackedParameter<std::string>("EBCollection","EcalHitsEB");
  hitLabEE= ps.getUntrackedParameter<std::string>("EECollection","EcalHitsEE");
  hitLabES= ps.getUntrackedParameter<std::string>("ESCollection","EcalHitsES");
  hitLabHC= ps.getUntrackedParameter<std::string>("HCCollection","HcalHits");
  edm::LogInfo("HitParentTest") << "Module Label: " << g4Label << "   Hits: "
				<< hitLabEB << ", " << hitLabEE << ", "
				<< hitLabES << ", " << hitLabHC;

  tok_eb_ = consumes<edm::PCaloHitContainer>(edm::InputTag(g4Label,hitLabEB));
  tok_ee_ = consumes<edm::PCaloHitContainer>(edm::InputTag(g4Label,hitLabEE));
  tok_es_ = consumes<edm::PCaloHitContainer>(edm::InputTag(g4Label,hitLabES));
  tok_hc_ = consumes<edm::PCaloHitContainer>(edm::InputTag(g4Label,hitLabHC));
  tok_tk_ = consumes<edm::SimTrackContainer>(edm::InputTag(g4Label));
  tok_vtx_ = consumes<edm::SimVertexContainer>(edm::InputTag(g4Label));

  for (unsigned int i=0; i<2; ++i) {
    total_num_apd_hits_seen[i]      = 0; 
    num_apd_hits_no_parent[i]       = 0; 
    num_apd_hits_no_simtrack[i]     = 0;
    num_apd_hits_no_gen_particle[i] = 0;
  }
  std::string dets[7] = {"EB", "EE", "ES", "HB", "HE", "HO", "HF"};
  for (unsigned int i=0; i<7; ++i) {
    detector[i]      = dets[i];
    totalHits[i]     = 0;
    noParent[i]      = 0;
    noSimTrack[i]    = 0;
    noGenParticle[i] = 0;
  }

  edm::Service<TFileService> tfile;
  if ( !tfile.isAvailable() ) {
    edm::LogInfo("HitParentTest") << "TFileService unavailable: no histograms";
    histos = false;
  } else {
    char  name[20], title[100];
    histos = true;
    for (unsigned int i=0; i<7; ++i) {
      sprintf (name, "HitType%d", i);
      sprintf (title, "Hit types for %s", detector[i].c_str());
      hitType[i] = tfile->make<TH1F>(name, title, 10, 0., 10.);
      hitType[i]->GetXaxis()->SetTitle(title);
      hitType[i]->GetYaxis()->SetTitle("Hits");
      sprintf (name, "RhoVertex%d", i);
      sprintf (title, "#rho of the vertex for %s Hits", detector[i].c_str());
      hitRho[i] = tfile->make<TH1F>(name, title, 10000, 0., 100.);
      hitRho[i]->GetXaxis()->SetTitle(title);
      hitRho[i]->GetYaxis()->SetTitle("Hits");
      sprintf (name, "ZVertex%d", i);
      sprintf (title, "z of the vertex for %s Hits", detector[i].c_str());
      hitZ[i] = tfile->make<TH1F>(name, title, 2000, -100., 100.);
      hitZ[i]->GetXaxis()->SetTitle(title);
      hitZ[i]->GetYaxis()->SetTitle("Hits");
    }
  } 
}

void HitParentTest::analyze(const edm::Event& e, const edm::EventSetup& ) {

  LogDebug("HitParentTest") << "Run = " << e.id().run() << " Event = " 
			       << e.id().event();

  // get PCaloHits for ecal barrel
  edm::Handle<edm::PCaloHitContainer> caloHitEB;
  e.getByToken(tok_eb_,caloHitEB); 

  // get PCaloHits for ecal endcap
  edm::Handle<edm::PCaloHitContainer> caloHitEE;
  e.getByToken(tok_ee_,caloHitEE); 

  // get PCaloHits for preshower
  edm::Handle<edm::PCaloHitContainer> caloHitES;
  e.getByToken(tok_es_,caloHitES); 

  // get PCaloHits for hcal
  edm::Handle<edm::PCaloHitContainer> caloHitHC;
  e.getByToken(tok_hc_,caloHitHC); 

  // get sim tracks
  e.getByToken(tok_tk_, SimTk);
  
  // get sim vertices
  e.getByToken(tok_vtx_, SimVtx);
  
  LogDebug("HitParentTest") << "HitParentTest: hits valid[EB]: " << caloHitEB.isValid() << " valid[EE]: " << caloHitEE.isValid() << " valid[ES]: " << caloHitES.isValid() << " valid[HC]: " << caloHitHC.isValid();
  
  if (caloHitEB.isValid()) {
    for (int depth = 1; depth <= 2; ++depth)
      analyzeAPDHits(*caloHitEB, depth);
    analyzeHits(*caloHitEB, 0);
  }
  if (caloHitEE.isValid()) analyzeHits(*caloHitEE, 1);
  if (caloHitES.isValid()) analyzeHits(*caloHitES, 2);
  if (caloHitHC.isValid()) analyzeHits(*caloHitHC, 3);

}

/** define the comparison for sorting the particle ids counting map */
class HitParentTestComparison {

protected:
  const std::map<int, unsigned> &particle_type_count;

public:
  HitParentTestComparison(const std::map<int, unsigned> &_particle_type_count) : 
    particle_type_count(_particle_type_count) {}

  bool operator()(int pid1, int pid2) const {
    return particle_type_count.at(pid1) > particle_type_count.at(pid2);
  }
};

void HitParentTest::endJob() {

  edm::LogVerbatim("HitParentTest") << "Total number of APD hits seen: " << total_num_apd_hits_seen[0]+ total_num_apd_hits_seen[1]; 
  for (unsigned int depth=0; depth<2; ++depth) {
    edm::LogVerbatim("HitParentTest") << "APD Hits in depth = " << depth+1 << " Total = " << total_num_apd_hits_seen[depth];
    edm::LogVerbatim("HitParentTest") << "summary of errors:";
    edm::LogVerbatim("HitParentTest") << "number of APD hits with zero geant track id: " << num_apd_hits_no_parent[depth];
    edm::LogVerbatim("HitParentTest") << "number of APD hits for which the parent simtrack was not found in the simtrack collection: " << num_apd_hits_no_simtrack[depth];
    edm::LogVerbatim("HitParentTest") << "number of APD hits for which no generator particle was found: "  << num_apd_hits_no_gen_particle[depth];
    edm::LogVerbatim("HitParentTest") << "";
  }

  for (unsigned int det=0; det<7; ++det) {
    edm::LogVerbatim("HitParentTest") << "Total number of hits seen in " << detector[det] << ": " << totalHits[det];
    edm::LogVerbatim("HitParentTest") << "summary of errors:";
    edm::LogVerbatim("HitParentTest") << "number of hits with zero geant track id: " << noParent[det];
    edm::LogVerbatim("HitParentTest") << "number of hits for which the parent simtrack was not found in the simtrack collection: " << noSimTrack[det];
    edm::LogVerbatim("HitParentTest") << "number of hits for which no generator particle was found: "  << noGenParticle[det];
    edm::LogVerbatim("HitParentTest") << "";
  }  

  // sort in decreasing order of occurence 
  std::vector<int> sorted_pids;
  for (std::map<int, unsigned>::const_iterator it = particle_type_count.begin(); it != particle_type_count.end(); ++it)
    sorted_pids.push_back(it->first);

  // now sort the pids 

  std::sort(sorted_pids.begin(), sorted_pids.end(),   HitParentTestComparison(particle_type_count));

  edm::LogVerbatim("HitParentTest") << "frequency particle types through the APD (pid/frequency):";
  for (unsigned i = 0; i < sorted_pids.size(); ++i) {
    int pid = sorted_pids[i];
    edm::LogVerbatim("HitParentTest") << "  pid " << boost::format("%6d") % pid << ": count " 
					 << boost::format("%6d") % particle_type_count[pid];
  }  

}

void HitParentTest::analyzeHits(const std::vector<PCaloHit>& hits, int type) {

  for (std::vector<PCaloHit>::const_iterator hit_it = hits.begin(); hit_it != hits.end(); ++hit_it) {
    int id(type), flag(0);
    if (type == 3) {
      HcalDetId id_    = HcalDetId(hit_it->id());
      int subdet       = id_.subdet();
      if      (subdet == static_cast<int>(HcalEndcap))  id = type+1;
      else if (subdet == static_cast<int>(HcalOuter))   id = type+2;
      else if (subdet == static_cast<int>(HcalForward)) id = type+3;
    }
    ++totalHits[id];

    // get the geant track id
    int hit_geant_track_id = hit_it->geantTrackId();

    if (hit_geant_track_id <= 0) {
      ++noParent[id];
      flag = 1;
    } else {
      bool found = false;
      flag = 2;
      // check whether this id is actually there in the list of simtracks
      for (edm::SimTrackContainer::const_iterator simTrkItr = SimTk->begin(); simTrkItr != SimTk->end() && !found; ++simTrkItr) {
	if (hit_geant_track_id == (int)(simTrkItr->trackId())) {
	  found = true;
	  flag  = 3;
	  bool match = validSimTrack(hit_geant_track_id, simTrkItr);
	
	  LogDebug("HitParentTest") << "[" << detector[type] << "] Match = " << match << " hit_geant_track_id=" << hit_geant_track_id  << " particle id=" << simTrkItr->type();
	
	  if (!match) {
	    LogDebug("HitParentTest") << "NO MATCH FOUND !";
	    ++noGenParticle[id];
	  }

	  // beam pipe...
	  int pid = simTrkItr->type();
	  math::XYZTLorentzVectorD oldest_parent_vertex = getOldestParentVertex(simTrkItr);

	  edm::SimTrackContainer::const_iterator oldest_parent_track = parentSimTrack(simTrkItr);
              
	  LogDebug("HitParentTest") << "[" << detector[type] << "] Hit pid = " << pid 
				    << "  hit track id = " << hit_geant_track_id
				    << " Oldest Parent's Vertex: " << oldest_parent_vertex 
				    << " rho = " << oldest_parent_vertex.Rho() 
				    << " Oldest Parent's pid: " << oldest_parent_track->type()
				    << " Oldest Parent's track id: " << oldest_parent_track->trackId()
				    << "\nHit vertex index: " << simTrkItr->vertIndex() << " (tot #vertices: " << SimVtx->size() << ")"
				    << "\nHit vertex parent track: " << (*SimVtx)[simTrkItr->vertIndex()].parentIndex() << " present=" << simTrackPresent((*SimVtx)[simTrkItr->vertIndex()].parentIndex());
	  if (histos) {
	    hitRho[id]->Fill(oldest_parent_vertex.Rho());
	    hitZ[id]->Fill(oldest_parent_vertex.Z());
	  }
	} // a match was found
      } // geant track id found in simtracks
      
      if (!found) ++noSimTrack[id];
    } // hits with a valid SimTrack Id
    if (histos) hitType[id]->Fill((double)(flag));
  } // loop over all calohits (of the given depth)
}

void HitParentTest::analyzeAPDHits(const std::vector<PCaloHit>& hits, int depth) {

  for (std::vector<PCaloHit>::const_iterator hit_it = hits.begin(); hit_it != hits.end(); ++hit_it) {

    if (hit_it->depth() == depth) {

      ++total_num_apd_hits_seen[depth-1];

      // get the geant track id
      int hit_geant_track_id = hit_it->geantTrackId();

      if (hit_geant_track_id <= 0) {
	++num_apd_hits_no_parent[depth-1];
      } else {
    
	bool found = false;
	// check whether this id is actually there in the list of simtracks
	for (edm::SimTrackContainer::const_iterator simTrkItr = SimTk->begin(); simTrkItr != SimTk->end() && !found; ++simTrkItr) {
	  if (hit_geant_track_id == (int)(simTrkItr->trackId())) {
	    found = true;
	    bool match = validSimTrack(hit_geant_track_id, simTrkItr);
	
	    edm::LogInfo("HitParentTest") << "APDHIT Match = " << match << " hit_geant_track_id = " << hit_geant_track_id  << " particle id=" << simTrkItr->type();
	
	    if (!match) {
	      edm::LogInfo("HitParentTest") << "NO MATCH FOUND !";
	      ++num_apd_hits_no_gen_particle[depth-1];
	    }

	    int apd_pid = simTrkItr->type();
	    std::map<int, unsigned>::iterator count_it = particle_type_count.find(apd_pid);
	    if (count_it == particle_type_count.end())
	      // first occurence of this particle pid
	      particle_type_count[apd_pid] = 1;
	    else
	      ++count_it->second;

	    //--------------------
	    // check where the oldest parent has its vertex. Should be close to the
	    // beam pipe...
	    math::XYZTLorentzVectorD oldest_parent_vertex = getOldestParentVertex(simTrkItr);

	    edm::SimTrackContainer::const_iterator oldest_parent_track = parentSimTrack(simTrkItr);
              
	    edm::LogInfo("HitParentTest") << "APD hit pid = " << apd_pid 
					  << " APD hit track id = " << hit_geant_track_id
					  << " depth = " << hit_it->depth()
					  << " OLDEST PARENT'S VERTEX: " << oldest_parent_vertex 
					  << " rho = " << oldest_parent_vertex.Rho() 
					  << " OLDEST PARENT'S PID: " << oldest_parent_track->type()
					  << " OLDEST PARENT'S track id: " << oldest_parent_track->trackId() << "\n"
					  << "APD hit vertex index: " << simTrkItr->vertIndex() << " (tot #vertices: " << SimVtx->size() << ")" << "\n"
					  << "APD hit vertex parent track: " << (*SimVtx)[simTrkItr->vertIndex()].parentIndex() << " present=" << simTrackPresent((*SimVtx)[simTrkItr->vertIndex()].parentIndex());
	
	  } // a match was found
	} // geant track id found in simtracks

	if (!found)
	  ++num_apd_hits_no_simtrack[depth-1];
      } // hits with a valid SimTrack Id
    } // right depth index
  } // loop over all calohits (of the given depth)
}

bool HitParentTest::simTrackPresent(int id) {

  for (edm::SimTrackContainer::const_iterator simTrkItr = SimTk->begin(); simTrkItr != SimTk->end(); ++simTrkItr) {
    if ((int)(simTrkItr->trackId()) == id)
      return true;
  }
  return false;
}

math::XYZTLorentzVectorD HitParentTest::getOldestParentVertex(edm::SimTrackContainer::const_iterator track) {

  static const math::XYZTLorentzVectorD invalid_vertex(10000,10000,10000,10000); // default value if no valid vertex found
  
  edm::SimTrackContainer::const_iterator oldest_parent_track = parentSimTrack(track);

  int vertex_index = oldest_parent_track->vertIndex();

  // sanity checks
  if (vertex_index < 0 || vertex_index >= (int)(SimVtx->size()))
    return invalid_vertex;

  return (*SimVtx)[vertex_index].position();
}

edm::SimTrackContainer::const_iterator HitParentTest::parentSimTrack(edm::SimTrackContainer::const_iterator thisTrkItr) {

  edm::SimTrackContainer::const_iterator itr = SimTk->end();

  int vertIndex = thisTrkItr->vertIndex();
  int type = thisTrkItr->type(); int charge = (int)thisTrkItr->charge();
  LogDebug("HitParentTest") << "SimTrackParent " << thisTrkItr->trackId() << " Vertex " <<  vertIndex << " Type " << type << " Charge " << charge << std::endl;

  if( vertIndex == -1 )                      return thisTrkItr;
  else if (vertIndex >= (int)SimVtx->size()) return itr;

  edm::SimVertexContainer::const_iterator simVtxItr= SimVtx->begin();
  for (int iv=0; iv<vertIndex; iv++) simVtxItr++;
  int parent = simVtxItr->parentIndex();

  if (parent < 0 && simVtxItr != SimVtx->begin()) {
    const math::XYZTLorentzVectorD pos1 = simVtxItr->position();
    for (simVtxItr=SimVtx->begin(); simVtxItr!=SimVtx->end(); ++simVtxItr) {
      if (simVtxItr->parentIndex() > 0) {
	const math::XYZTLorentzVectorD pos2 = pos1 - simVtxItr->position();
	double dist = pos2.P();
	if (dist < 0.001) {
	  parent = simVtxItr->parentIndex();
	  break;
	}
      }
    }
  }
  for (edm::SimTrackContainer::const_iterator simTrkItr= SimTk->begin(); simTrkItr!= SimTk->end(); simTrkItr++){
    if ((int)simTrkItr->trackId() == parent && simTrkItr != thisTrkItr) return  parentSimTrack(simTrkItr);
  }

  return thisTrkItr;
}

bool HitParentTest::validSimTrack(unsigned int simTkId, edm::SimTrackContainer::const_iterator thisTrkItr) {

  LogDebug("HitParentTest") << "Inside validSimTrack: trackId " << thisTrkItr->trackId() << " vtxIndex " <<  thisTrkItr->vertIndex() << " to be matched to " << simTkId;

  //This track originates from simTkId
  if (thisTrkItr->trackId() == simTkId) return true;

  //Otherwise trace back the history using SimTracks and SimVertices
  int vertIndex = thisTrkItr->vertIndex();
  if (vertIndex == -1 || vertIndex >= (int)SimVtx->size()) return false;
  edm::SimVertexContainer::const_iterator simVtxItr= SimVtx->begin();
  for (int iv=0; iv<vertIndex; iv++) simVtxItr++;
  int parent = simVtxItr->parentIndex();
  LogDebug("HitParentTest") << "validSimTrack:: parent index " << parent <<" ";
  if (parent < 0 && simVtxItr != SimVtx->begin()) {
    const math::XYZTLorentzVectorD pos1 = simVtxItr->position();
    for (simVtxItr=SimVtx->begin(); simVtxItr!=SimVtx->end(); ++simVtxItr) {
      if (simVtxItr->parentIndex() > 0) {
	const math::XYZTLorentzVectorD pos2 = pos1 - simVtxItr->position();
	double dist = pos2.P();
	if (dist < 0.001) {
	  parent = simVtxItr->parentIndex();
	  break;
	}
      }
    }
  }
  LogDebug("HitParentTest") << "final index " << parent;
  for (edm::SimTrackContainer::const_iterator simTrkItr= SimTk->begin(); simTrkItr!= SimTk->end(); simTrkItr++){
    if ((int)simTrkItr->trackId() == parent && simTrkItr != thisTrkItr) return  validSimTrack(simTkId, simTrkItr) ;
  }

  return false;
}

//define this as a plug-in
DEFINE_FWK_MODULE(HitParentTest);
