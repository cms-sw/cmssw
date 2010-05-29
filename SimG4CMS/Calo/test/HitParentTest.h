#ifndef SimG4CMS_Calo_HitParentTest_H
#define SimG4CMS_Calo_HitParentTest_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"

#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

#include <TH1F.h>

#include <memory>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>

class HitParentTest: public edm::EDAnalyzer{

public:

  HitParentTest(const edm::ParameterSet& ps);
  ~HitParentTest() {}

protected:

  //  void beginJob () {}
  void analyze  (const edm::Event& e, const edm::EventSetup& c);
  void endJob   ();

private:

  /** performs some checks on hits */
  void analyzeHits(const std::vector<PCaloHit> &, int type);
  void analyzeAPDHits(const std::vector<PCaloHit> &, int depth);

  bool simTrackPresent(int id);
  math::XYZTLorentzVectorD getOldestParentVertex(edm::SimTrackContainer::const_iterator track);
  edm::SimTrackContainer::const_iterator parentSimTrack(edm::SimTrackContainer::const_iterator thisTrkItr);
  bool validSimTrack(unsigned int simTkId, edm::SimTrackContainer::const_iterator thisTrkItr);

private:

  std::string    sourceLabel, g4Label, hitLabEB, hitLabEE, hitLabES, hitLabHC;

  /** error and other counters */
  unsigned int                         total_num_apd_hits_seen[2];
  unsigned int                         num_apd_hits_no_parent[2];

  /** number of apd hits for which the parent sim track id was not found in
      the simtrack collection. */
  unsigned int                         num_apd_hits_no_simtrack[2];

  /** number of APD hits for which no generator particle was found */
  unsigned int                         num_apd_hits_no_gen_particle[2];

  edm::Handle<edm::SimTrackContainer>  SimTk;
  edm::Handle<edm::SimVertexContainer> SimVtx;

  /** 'histogram' of types of particles going through the APD. Maps from numeric particle code
      to the number of occurences. */
  std::map<int, unsigned>              particle_type_count;

  std::string                          detector[7];
  bool                                 histos;
  unsigned int                         totalHits[7], noParent[7];
  unsigned int                         noSimTrack[7], noGenParticle[7];
  TH1F                                 *hitType[7], *hitRho[7], *hitZ[7];
};

#endif
