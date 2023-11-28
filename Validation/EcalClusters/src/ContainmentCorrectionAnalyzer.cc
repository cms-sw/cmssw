/*\class to compute not containment parameter */

#include "FWCore/Framework/interface/MakerMacros.h"
#include "Validation/EcalClusters/interface/ContainmentCorrectionAnalyzer.h"

using namespace cms;
using namespace edm;
using namespace std;
using namespace reco;

ContainmentCorrectionAnalyzer::ContainmentCorrectionAnalyzer(const ParameterSet &ps) : pTopologyToken(esConsumes()) {
  BarrelSuperClusterCollection_ =
      consumes<reco::SuperClusterCollection>(ps.getParameter<InputTag>("BarrelSuperClusterCollection"));
  EndcapSuperClusterCollection_ =
      consumes<reco::SuperClusterCollection>(ps.getParameter<InputTag>("EndcapSuperClusterCollection"));
  reducedBarrelRecHitCollection_ =
      consumes<EcalRecHitCollection>(ps.getParameter<InputTag>("reducedBarrelRecHitCollection"));
  reducedEndcapRecHitCollection_ =
      consumes<EcalRecHitCollection>(ps.getParameter<InputTag>("reducedEndcapRecHitCollection"));
  SimTrackCollection_ = consumes<SimTrackContainer>(ps.getParameter<InputTag>("simTrackCollection"));
  SimVertexCollection_ = consumes<SimVertexContainer>(ps.getParameter<InputTag>("simVertexCollection"));
}

ContainmentCorrectionAnalyzer::~ContainmentCorrectionAnalyzer() {}

void ContainmentCorrectionAnalyzer::beginJob() {
  Service<TFileService> fs;

  // Define reference histograms
  h_EB_eRecoEtrueReference = fs->make<TH1F>("EB_eRecoEtrueReference", "EB_eRecoEtrueReference", 440, 0., 1.1);
  h_EB_e9EtrueReference = fs->make<TH1F>("EB_e9EtrueReference", "EB_e9EtrueReference", 440, 0., 1.1);
  h_EB_e25EtrueReference = fs->make<TH1F>("EB_e25EtrueReference", "EB_e25EtrueReference", 440, 0., 1.1);
  h_EE_eRecoEtrueReference = fs->make<TH1F>("EE_eRecoEtrueReference", "EE_eRecoEtrueReference", 440, 0., 1.1);
  h_EE_e9EtrueReference = fs->make<TH1F>("EE_e9EtrueReference", "EE_e9EtrueReference", 440, 0., 1.1);
  h_EE_e25EtrueReference = fs->make<TH1F>("EE_e25EtrueReference", "EE_e25EtrueReference", 440, 0., 1.1);
  h_EB_eTrue = fs->make<TH1F>("EB_eTrue", "EB_eTrue", 41, 40., 60.);
  h_EE_eTrue = fs->make<TH1F>("EE_eTrue", "EE_eTrue", 41, 40., 60.);
  h_EB_converted = fs->make<TH1F>("EB_converted", "EB_converted", 2, 0., 2.);
  h_EE_converted = fs->make<TH1F>("EE_converted", "EE_converted", 2, 0., 2.);
}

void ContainmentCorrectionAnalyzer::analyze(const Event &evt, const EventSetup &es) {
  LogInfo("ContainmentCorrectionAnalyzer") << "Analyzing event " << evt.id() << "\n";

  // taking the needed collections
  std::vector<SimTrack> theSimTracks;
  Handle<SimTrackContainer> SimTk;
  evt.getByToken(SimTrackCollection_, SimTk);
  Labels l;
  labelsForToken(SimTrackCollection_, l);

  if (SimTk.isValid())
    theSimTracks.insert(theSimTracks.end(), SimTk->begin(), SimTk->end());
  else {
    LogError("ContainmentCorrectionAnalyzer") << "Error! can't get collection with label " << l.module;
  }

  std::vector<SimVertex> theSimVertexes;
  Handle<SimVertexContainer> SimVtx;
  evt.getByToken(SimVertexCollection_, SimVtx);
  labelsForToken(SimVertexCollection_, l);

  if (SimVtx.isValid())
    theSimVertexes.insert(theSimVertexes.end(), SimVtx->begin(), SimVtx->end());
  else {
    LogError("ContainmentCorrectionAnalyzer") << "Error! can't get collection with label " << l.module;
  }

  const reco::SuperClusterCollection *BarrelSuperClusters = nullptr;
  Handle<reco::SuperClusterCollection> pHybridBarrelSuperClusters;
  evt.getByToken(BarrelSuperClusterCollection_, pHybridBarrelSuperClusters);
  labelsForToken(BarrelSuperClusterCollection_, l);

  if (pHybridBarrelSuperClusters.isValid()) {
    BarrelSuperClusters = pHybridBarrelSuperClusters.product();
  } else {
    LogError("ContainmentCorrectionAnalyzer") << "Error! can't get collection with label " << l.module;
  }

  const reco::SuperClusterCollection *EndcapSuperClusters = nullptr;
  Handle<reco::SuperClusterCollection> pMulti5x5EndcapSuperClusters;
  evt.getByToken(EndcapSuperClusterCollection_, pMulti5x5EndcapSuperClusters);
  labelsForToken(EndcapSuperClusterCollection_, l);

  if (pMulti5x5EndcapSuperClusters.isValid())
    EndcapSuperClusters = pMulti5x5EndcapSuperClusters.product();
  else {
    LogError("ContainmentCorrectionAnalyzer") << "Error! can't get collection with label " << l.module;
  }

  const EcalRecHitCollection *ebRecHits = nullptr;
  Handle<EcalRecHitCollection> pEBRecHits;
  evt.getByToken(reducedBarrelRecHitCollection_, pEBRecHits);
  labelsForToken(reducedBarrelRecHitCollection_, l);

  if (pEBRecHits.isValid())
    ebRecHits = pEBRecHits.product();
  else {
    LogError("ContainmentCorrectionAnalyzer") << "Error! can't get collection with label " << l.module;
  }

  const EcalRecHitCollection *eeRecHits = nullptr;
  Handle<EcalRecHitCollection> pEERecHits;
  evt.getByToken(reducedEndcapRecHitCollection_, pEERecHits);
  labelsForToken(reducedEndcapRecHitCollection_, l);

  if (pEERecHits.isValid())
    eeRecHits = pEERecHits.product();
  else {
    LogError("ContainmentCorrectionAnalyzer") << "Error! can't get collection with label " << l.module;
  }

  const CaloTopology *topology = nullptr;
  auto pTopology = es.getHandle(pTopologyToken);
  if (pTopology.isValid())
    topology = &es.getData(pTopologyToken);

  std::vector<EcalSimPhotonMCTruth> photons = findMcTruth(theSimTracks, theSimVertexes);

  nMCphotons = 0;
  nRECOphotons = 0;

  int mc_size = photons.size();
  mcEnergy.resize(mc_size);
  mcEta.resize(mc_size);
  mcPhi.resize(mc_size);
  mcPt.resize(mc_size);
  isConverted.resize(mc_size);
  x_vtx.resize(mc_size);
  y_vtx.resize(mc_size);
  z_vtx.resize(mc_size);

  superClusterEnergy.resize(mc_size);
  superClusterEta.resize(mc_size);
  superClusterPhi.resize(mc_size);
  superClusterEt.resize(mc_size);
  e1.resize(mc_size);
  e9.resize(mc_size);
  e25.resize(mc_size);
  seedXtal.resize(mc_size);
  iMC.resize(mc_size);

  // loop over MC truth photons
  for (unsigned int ipho = 0; ipho < photons.size(); ipho++) {
    math::XYZTLorentzVectorD vtx = photons[ipho].primaryVertex();
    double phiTrue = photons[ipho].fourMomentum().phi();
    double vtxPerp = sqrt(vtx.x() * vtx.x() + vtx.y() * vtx.y());
    double etaTrue = ecalEta(photons[ipho].fourMomentum().eta(), vtx.z(), vtxPerp);
    double etTrue = photons[ipho].fourMomentum().e() / cosh(etaTrue);
    nMCphotons++;
    mcEnergy[nMCphotons - 1] = photons[ipho].fourMomentum().e();
    mcEta[nMCphotons - 1] = etaTrue;
    mcPhi[nMCphotons - 1] = phiTrue;
    mcPt[nMCphotons - 1] = etTrue;
    isConverted[nMCphotons - 1] = photons[ipho].isAConversion();
    x_vtx[nMCphotons - 1] = vtx.x();
    y_vtx[nMCphotons - 1] = vtx.y();
    z_vtx[nMCphotons - 1] = vtx.z();

    // check histos for MC truth
    if (std::fabs(etaTrue) < 1.479) {
      h_EB_eTrue->Fill(photons[ipho].fourMomentum().e());
      h_EB_converted->Fill(photons[ipho].isAConversion());
    }
    if (std::fabs(etaTrue) >= 1.6) {
      h_EE_eTrue->Fill(photons[ipho].fourMomentum().e());
      h_EE_converted->Fill(photons[ipho].isAConversion());
    }

    // barrel
    if (std::fabs(etaTrue) < 1.479) {
      double etaCurrent;  // , etaFound = 0; // UNUSED
      double phiCurrent;
      // double etCurrent,  etFound  = 0; // UNUSED
      const reco::SuperCluster *nearSC = nullptr;

      double closestParticleDistance = 999;
      for (reco::SuperClusterCollection::const_iterator aClus = BarrelSuperClusters->begin();
           aClus != BarrelSuperClusters->end();
           aClus++) {
        etaCurrent = aClus->position().eta();
        phiCurrent = aClus->position().phi();
        // etCurrent  = aClus->energy()/std::cosh(etaCurrent); // UNUSED
        double deltaR = std::sqrt(std::pow(etaCurrent - etaTrue, 2) + std::pow(phiCurrent - phiTrue, 2));
        if (deltaR < closestParticleDistance) {
          // etFound  = etCurrent; // UNUSED
          // etaFound = etaCurrent; // UNUSED
          closestParticleDistance = deltaR;
          nearSC = &(*aClus);
        }
      }

      if (closestParticleDistance < 0.3) {
        // Is a matched particle dumping informations
        nRECOphotons++;
        superClusterEnergy[nRECOphotons - 1] = nearSC->rawEnergy();
        superClusterEta[nRECOphotons - 1] = nearSC->position().eta();
        superClusterPhi[nRECOphotons - 1] = nearSC->position().phi();
        superClusterEt[nRECOphotons - 1] = nearSC->rawEnergy() / std::cosh(nearSC->position().eta());
        iMC[nRECOphotons - 1] = nMCphotons - 1;

        const reco::CaloClusterPtr &theSeed = nearSC->seed();
        seedXtal[nRECOphotons - 1] = EcalClusterTools::getMaximum(*theSeed, ebRecHits).first;
        e1[nRECOphotons - 1] = EcalClusterTools::eMax(*theSeed, ebRecHits);
        e9[nRECOphotons - 1] = EcalClusterTools::e3x3(*theSeed, ebRecHits, topology);
        e25[nRECOphotons - 1] = EcalClusterTools::e5x5(*theSeed, ebRecHits, topology);
      }
    }

    // endcap
    if (std::fabs(etaTrue) >= 1.6) {
      double etaCurrent;  // , etaFound = 0; // UNUSED
      double phiCurrent;
      // double etCurrent,  etFound  = 0; // UNUSED
      const reco::SuperCluster *nearSC = nullptr;

      double closestParticleDistance = 999;
      for (reco::SuperClusterCollection::const_iterator aClus = EndcapSuperClusters->begin();
           aClus != EndcapSuperClusters->end();
           aClus++) {
        etaCurrent = aClus->position().eta();
        phiCurrent = aClus->position().phi();
        // etCurrent  =  aClus->energy()/std::cosh(etaCurrent);
        double deltaR = std::sqrt(std::pow(etaCurrent - etaTrue, 2) + std::pow(phiCurrent - phiTrue, 2));
        if (deltaR < closestParticleDistance) {
          // etFound  = etCurrent; // UNUSED
          // etaFound = etaCurrent; // UNUSED
          closestParticleDistance = deltaR;
          nearSC = &(*aClus);
        }
      }

      if (closestParticleDistance < 0.3) {
        // Is a matched particle dumping informations
        nRECOphotons++;
        float psEnergy = nearSC->preshowerEnergy();
        superClusterEnergy[nRECOphotons - 1] = nearSC->rawEnergy() + psEnergy;
        superClusterEta[nRECOphotons - 1] = nearSC->position().eta();
        superClusterPhi[nRECOphotons - 1] = nearSC->position().phi();
        superClusterEt[nRECOphotons - 1] = (nearSC->rawEnergy() + psEnergy) / std::cosh(nearSC->position().eta());
        iMC[nRECOphotons - 1] = nMCphotons - 1;

        const reco::CaloClusterPtr &theSeed = nearSC->seed();
        seedXtal[nRECOphotons - 1] = EcalClusterTools::getMaximum(*theSeed, eeRecHits).first;
        e1[nRECOphotons - 1] = EcalClusterTools::eMax(*theSeed, eeRecHits) + psEnergy;
        e9[nRECOphotons - 1] = EcalClusterTools::e3x3(*theSeed, eeRecHits, topology) + psEnergy;
        e25[nRECOphotons - 1] = EcalClusterTools::e5x5(*theSeed, eeRecHits, topology) + psEnergy;
      }
    }
  }

  // containment analysis for unconverted photons in the reference region only
  for (int i = 0; i < nRECOphotons; i++) {
    // barrel
    if (fabs(superClusterEta[i]) < 1.479) {
      if (isConverted[iMC[i]] != 1) {
        int ietaAbs = (seedXtal[i] >> 9) & 0x7F;
        int iphi = seedXtal[i] & 0x1FF;
        if (ietaAbs > 5 && ietaAbs < 21 && ((iphi % 20) > 5) && ((iphi % 20) < 16)) {
          h_EB_eRecoEtrueReference->Fill(superClusterEnergy[i] / mcEnergy[iMC[i]]);
          h_EB_e9EtrueReference->Fill(e9[i] / mcEnergy[iMC[i]]);
          h_EB_e25EtrueReference->Fill(e25[i] / mcEnergy[iMC[i]]);
        }
      }
    }

    // endcap
    if (fabs(superClusterEta[i]) > 1.6) {
      if (isConverted[iMC[i]] != 1) {
        if (fabs(superClusterEta[i]) > 1.7 && fabs(superClusterEta[i] < 2.3) &&
            ((superClusterPhi[i] > -CLHEP::pi / 2. + 0.1 && superClusterPhi[i] < CLHEP::pi / 2. - 0.1) ||
             (superClusterPhi[i] > CLHEP::pi / 2. + 0.1) || (superClusterPhi[i] < -CLHEP::pi / 2. - 0.1))) {
          h_EE_eRecoEtrueReference->Fill(superClusterEnergy[i] / mcEnergy[iMC[i]]);
          h_EE_e9EtrueReference->Fill(e9[i] / mcEnergy[iMC[i]]);
          h_EE_e25EtrueReference->Fill(e25[i] / mcEnergy[iMC[i]]);
        }
      }
    }

  }  // loop over reco photons
}

void ContainmentCorrectionAnalyzer::endJob() {}

float ContainmentCorrectionAnalyzer::ecalEta(float EtaParticle, float Zvertex, float plane_Radius) {
  const float R_ECAL = 136.5;
  const float Z_Endcap = 328.0;
  const float etaBarrelEndcap = 1.479;

  if (EtaParticle != 0.) {
    float Theta = 0.0;
    float ZEcal = (R_ECAL - plane_Radius) * sinh(EtaParticle) + Zvertex;

    if (ZEcal != 0.0)
      Theta = atan(R_ECAL / ZEcal);
    if (Theta < 0.0)
      Theta = Theta + Geom::pi();

    float ETA = -log(tan(0.5 * Theta));

    if (fabs(ETA) > etaBarrelEndcap) {
      float Zend = Z_Endcap;
      if (EtaParticle < 0.0)
        Zend = -Zend;
      float Zlen = Zend - Zvertex;
      float RR = Zlen / sinh(EtaParticle);
      Theta = atan((RR + plane_Radius) / Zend);
      if (Theta < 0.0)
        Theta = Theta + Geom::pi();
      ETA = -log(tan(0.5 * Theta));
    }

    return ETA;
  } else {
    LogWarning("") << "[ContainmentCorrectionAnalyzer::ecalEta] Warning: Eta "
                      "equals to zero, not correcting";
    return EtaParticle;
  }
}

// taken from an old version of RecoEgamma/EgammaMCTools/src/PhotonMCTruthFinder
std::vector<EcalSimPhotonMCTruth> ContainmentCorrectionAnalyzer::findMcTruth(std::vector<SimTrack> &theSimTracks,
                                                                             std::vector<SimVertex> &theSimVertices) {
  std::vector<EcalSimPhotonMCTruth> result;

  geantToIndex_.clear();
  // int   idTrk1_[10]; // UNUSED
  // int   idTrk2_[10]; // UNUSED

  // Local variables
  // const int SINGLE=1; // UNUSED
  // const int DOUBLE=2; // UNUSED
  // const int PYTHIA=3; // UNUSED
  const int ELECTRON_FLAV = 1;
  const int PIZERO_FLAV = 2;
  const int PHOTON_FLAV = 3;

  // int ievtype=0; // UNUSED
  int ievflav = 0;
  std::vector<SimTrack *> photonTracks;
  std::vector<SimTrack *> pizeroTracks;
  std::vector<const SimTrack *> trkFromConversion;
  SimVertex primVtx;
  std::vector<int> convInd;

  fillMcTruth(theSimTracks, theSimVertices);
  int iPV = -1;
  int partType1 = 0;
  int partType2 = 0;
  std::vector<SimTrack>::iterator iFirstSimTk = theSimTracks.begin();
  if (!(*iFirstSimTk).noVertex()) {
    iPV = (*iFirstSimTk).vertIndex();
    int vtxId = (*iFirstSimTk).vertIndex();
    primVtx = theSimVertices[vtxId];
    partType1 = (*iFirstSimTk).type();
  }

  // Look at a second track
  iFirstSimTk++;
  if (iFirstSimTk != theSimTracks.end()) {
    if ((*iFirstSimTk).vertIndex() == iPV) {
      partType2 = (*iFirstSimTk).type();
    }
  }
  int npv = 0;
  int iPho = 0;
  for (std::vector<SimTrack>::iterator iSimTk = theSimTracks.begin(); iSimTk != theSimTracks.end(); ++iSimTk) {
    if ((*iSimTk).noVertex())
      continue;
    // int vertexId = (*iSimTk).vertIndex(); // UNUSED
    // SimVertex vertex = theSimVertices[vertexId]; // UNUSED
    if ((*iSimTk).vertIndex() == iPV) {
      npv++;
      if ((*iSimTk).type() == 22) {
        convInd.push_back(0);
        photonTracks.push_back(&(*iSimTk));
        // math::XYZTLorentzVectorD momentum = (*iSimTk).momentum(); // UNUSED
      }
    }
  }

  if (npv > 4) {  // ievtype = PYTHIA; // UNUSED
  } else if (npv == 1) {
    if (abs(partType1) == 11) { /* ievtype = SINGLE; ==UNUSED== */
      ievflav = ELECTRON_FLAV;
    } else if (partType1 == 111) { /* ievtype = SINGLE; ==UNUSED== */
      ievflav = PIZERO_FLAV;
    } else if (partType1 == 22) { /* ievtype = SINGLE; ==UNUSED== */
      ievflav = PHOTON_FLAV;
    }
  } else if (npv == 2) {
    if (abs(partType1) == 11 && abs(partType2) == 11) { /* ievtype = DOUBLE; ==UNUSED== */
      ievflav = ELECTRON_FLAV;
    } else if (partType1 == 111 && partType2 == 111) { /* ievtype = DOUBLE; ==UNUSED== */
      ievflav = PIZERO_FLAV;
    } else if (partType1 == 22 && partType2 == 22) { /* ievtype = DOUBLE; ==UNUSED== */
      ievflav = PHOTON_FLAV;
    }
  }

  //  Look into converted photons
  int isAconversion = 0;
  if (ievflav == PHOTON_FLAV) {
    int nConv = 0;
    iPho = 0;
    for (std::vector<SimTrack *>::iterator iPhoTk = photonTracks.begin(); iPhoTk != photonTracks.end(); ++iPhoTk) {
      trkFromConversion.clear();
      for (std::vector<SimTrack>::iterator iSimTk = theSimTracks.begin(); iSimTk != theSimTracks.end(); ++iSimTk) {
        if ((*iSimTk).noVertex())
          continue;
        if ((*iSimTk).vertIndex() == iPV)
          continue;
        if (abs((*iSimTk).type()) != 11)
          continue;
        int vertexId = (*iSimTk).vertIndex();
        SimVertex vertex = theSimVertices[vertexId];
        int motherId = -1;
        if (vertex.parentIndex()) {
          unsigned motherGeantId = vertex.parentIndex();
          std::map<unsigned, unsigned>::iterator association = geantToIndex_.find(motherGeantId);
          if (association != geantToIndex_.end())
            motherId = association->second;
          // int motherType = motherId == -1 ? 0 :
          // theSimTracks[motherId].type();

          if (theSimTracks[motherId].trackId() == (*iPhoTk)->trackId()) {
            /// store this electron since it's from a converted photon
            trkFromConversion.push_back(&(*iSimTk));
          }
        }
      }  // loop over the SimTracks

      if (!trkFromConversion.empty()) {
        isAconversion = 1;
        nConv++;
        convInd[iPho] = nConv;
        int convVtxId = trkFromConversion[0]->vertIndex();
        SimVertex convVtx = theSimVertices[convVtxId];
        const math::XYZTLorentzVectorD &vtxPosition = convVtx.position();
        // math::XYZTLorentzVectorD momentum = (*iPhoTk)->momentum(); // UNUSED

        result.push_back(EcalSimPhotonMCTruth(isAconversion,
                                              (*iPhoTk)->momentum(),
                                              vtxPosition.pt(),
                                              vtxPosition.z(),
                                              vtxPosition,
                                              primVtx.position(),
                                              trkFromConversion));
      } else {
        isAconversion = 0;
        math::XYZTLorentzVectorD vtxPosition(0., 0., 0., 0.);
        result.push_back(EcalSimPhotonMCTruth(isAconversion,
                                              (*iPhoTk)->momentum(),
                                              vtxPosition.pt(),
                                              vtxPosition.z(),
                                              vtxPosition,
                                              primVtx.position(),
                                              trkFromConversion));
      }
      iPho++;
    }  // loop over the primary photons
  }    // Event with one or two photons

  return result;
}

void ContainmentCorrectionAnalyzer::fillMcTruth(std::vector<SimTrack> &simTracks, std::vector<SimVertex> &simVertices) {
  unsigned nVtx = simVertices.size();
  unsigned nTks = simTracks.size();
  if (nVtx == 0)
    return;
  // create a map associating geant particle id and position in the event
  // SimTrack vector
  for (unsigned it = 0; it < nTks; ++it) {
    geantToIndex_[simTracks[it].trackId()] = it;
  }
}
