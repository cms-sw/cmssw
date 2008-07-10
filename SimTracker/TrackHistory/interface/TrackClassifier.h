/*
 *  TrackClassifier.h
 *
 *  Created by Victor Eduardo Bazterra on 5/29/07.
 *  Copyright 2007 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef TrackClassifier_h
#define TrackClassifier_h

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"

#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"

#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"

#include "SimTracker/TrackHistory/interface/TrackCategories.h"
#include "SimTracker/TrackHistory/interface/TrackHistory.h"

#include "TrackingTools/PatternTools/interface/TSCPBuilderNoMaterial.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"


//! Get track history and classify it in function of their .
class TrackClassifier {

public:
  
  //! Constructor by ParameterSet
  TrackClassifier( edm::ParameterSet const & pset);

  //! Pre-process event information (for accessing reconstraction information)
  void newEvent(edm::Event const &, edm::EventSetup const &);
  
  //! Classify the RecoTrack in categories.
  TrackClassifier const & evaluate (edm::RefToBase<reco::Track> const &);

  //! Classify the TrackingParticle in categories.
  TrackClassifier const & evaluate (TrackingParticleRef const &);

  //! Classify the RecoTrack in categories.
  TrackClassifier const & evaluate (reco::TrackRef const & track)
  { 
  	return evaluate( edm::RefToBase<reco::Track>(track) );
  }

  //! Returns track flag for a given category.
  bool is(TrackCategories::Category category) const
  {
    return flags_[category];
  }

  //! Returns track flags with the categories description.
  const TrackCategories::Flags & flags() const
  {
    return flags_;
  }

  //! Returns a reference to the track history used in the classification.
  TrackHistory const & history() const
  {
    return tracer_;
  }

private:

  double badD0Pull_;

  double longLivedDecayLenght_;

  double vertexClusteringSqDistance_;

private:

  struct G4 { 
    enum Process {
      Undefined = 0,
      Unknown,
      Primary,
      Hadronic,
      Decay,
      Compton,
      Annihilation,
      EIoni,
      HIoni,
      MuIoni,
      Photon,
      MuPairProd,
      Conversions,
      EBrem,
      SynchrotronRadiation,
      MuBrem,
      MuNucl
    };
  };
 
  TrackCategories::Flags flags_;
  
  TrackHistory tracer_;

  edm::ESHandle<MagneticField> magneticField_;
  
  edm::Handle<edm::HepMCProduct> mcInformation_;

  edm::ESHandle<ParticleDataTable> particleDataTable_;

  edm::ESHandle<TransientTrackBuilder> transientTrackBuilder_;

  //! Reset the categories flags.
  void reset()
  {
    flags_ = TrackCategories::Flags(TrackCategories::Unknown + 1, false);
  }

  //! Classify all the tracks by their association and reconstruction information
  void reconstructionInformation(edm::RefToBase<reco::Track> const &);

  //! Get all the information related to the simulation details
  void simulationInformation();
  
  //! Get hadron flavor of the initial hadron
  void hadronFlavor();
  
  //! Get all the information related to decay process
  void decayProcesses();
  
  //! Get information about conversion and other interactions
  void conversionInteraction();
  
  //! Get geometrical information about the vertexes
  void vertexInformation();

  // Check for unkown classification
  void unknownTrack();
    
  //! Auxiliary class holding simulated primary vertices
  struct GeneratedPrimaryVertex {
    
    GeneratedPrimaryVertex(double x1,double y1,double z1): x(x1), y(y1), z(z1), ptsq(0), nGenTrk(0){}

    bool operator< ( GeneratedPrimaryVertex const & reference) const
    {
      return ptsq < reference.ptsq;
    }
    
    double x, y, z;
    double ptsq;
    int nGenTrk;
    
    HepMC::FourVector ptot;
    
    std::vector<int> finalstateParticles;
    std::vector<int> simTrackIndex;
    std::vector<int> genVertex;
  };
  
  std::vector<GeneratedPrimaryVertex> genpvs_;

  // Auxiliary function to get the generated primary vertex
  bool isFinalstateParticle(const HepMC::GenParticle *);
  bool isCharged(const HepMC::GenParticle *);  
  void genPrimaryVertexes();

};

#endif
