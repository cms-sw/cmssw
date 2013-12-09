#ifndef MultiTrackValidatorBase_h
#define MultiTrackValidatorBase_h

/** \class MultiTrackValidatorBase
 *  Base class for analyzers that produces histrograms to validate Track Reconstruction performances
 *
 *  $Date: 2012/12/03 11:19:34 $
 *  $Revision: 1.30 $
 *  \author cerati
 */

#include <memory>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "MagneticField/Engine/interface/MagneticField.h" 
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h" 

#include "SimTracker/TrackAssociation/interface/TrackAssociatorBase.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CommonTools/RecoAlgos/interface/RecoTrackSelector.h"
#include "SimGeneral/TrackingAnalysis/interface/TrackingParticleSelector.h"
#include "CommonTools/RecoAlgos/interface/CosmicTrackingParticleSelector.h"

#include <DQMServices/Core/interface/DQMStore.h>

#include <iostream>
#include <sstream>
#include <string>

class MultiTrackValidatorBase {
 public:

  /// Constructor
  MultiTrackValidatorBase(const edm::ParameterSet& pset);
    
  /// Destructor
  virtual ~MultiTrackValidatorBase(){ }
  
  //virtual void initialize()=0;

 protected:

  DQMStore* dbe_;

  // MTV-specific data members
  std::vector<std::string> associators;
  edm::InputTag label_tp_effic;
  edm::InputTag label_tp_fake;
  edm::InputTag label_tv;
  edm::InputTag label_pileupinfo;
  std::string sim;
  std::string parametersDefiner;


  std::vector<edm::InputTag> label;
  edm::InputTag bsSrc;

  std::string out;

  edm::InputTag m_dEdx1Tag;
  edm::InputTag m_dEdx2Tag;

  edm::ESHandle<MagneticField> theMF;
  std::vector<const TrackAssociatorBase*> associator;


  bool ignoremissingtkcollection_;
  bool skipHistoFit;



};


#endif
