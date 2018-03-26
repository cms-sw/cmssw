#ifndef EcalEBTPCluster_h
#define EcalEBTPCluster_h
/** \class EcalEBTPCluster
 * forPhase II 
 * As of now we do not know yet how the electronics would look like
 * so for now we build some machinery to produce TPs which are taken from the RecHits
 *
 ************************************************************/
#include <sys/time.h>
#include <iostream>
#include <vector>
#include <map>
#include <utility>
#include "SimCalorimetry/EcalEBTrigPrimAlgos/interface/EcalEBTPCluster.h"


class EcalEBTPCluster {
  private:
    float eta_;
    float phi_;
    float et_;
    float energy_;

    EBDetId id_;

  public:
    typedef edm::SortedCollection<EcalEBTPCluster> EcalEBTrigPrimClusterCollection;
    EcalEBTPCluster(){ energy_=0;} // for persistence
    ~EcalEBTPCluster(){;} // for persistence
    EcalEBTPCluster(const EcalEBTPCluster & rh) {
      id_ = rh.id_;
      eta_ = rh.eta_;
      phi_ = rh.phi_;
      et_ = rh.et_;
      energy_ = rh.energy_;
    };
    EcalEBTPCluster(const EBDetId& id){ 
     id_ = id; 
    }
    const EBDetId& id() const { return id_; }
    typedef EBDetId key_type; ///< For the sorted collection
    void SetEta(float eta){eta_ = eta;}
    void SetPhi(float phi){phi_ = phi;}
    void SetEt(float et){et_ = et;}
    void SetEnergy(float energy){energy_ = energy;}
    float Eta() {return eta_;}
    float Phi() {return phi_;}
    float Et() {return  eta_;}
    float Energy() {return energy_;}

};

#endif
