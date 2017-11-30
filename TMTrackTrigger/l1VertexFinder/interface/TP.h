#ifndef __TMTrackTrigger_VertexFinder_TP_h__
#define __TMTrackTrigger_VertexFinder_TP_h__


#include "DataFormats/Common/interface/Ptr.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"

#include "TMTrackTrigger/l1VertexFinder/interface/utility.h"



namespace l1tVertexFinder {

typedef edm::Ptr<TrackingParticle> TrackingParticlePtr;

class Settings;
class Stub;


class TP : public TrackingParticlePtr {

public:
  // Fill useful info about tracking particle.
  TP(TrackingParticlePtr tpPtr, unsigned int index_in_vTPs, const Settings* settings);
  ~TP(){}

  // Fill truth info with association from tracking particle to stubs.
  void fillTruth(const std::vector<Stub>& vStubs);

  // == Functions for returning info about tracking particles ===

  // Location in InputData::vTPs_
  unsigned int                           index() const { return     index_in_vTPs_; }
  // Did it come from the main physics collision or from pileup?
   // Basic TP properties
  int                                    pdgId() const { return             pdgId_; }
  bool                        physicsCollision() const { return  physicsCollision_; }
  int                                   charge() const { return            charge_; }
  float                                   mass() const { return              mass_; }
  float                                     pt() const { return                pt_; }
  float                                qOverPt() const { return  (pt_ > 0)  ?  charge_/pt_  :  9.9e9; }
  float                                    eta() const { return               eta_; }
  float                                  theta() const { return             theta_; }
  float                              tanLambda() const { return         tanLambda_; }
  float                                   phi0() const { return              phi0_; }
  // TP production vertex (x,y,z) coordinates.
  float                                     vx() const { return                vx_; }
  float                                     vy() const { return                vy_; }
  float                                     vz() const { return                vz_; }
  // d0 and z0 impact parameters with respect to (x,y) = (0,0).
  float                                     d0() const { return                d0_;}
  float                                     z0() const { return                z0_;}
  float                                    tip() const { return               tip_;}
  // == Functions returning stubs produced by tracking particle.
  unsigned int                   numAssocStubs() const { return assocStubs_.size(); }

  // TP is worth keeping (e.g. for fake rate measurement)
  bool                                     use() const { return               use_; }

  bool                               useForEff() const {return           useForEff_;}
  // TP can be used for algorithmic efficiency measurement (also requires stubs in enough layers).
  bool                            useForAlgEff() const { return      useForAlgEff_; } 
 // TP can be used for vertex reconstruction measuremetn
  bool                        useForVertexReco() const { return   useForVertexReco_;}

private:

  void fillUse();          // Fill the use_ flag.
  void fillUseForEff();    // Fill the useForEff_ flag.
  void fillUseForAlgEff(); // Fill the useforAlgEff_ flag.
  void fillUseForVertexReco();

  // Calculate how many tracker layers this TP has stubs in.
  void calcNumLayers() { nLayersWithStubs_ = utility::countLayers( settings_, assocStubs_, false); }

private:
  unsigned int                      index_in_vTPs_; // location of this TP in InputData::vTPs

  const Settings*                        settings_; // Configuration parameters

  int                                       pdgId_;
  bool                                   inTimeBx_; // TP came from in-time bunch crossing.
  bool                           physicsCollision_; // True if TP from physics collision rather than pileup.
  int                                      charge_;
  float                                      mass_;
  float                                        pt_; // TP kinematics
  float                                       eta_;
  float                                     theta_;
  float                                 tanLambda_;
  float                                      phi0_;
  float                                        vx_; // TP production point.
  float                                        vy_;
  float                                        vz_;
  float                                        d0_; // d0 impact parameter with respect to (x,y) = (0,0)
  float                                        z0_; // z0 impact parameter with respect to (x,y) = (0,0)
  float                                       tip_;
  bool                                        use_; // TP is worth keeping (e.g. for fake rate measurement)
  bool                                  useForEff_; // TP can be used for tracking efficiency measurement.
  bool                               useForAlgEff_; // TP can be used for tracking algorithmic efficiency measurement.
  bool                           useForVertexReco_;

  std::vector<const Stub*>             assocStubs_;
  unsigned int                   nLayersWithStubs_; // Number of tracker layers with stubs from this TP.
};


} // end namespace l1tVertexFinder

#endif
