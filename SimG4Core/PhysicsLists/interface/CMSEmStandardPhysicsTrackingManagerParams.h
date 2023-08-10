#ifndef SimG4Core_PhysicsLists_CMSEmStandardPhysicsTrackingManagerParams_h
#define SimG4Core_PhysicsLists_CMSEmStandardPhysicsTrackingManagerParams_h

struct CMSEmStandardPhysicsTrackingManagerParams {
public:
  double rangeFactor_ = 0.0;
  double geomFactor_ = 0.0;
  double safetyFactor_ = 0.0;
  double lambdaLimit_ = 0.0;
  std::string stepLimit_;
};

#endif
