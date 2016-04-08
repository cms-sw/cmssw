/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors:
*   Hubert Niewiadomski
*   Jan Kašpar (jan.kaspar@gmail.com)
*
****************************************************************************/

#ifndef DataFormats_TotemRPReco_TotemTotemRPRecHit
#define DataFormats_TotemRPReco_TotemTotemRPRecHit

/**
 *\brief Reconstructed hit in TOTEM RP.
 *
 * Basically a cluster (TotemRPCluster), the position of which has been converted into actual geometry (in mm).
 **/
class TotemRPRecHit
{
 public:
  TotemRPRecHit(unsigned int det_id, double position, double sigma) : det_id_(det_id), 
    position_(position), sigma_(sigma) {}
  TotemRPRecHit() : det_id_(0), position_(0), sigma_(0) {}

  inline void Position(double position) {position_=position;}
  inline double Position() const {return position_;}

  inline void Sigma(double sigma) {sigma_=sigma;}
  inline double Sigma() const {return sigma_;}

  inline void DetId(unsigned int det_id) {det_id_=det_id;}
  inline unsigned int DetId() const {return det_id_;}

  inline TotemRPRecHit *clone() const {return new TotemRPRecHit(*this); }

 private:
  unsigned int det_id_;    ///< the raw ID of detector
  double position_;   ///< position of the hit in mm, wrt detector center (see RPTopology::GetHitPositionInReadoutDirection)
  double sigma_;      ///< position uncertainty, in mm
};



inline bool operator<(const TotemRPRecHit &in1, const TotemRPRecHit &in2)
{
  if(in1.DetId() < in2.DetId())
    return true;
  else if(in1.DetId() == in2.DetId())
    return in1.Position() < in2.Position();
  else 
    return false;
}

#endif

