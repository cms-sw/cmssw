#ifndef SimTransport_Hector_h
#define SimTransport_Hector_h

// HepMC headers
#include "HepMC/GenEvent.h"
#include "HepMC/GenParticle.h"

// SimpleConfigurable replacement
#include "FWCore/ParameterSet/interface/ParameterSet.h"

//Hector headers
#include "H_BeamLine.h"
#include "H_RecRPObject.h"
#include "H_BeamParticle.h"
#include <string>


class Hector {

public:
  Hector(const edm::ParameterSet & ps);
  //  Hector();
  virtual ~Hector();

  /*!Clears all data, prepares Hector for a next event*/
  void clear();
  /*! \brief Sets IP position.
      
      \a x and \a y are given in \f$\mu m\f$, \a z is in meters
  */
  void setIP( double ip_x = 0, double ip_y = 0, double ip_z = 0, double ip_time = 0 );
  /*!Adds the particle \a eventParticle to a beamline*/
  unsigned int add( const  HepMC::GenParticle * eventParticle );
  /*!Adds the stable protons from the event \a ev to a beamline*/
  void add( const HepMC::GenEvent * ev );
  /*!Smears the particle parameters if necessary and then propagate the particles through a beamline*/
  void reconstruct();

  /*!Returns the number of the particles in a beamline*/
  // unsigned int getNp() const { return m_beamPart.size(); }
  /*!Return \a true if the particle have interacted with something in a beamline nad \a false if not.
    \param part_n code of a particle (HepMC::GenParticle::barcode())
   */
  //bool isStopped( unsigned int part_n ) const;
  /*!Returns the particle \a x coordinate at \a z position in a beamline
    \param part_n code of a particle (HepMC::GenParticle::barcode())
  */
  //double getX( double z, unsigned int part_n ) const;
  /*!Returns the particle \a y coordinate at \a z position in a beamline
    \param part_n code of a particle (HepMC::GenParticle::barcode())
  */
  //double getY( double z, unsigned int part_n ) const;
  //
  /*!Returns reconstructed in RP420 x coordinate of particle \a part_n at the IP
    \param part_n code of a particle (HepMC::GenParticle::barcode())
  */
  //float getSimIPX( unsigned int part_n ) const;
  /*!Returns reconstructed in RP420 y coordinate of particle \a part_n at the IP
    \param part_n code of a particle (HepMC::GenParticle::barcode())
  */
  //float getSimIPY( unsigned int part_n ) const;
  /*!Returns reconstructed in RP420 \a tx (\f$\theta_x\f$) of particle \a part_n at the IP
    \param part_n code of a particle (HepMC::GenParticle::barcode())
  */
  //float getSimIPTX( unsigned int part_n ) const;
  /*!Returns reconstructed in RP420 \a ty (\f$\theta_y\f$)of particle \a part_n at the IP
    \param part_n code of a particle (HepMC::GenParticle::barcode())
  */
  //float getSimIPTY( unsigned int part_n ) const;
  /*!Returns reconstructed in RP420 dE ( 7000. - E ) of particle \a part_n at the IP
    \param part_n code of a particle (HepMC::GenParticle::barcode())
  */
  //float getSimIPdE( unsigned int part_n ) const;
  //
  /*!Returns generated x coordinate of particle \a part_n at the IP
    \param part_n code of a particle (HepMC::GenParticle::barcode())
  */
  //loat getGenIPX( unsigned int part_n ) const;
  /*!Returns generated y coordinate of particle \a part_n at the IP
    \param part_n code of a particle (HepMC::GenParticle::barcode())
  */
  //float getGenIPY( unsigned int part_n ) const;
  /*!Returns generated \a tx (\f$\theta_x\f$) of particle \a part_n at the IP
    \param part_n code of a particle (HepMC::GenParticle::barcode())
  */
  //float getGenIPTX( unsigned int part_n ) const;
  /*!Returns generated \a ty (\f$\theta_y\f$) of particle \a part_n at the IP
    \param part_n code of a particle (HepMC::GenParticle::barcode())
  */
  //float getGenIPTY( unsigned int part_n ) const;
  /*!Returns generated energy of particle \a part_n at the IP
    \param part_n code of a particle (HepMC::GenParticle::barcode())
  */
  //float getGenIPE( unsigned int part_n ) const;

  /*!Returns the direction of fly for the particle \a n, 1 in case of forward, -1 in case of backward
    \param part_n code of a particle (HepMC::GenParticle::barcode())
  */

  int getDirect( unsigned int part_n ) const;

  /*!Prints properties of all particles in a beamline*/
  void print() const;
  /*!Return vector of the particle lines (HepMC::GenParticle::barcode()) in a beamline*/
  std::vector<unsigned int> part_list() const;

  HepMC::GenEvent * addPartToHepMC( HepMC::GenEvent * event );

  /*  
private:
  //  edm::ParameterSet m_pBeamLine;
  */
private:
  bool m_verbosity;

  bool m_smearPos;
  bool m_smearS;
  bool m_smearAng;
  bool m_smearE;

  float m_rpp420_f;
  float m_rpp420_b;
  float m_rpp220_f;
  float m_rpp220_b;

  double m_sig_e;

  float m_shiftX;
  float m_shiftY;
  float m_shiftZ;

  
  /*
  std::map<unsigned int, float> m_tx0;
  std::map<unsigned int, float> m_ty0;
  std::map<unsigned int, float> m_x0;
  std::map<unsigned int, float> m_y0;
  std::map<unsigned int, float> m_sim_e;
*/

  /*
  std::map<unsigned int, float> m_gen_x;
  std::map<unsigned int, float> m_gen_y;
  std::map<unsigned int, float> m_gen_tx;
  std::map<unsigned int, float> m_gen_ty;
  std::map<unsigned int, float> m_gen_e;
*/
  double m_IPx;
  double m_IPy;
  double m_IPz;
  double m_IPt;

  // Hector
  H_BeamLine * m_beamline1;
  H_BeamLine * m_beamline2;
  //
//  H_RecRPObject * m_rp220_f;
//  H_RecRPObject * m_rp220_b;

    H_RecRPObject * m_rp420_f;
    H_RecRPObject * m_rp420_b;

  std::map<unsigned int, H_BeamParticle*> m_beamPart;
  std::map<unsigned int, int> m_direct;
  std::map<unsigned int, bool> m_isStopped;
  std::map<unsigned int, double> m_xAtRP420;
  std::map<unsigned int, double> m_yAtRP420;
  std::map<unsigned int, double> m_TxAtRP420;
  std::map<unsigned int, double> m_TyAtRP420;
  std::map<unsigned int, double> m_eAtRP420;

  string beam1filename;
  string beam2filename;
};
#endif
