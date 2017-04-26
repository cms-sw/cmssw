#ifndef _beam_conditions_h_
#define _beam_conditions_h_

#include <cstdio>

enum LHCSector { unknownSector, sector45, sector56 };

//----------------------------------------------------------------------------------------------------

struct BeamConditions
{
  double si_vtx = 10E-6;          // vertex size, m
  double si_beam_div = 20E-6;     // beam divergence, rad

  double vtx0_y_45 = 300E-6;         // vertex offset, sector 45 (beam 2), m
  double vtx0_y_56 = 200E-6;         // vertex offset, sector 56 (beam 1), m

  double half_crossing_angle_45 = +179.394E-6;   // crossing angle, sector 45 (beam 2), rad
  double half_crossing_angle_56 = +191.541E-6;   // crossing angle, sector 56 (beam 1), rad

  void Print() const
  {
    printf(">> BeamConditions\n");
    printf("    si_vtx = %.3E\n", si_vtx);
    printf("    si_beam_div = %.3E\n", si_beam_div);
    printf("    vtx0_y_45 = %.3E\n", vtx0_y_45);
    printf("    vtx0_y_56 = %.3E\n", vtx0_y_56);
    printf("    half_crossing_angle_45 = %.3E\n", half_crossing_angle_45);
    printf("    half_crossing_angle_56 = %.3E\n", half_crossing_angle_56);
  }
};

BeamConditions beamConditions;

#endif
