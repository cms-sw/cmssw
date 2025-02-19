#ifndef SurfaceSideDefinition_h_
#define SurfaceSideDefinition_h_

/** Defines side of surface for use in TrajectoryStateOnSurface
 *  and related classes
 */
namespace SurfaceSideDefinition {
  enum SurfaceSide /*: signed short*/ { beforeSurface, afterSurface, atCenterOfSurface };
}
#endif
