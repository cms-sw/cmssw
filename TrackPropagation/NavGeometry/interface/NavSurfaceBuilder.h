#ifndef NavSurfaceBuilder_H
#define NavSurfaceBuilder_H

class NavSurface;
class Surface;

/// helper: builde a NavSurface for a Surface
class NavSurfaceBuilder {
public:

  NavSurface* build( const Surface& surface) const;

};

#endif 
