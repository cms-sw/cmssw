#include "RecoTracker/TkSeedGenerator/src/FastCircle.cc"



int main() {

{
   GlobalPoint p1( 0., -1.,0.);
   GlobalPoint p2(-1.,  0.,0.);
   GlobalPoint p3( 0.,  1.,0.);

   FastCircle c(p1,p2,p3);

   std::cout << c.x0() << " " << c.y0() << " " << c.rho() 
             << " " << c.n1() << " " << c.n2() << " " << c.c()<< std::endl;

}

{

   GlobalPoint p1( -10.,  0.,0.);
   GlobalPoint p2(   0., 10.,0.);
   GlobalPoint p3(  10.,  0.,0.);

   FastCircle c(p1,p2,p3);

   std::cout << c.x0() << " " << c.y0() << " " << c.rho()
             << " " << c.n1() << " " << c.n2() << " " << c.c()<< std::endl;

}

{

   GlobalPoint p1( -15.,  15.,0.);
   GlobalPoint p2(  -5.,   5.,0.);
   GlobalPoint p3(   5.,  15.,0.);

   FastCircle c(p1,p2,p3);

   std::cout << c.x0() << " " << c.y0() << " " << c.rho()
             << " " << c.n1() << " " << c.n2() << " " << c.c()<< std::endl;

}


   return 0;

}
