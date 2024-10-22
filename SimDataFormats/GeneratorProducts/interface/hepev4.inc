c -------------------------------------------------------------
c hepev4 holds generator level information
c
      double precision eventweightlh, scalelh
      double precision alphaqedlh, alphaqcdlh, spinlh
      integer          icolorflowlh, idruplh
c
      common/hepev4/eventweightlh, alphaqedlh, alphaqcdlh, scalelh(10),
     1              spinlh(3,NMXHEP), icolorflowlh(2,NMXHEP), idruplh
      save /hepev4/

C idruplh            : The identity of the current process, 
C                      as given by the LPRUP codes.
C eventweightlh      : The event weight:  
C                      Equal to (total cross section)/(total generated)
C                      for the output of Pythia, Herwig, etc.
C alphaqedlh         : QED coupling alpha_em.
C alphaqcdlh         : QCD coupling alpha_s.
C scalelh(10)        : Squared Scale Q of the event.
C......Defined for standard 2->1->2 or 2->2 process
C.......kinematics are p1 + p2 -> q1 + q2
C      scalehl(1)= Q2 hard scale (used in PDF and couplings)
C      scalehl(2)= Q2 scale of parton shower
C      scalehl(3)= s-hat, invariant (p1+p2)**2
C      scalehl(4)= t-hat, invariant (p1-q1)**2
C      scalehl(5)= u-hat, invariant (p1-q2)**2
C      scalehl(6)= squared transverse momentum of q1 (i.e., pt-hat**2)
C.......Additionally, for 2->3 processes, p1 + p2 -> q1 + q2 + q3
C      scalehl(7)= squared transverse momentum of q2
C      scalehl(8)= user defined, 0 by default
C      scalehl(9)= user defined, 0 by default
C      scalehl(10)= user defined, 0 by default
C spinlh(3,..)       : spin information
C icolorflowlh(2,..) : (Anti-)Colour flow.
C
c -------------------------------------------------------------

      double precision eventweightmulti, scalemulti
      double precision alphaqedmulti, alphaqcdmulti
      integer          idrupmulti

      common/hepev5/eventweightmulti(NMXMLT),alphaqedmulti(NMXMLT),
     1              alphaqcdmulti(NMXMLT),scalemulti(10,NMXMLT),
     2              idrupmulti(NMXMLT)
      save /hepev5/

C idrupmulti(i)       : identity of the original interaction
C eventweightmulti(i) : event weight of the original interaction
C alphaqedmulti       : QED coupling alpha_em of the original interaction
C alphaqcdmulti       : QCD coupling alpha_s of the original interaction
C scalemulti(10,i)    : Scales of the original interaction
C
C -------------------------------------------------------------
