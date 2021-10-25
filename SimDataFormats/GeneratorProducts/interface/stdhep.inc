C -------------------------------------------------------------
C
      integer NMXHEP
      parameter (NMXHEP=4000)
      common/HEPEVT/NEVHEP,NHEP,ISTHEP(NMXHEP),IDHEP(NMXHEP),
     &JMOHEP(2,NMXHEP),JDAHEP(2,NMXHEP),PHEP(5,NMXHEP),VHEP(4,NMXHEP)
      integer NEVHEP,NHEP,ISTHEP,IDHEP,JMOHEP,JDAHEP
      double precision PHEP,VHEP
      save /HEPEVT/
C...      NEVHEP       - event number
C...      NHEP         - number of entries in this event
C...      ISTHEP(..)   - status code
C...      IDHEP(..)    - particle ID, P.D.G. standard
C...      JMOHEP(1,..) - position of mother particle in list
C...      JMOHEP(2,..) - position of second mother particle in list
C...      JDAHEP(1,..) - position of first daughter in list
C...      JDAHEP(2,..) - position of last daughter in list
C...      PHEP(1,..)   - x momentum in GeV/c
C...      PHEP(2,..)   - y momentum in GeV/c
C...      PHEP(3,..)   - z momentum in GeV/c
C...      PHEP(4,..)   - energy in GeV
C...      PHEP(5,..)   - mass in GeV/c**2
C...      VHEP(1,..)   - x vertex position in mm
C...      VHEP(2,..)   - y vertex position in mm
C...      VHEP(3,..)   - z vertex position in mm
C...      VHEP(4,..)   - production time in mm/c
C
C -------------------------------------------------------------
C
      common/hepev2/nmulti,jmulti(NMXHEP)
      integer nmulti,jmulti
      save /hepev2/
C...      nmulti       - number of interactions in the list
C...      jmulti(..)   - multiple interaction number
C
C -------------------------------------------------------------
      integer NMXMLT
      parameter (NMXMLT=16)
      common/hepev3/nevmulti(NMXMLT),itrkmulti(NMXMLT),mltstr(NMXMLT)
      integer nevmulti,itrkmulti,mltstr
      save /hepev3/
C...      nevmulti(i)  - event number of original interaction
C...      itrkmulti(i) - first particle in the original interaction
C...      mltstr(i)    - stream this event is from
C
C -------------------------------------------------------------
