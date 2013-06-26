#!/bin/bash

USAGE="Usage: `basename $0` NrOfEvents MinimumEnergy MaximumEnergy Tag"

case $# in
4)
        MAXEVENTS=$1
        MINIMUM_ENERGY=$2
        MAXIMUM_ENERGY=$3
        TAG=$4
        ;;
*)
        echo $USAGE; exit 1;
        ;;
esac

# directory where the job is run or submitted
if [ "${LS_SUBCWD+set}" = set ]; then
   LK_WKDIR="${LS_SUBCWD}" # directory where you submit in case of bsub
   WWDIR="${WORKDIR}"
else
   LK_WKDIR=`pwd`          # directory where you run locally otherwise
   WWDIR=`pwd`
fi

#
TEMPLATE_CFG="${LK_WKDIR}/template_pion_cfg.py"
OUTPUT="output_${MAXEVENTS}_${MINIMUM_ENERGY}_${MAXIMUM_ENERGY}_${TAG}"
CFGFILE=${WWDIR}/${OUTPUT}_cfg.py
NTUPLEFILE=${WWDIR}/${OUTPUT}.eventNtuple.root
POOLOUTPUTFILE=${WWDIR}/${OUTPUT}.pool.root
LOGFILE=${WWDIR}/${OUTPUT}.log
#
CASTOR_DIR="/castor/cern.ch/user/h/hvanhaev/CMSSW_3_3_0_pre2/pions"
rfdir $CASTOR_DIR/${OUTPUT}.eventNtuple.root 2>/dev/null
return_val=$?
if [ $return_val -eq 0 ]; then
  echo "castor file $CASTOR_DIR/${OUTPUT}.eventNtuple.root exists, delete it first. Stopping here!"
  exit 1;
fi
#
if [ -e $CFGFILE ]; then
   echo "The cfg file $CFGFILE already exists. Stopping here!"
   exit 1;
fi

#
#
DIR_WHERE_TO_EVAL="/afs/cern.ch/user/h/hvanhaev/scratch0/CMSSW_3_3_0_pre2/src"

# random numbers
RAND_sourceSeed=`head -c4 /dev/urandom | od -N3 -tu4 | sed -ne '1s/.* //p'`
RAND_VtxSmeared=`head -c4 /dev/urandom | od -N3 -tu4 | sed -ne '1s/.* //p'`
RAND_g4SimHits=`head -c3 /dev/urandom  | od -N1 -tu4 | sed -ne '1s/.* //p'`
RAND_mix=`head -c4 /dev/urandom        | od -N2 -tu4 | sed -ne '1s/.* //p'`
RAND_generator=`head -c4 /dev/urandom  | od -N3 -tu4 | sed -ne '1s/.* //p'`

cat $TEMPLATE_CFG | perl -p -e "s@MAXEVENTS@$MAXEVENTS@" | perl -p -e "s@POOLOUTPUTFILE@$POOLOUTPUTFILE@"  | perl -p -e "s@RAND_VtxSmeared@$RAND_VtxSmeared@" | perl -p -e "s@RAND_generator@$RAND_generator@"| perl -p -e "s@RAND_g4SimHits@$RAND_g4SimHits@" | perl -p -e "s@RAND_mix@$RAND_mix@" | perl -p -e "s@RAND_sourceSeed@$RAND_sourceSeed@" | perl -p -e "s@NTUPLEFILE@$NTUPLEFILE@" | perl -p -e "s@MAXIMUM_ENERGY@$MAXIMUM_ENERGY@" | perl -p -e "s@MINIMUM_ENERGY@$MINIMUM_ENERGY@"   > $CFGFILE

cd $DIR_WHERE_TO_EVAL; eval `scramv1 runtime -sh`; cd -;
cmsRun $CFGFILE > $LOGFILE 2>&1
return_val=$?
if [ $return_val -eq 0 ]; then
  rfcp $CFGFILE $CASTOR_DIR/
  rfcp $POOLOUTPUTFILE $CASTOR_DIR/
  rfcp $NTUPLEFILE $CASTOR_DIR/
fi

