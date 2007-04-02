#!/bin/bash 

RETVAL=0

if [ "_"$DORECO == "_yes" ]; then
	cmsRun reco_$1.cfg 2>&1 | gzip > $OUTDIR/reco_$1.log.gz
	RETVAL=$?
fi

if [ $RETVAL == 0 ]; then
	cmsRun $1.cfg 2>&1 | gzip > $OUTDIR/$1.log.gz
	RETVAL=$?
fi

rm -f reco.root
\mv -f *.root $OUTDIR/

exit $RETVAL
