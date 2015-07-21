#!/bin/bash

set -o nounset

# This is the list of input files containing the GEN-SIM events to be digitized:
TheInputFile="'/store/mc/TTI2023Upg14/SingleMuPlusFlatPt0p2To150/GEN-SIM/DES23_62_V1-v1/00000/8A16A067-13D2-E311-94A6-0025905A612A.root','/store/mc/TTI2023Upg14/SingleMuPlusFlatPt0p2To150/GEN-SIM/DES23_62_V1-v1/00000/B8366020-F2D1-E311-98AE-0025905A6070.root'"
TYPE=SingleMuPlus
EOSDIR=Muons

PILEUP=140

# Example, with  NumberofPUFiles=2 :
# Each job is making use of 2 PU files. From the number of MinBias events
# that we have on these 2 files (nPUevents), we deduce the number of events that
# should be processed in order to use each PU event only once (mevents=nPUevents/PILEUP).
#

NTOT=1000      # number events to be digitized
NJOBS=1000	# max. number of jobs


NumberofPUFiles=2	# means that each job will use 2 PU files

let "i=0"

let "kk=1"

pufiles=""
let "nPUevents=0"
let "nevents=0"    # parameter of skipEvents

for thefileLine in `cat TheMinBiasFiles.txt`

do

    if [ $nevents -gt $NTOT ]
    then
	break
    fi

    commastring=","
    indexcomma=`expr index $thefileLine $commastring`
    thefile=`expr substr $thefileLine 1 $indexcomma`
    nbevts=${thefileLine:$indexcomma}

    if [ $kk -lt $NumberofPUFiles ]
    then
        # in case NumberofPUFiles > 1 : one first determines the
        # list of PU files to be used for the next job
       let "kk=kk+1"
        echo " ... add PU file"
        pufiles=$pufiles$thefile
	let "nPUevents=nPUevents+nbevts"
       
    else
	# now the list of PU files to be used is complete
	# (this could be one single PU file, if NumberofPUFiles = 1
 	# or if it is < 1 )
       let "kk=1"
       pufiles=$pufiles$thefile
       let "nPUevents=nPUevents+nbevts"

    let "i=i+1"
        echo "ready for job "$i
	#echo "number of PU events available = " $nPUevents
        let "mevents=nPUevents/PILEUP"
	echo "number of events to process in this job =" $mevents
    if [ $i -le $NJOBS ]
    then
    let "j=i+1"
    let "evtstoskip=nevents"
    echo $pufiles

    cname="Digi_""$TYPE""_""$i""_cfg.py"
    echo $cname
    sed -e "s@JOB_ID@$i@g" \
        -e "s@NUM_EVENTS@$mevents@" \
        -e "s@FILES_FOR_PILEUP@$pufiles@" \
        -e "s@EVTSTOSKIP@$evtstoskip@g" \
        -e "s@TheInputFile@$TheInputFile@g" \
        -e "s@PILEUP@$PILEUP@g" \
        -e "s@TYPE@$TYPE@g" \
            DIGI.tmpl  > $cname

    jname='sub_'$PILEUP'_'$TYPE'_'$i
    jtemp='submit_DIGI.tmpl'
    sed -e "s@INDX@$i@g" \
        -e "s@TYPE@$TYPE@g" \
        -e "s@PILEUP@$PILEUP@g" \
        -e "s@EOSDIR@$EOSDIR@g" \
        $jtemp > $jname
    chmod 755 $jname

	# First run the script with the bsub line commented/
	# once everything is fine, rerun the script with
	# the bsub commands :

    	#echo "submitting job : "$jname
    	#bsub -q1nd $jname

    fi

    pufiles=""
    let "nPUevents=0"
    let "nevents=nevents+mevents"

    fi
done

