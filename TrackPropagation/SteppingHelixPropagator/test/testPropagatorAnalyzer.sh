#!/bin/bash

currTime=`date '+%Y.%m.%d-%H.%M.%S'`
outLockFile=${XXOUT_FNAME}.lock

echo ${PATH}

which cmsRun >& /dev/null
[ "$?" != "0" ] && echo "No cmsRun in path" && exit 1

if [ "${XXIN_FNAME}" == "" -o "${XXOUT_FNAME}" == "" ]; then
    echo "XXIN_FNAME or XXOUT_FNAME are not set"
    echo "Set both and try again"
    exit 1
fi

if [ ! -f "${XXIN_FNAME}" ]; then
    echo "No input file specified or file is nonexistent"
    echo "Check the file location and do export XXIN_NAME"
    exit 1
fi


if [ -f "${outLockFile}" ]; then
    echo "Output file is locked by another process"
    echo "Change the output file destination and restart"
    exit 1
fi

if [ -f "${XXOUT_FNAME}" ]; then
    if [ "$1" == "skip" ]; then
	echo "Output file exists. Skip this input"
	exit 1
    else
	oldOutFName=${XXOUT_FNAME}_oldAt-${currTime}
	echo "Output file exists and will be moved to" ${oldOutFName}
	mv ${XXOUT_FNAME} ${oldOutFName}
    fi
fi

touch ${outLockFile}

sedInFile=`echo ${XXIN_FNAME} | sed -e 's/\//\\\\\//g'`
sedOutFile=`echo ${XXOUT_FNAME} | sed -e 's/\//\\\\\//g'`

outLogFile=${XXOUT_FNAME}.log
outCfgFile=${XXOUT_FNAME}.cfg

cat testPropagatorAnalyzer.cfg |\
 sed -e "s/XXIN_FNAME/${sedInFile}/g;s/XXOUT_FNAME/${sedOutFile}/g;s/\\\\\//\//g"\
 > ${outCfgFile}

echo start cmsRun ${outCfgFile} at ${currTime} >&  ${outLogFile}
cmsRun ${outCfgFile} >&  ${outLogFile}

rm ${outLockFile}

exit $?

