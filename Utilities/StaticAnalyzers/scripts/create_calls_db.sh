#!/usr/bin/env bash
export LC_ALL=C

eval `scram runtime -sh`

cd ${LOCALRT}/tmp

if [ ! -f ./function-calls-db.txt ]
        then 
	echo "run ${CMSSW_BASE}/src/Utilities/StaticAnalyzers/scripts/run_class_dumper.sh first"
	exit 1
fi

if [ ! -f ./function-statics-db.txt ]
        then
	echo "run ${CMSSW_BASE}/src/Utilities/StaticAnalyzers/scripts/run_class_checker.sh first"
	exit 2
fi
${CMSSW_BASE}/src/Utilities/StaticAnalyzers/scripts/create_statics_esd_reports.sh
