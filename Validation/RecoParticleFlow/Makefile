TMPDIR=tmp
RELVALCMD=${CMSSW_BASE}/src/Validation/RecoParticleFlow/test/run_relval.sh

QCD: QCD_reco QCD_dqm

QCD_reco:
	cd ${TMPDIR} && ${RELVALCMD} QCD reco 0

#Need to expand the CMSSW python configuration
QCD_dumpconf:
	cd ${TMPDIR}/QCD && python -c 'import step3_RAW2DIGI_L1Reco_RECO_RECOSIM_EI_PAT as step3; print step3.process.dumpPython()' > step3_dump.py
	cp ${TMPDIR}/QCD/step3_dump.py crab/

QCDPU_reco:
	cd ${TMPDIR} && ${RELVALCMD} QCDPU reco 0

ZMM_reco:
	cd ${TMPDIR} && ${RELVALCMD} ZMM reco 0

MinBias_reco:
	cd ${TMPDIR} && ${RELVALCMD} MinBias reco 0

dqm: QCD_dqm QCDPU_dqm ZMM_dqm MinBias_dqm

QCD_dqm:
	rm -f ${TMPDIR}/QCD/DQM*.root
	cd ${TMPDIR} && ${RELVALCMD} QCD dqm 0

QCDPU_dqm:
	rm -f ${TMPDIR}/QCDPU/DQM*.root
	cd ${TMPDIR} && ${RELVALCMD} QCDPU dqm 0

ZMM_dqm:
	rm -f ${TMPDIR}/ZMM/DQM*.root
	cd ${TMPDIR} && ${RELVALCMD} ZMM dqm 0

MinBias_dqm:
	rm -f ${TMPDIR}/MinBias/DQM*.root
	cd ${TMPDIR} && ${RELVALCMD} MinBias dqm 0

post: QCD_post QCDPU_post ZMM_post MinBias_post

QCD_post:
	rm -f ${TMPDIR}/QCD/*withResponse.root ${TMPDIR}/QCD/DQMTotal.root
	cd ${TMPDIR}/QCD && python ${CMSSW_BASE}/src/Validation/RecoParticleFlow/test/addResponse.py DQM*.root
	hadd -f ${TMPDIR}/QCD/DQMTotal.root ${TMPDIR}/QCD/DQM_*.root

QCDPU_post:
	rm -f ${TMPDIR}/QCDPU/*withResponse.root ${TMPDIR}/QCDPU/DQMTotal.root
	cd ${TMPDIR}/QCDPU && python ${CMSSW_BASE}/src/Validation/RecoParticleFlow/test/addResponse.py DQM*.root
	hadd -f ${TMPDIR}/QCDPU/DQMTotal.root ${TMPDIR}/QCDPU/DQM_*.root

ZMM_post:
	rm -f ${TMPDIR}/ZMM/*withResponse.root ${TMPDIR}/ZMM/DQMTotal.root
	cd ${TMPDIR}/ZMM && python ${CMSSW_BASE}/src/Validation/RecoParticleFlow/test/addResponse.py DQM*.root
	hadd -f ${TMPDIR}/ZMM/DQMTotal.root ${TMPDIR}/ZMM/DQM_*.root

MinBias_post:
	rm -f ${TMPDIR}/MinBias/*withResponse.root ${TMPDIR}/MinBias/DQMTotal.root
	cd ${TMPDIR}/MinBias && python ${CMSSW_BASE}/src/Validation/RecoParticleFlow/test/addResponse.py DQM*.root
	hadd -f ${TMPDIR}/MinBias/DQMTotal.root ${TMPDIR}/MinBias/DQM_*.root

.PHONY: plots # Enable re-running plots
plots:
	rm -Rf plots	
	python test/compare.py \
		--sample FlatQCD_noPU:${TMPDIR}/QCD/DQMTotal.root:${TMPDIR}/QCD/DQMTotal.root \
		--sample FlatQCD_PU25ns:${TMPDIR}/QCDPU/DQMTotal.root:${TMPDIR}/QCDPU/DQMTotal.root \
		--sample MinBias:${TMPDIR}/MinBias/DQMTotal.root:${TMPDIR}/MinBias/DQMTotal.root \
		--sample ZMM:${TMPDIR}/ZMM/DQMTotal.root:${TMPDIR}/ZMM/DQMTotal.root \
		--doResponsePlots --doOffsetPlots

