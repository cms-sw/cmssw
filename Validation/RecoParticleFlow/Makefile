TMPDIR=test/tmp
RELVALCMD=${CMSSW_BASE}/src/Validation/RecoParticleFlow/test/run_relval.sh
DQM_MC=DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO
PHASE ?= phase1
PHASE_Label :=
ifeq ($(PHASE),phase2)
   PHASE_Label := _phase2
endif

conf:
	cd ${TMPDIR} && ${RELVALCMD} conf reco 0 ${PHASE}

dumpconf:
	cd ${TMPDIR}/conf && python3 -c 'import step3${PHASE_Label} as step3; print(step3.process.dumpPython())' > step3${PHASE_Label}_dump.py
	cp ${TMPDIR}/conf/step3${PHASE_Label}.py test/crab/
	cp ${TMPDIR}/conf/step3${PHASE_Label}_dump.py test/crab/

QCD: QCD_reco QCD_dqm

QCD_reco:
	cd ${TMPDIR} && ${RELVALCMD} QCD reco 0 ${PHASE}

#Need to expand the CMSSW python3 configuration
QCD_dumpconf:
	cd ${TMPDIR}/QCD && python3 -c 'import step3_RAW2DIGI_L1Reco_RECO_RECOSIM_PAT as step3; print(step3.process.dumpPython())' > step3_dump.py
	cp ${TMPDIR}/QCD/step3_dump.py crab/

QCDPU_reco:
	cd ${TMPDIR} && ${RELVALCMD} QCDPU reco 0 ${PHASE}

ZMMPU_reco:
	cd ${TMPDIR} && ${RELVALCMD} ZMMPU reco 0 ${PHASE}

ZEEPU_reco:
	cd ${TMPDIR} && ${RELVALCMD} ZEEPU reco 0 ${PHASE}

TenTauPU_reco:
	cd ${TMPDIR} && ${RELVALCMD} TenTauPU reco 0 ${PHASE}

NuGunPU_reco:
	cd ${TMPDIR} && ${RELVALCMD} NuGunPU reco 0 ${PHASE}

#dqm: QCD_dqm QCDPU_dqm ZMM_dqm MinBias_dqm SN_dqm
dqm: QCD_dqm QCDPU_dqm NuGunPU_dqm

QCD_dqm:
	rm -f ${TMPDIR}/QCD/DQM*.root
	cd ${TMPDIR} && ${RELVALCMD} QCD dqm 0 ${PHASE}

QCDPU_dqm:
	rm -f ${TMPDIR}/QCDPU/DQM*.root
	cd ${TMPDIR} && ${RELVALCMD} QCDPU dqm 0 ${PHASE}

ZMMPU_dqm:
	rm -f ${TMPDIR}/ZMMPU/DQM*.root
	cd ${TMPDIR} && ${RELVALCMD} ZMMPU dqm 0 ${PHASE}

ZEEPU_dqm:
	rm -f ${TMPDIR}/ZEEPU/DQM*.root
	cd ${TMPDIR} && ${RELVALCMD} ZEEPU dqm 0 ${PHASE}

TenTauPU_dqm:
	rm -f ${TMPDIR}/TenTauPU/DQM*.root
	cd ${TMPDIR} && ${RELVALCMD} TenTauPU dqm 0 ${PHASE}

NuGunPU_dqm:
	rm -f ${TMPDIR}/NuGunPU/DQM*.root
	cd ${TMPDIR} && ${RELVALCMD} NuGunPU dqm 0 ${PHASE}

.PHONY: plots # Enable re-running plots

plots:
	rm -Rf plots
	python3 test/compare.py \
	    --sample FlatQCD_noPU:${TMPDIR}/QCD/${DQM_MC}.root:${TMPDIR}/QCD/${DQM_MC}.root \
		--sample FlatQCD_PU25ns:${TMPDIR}/QCDPU/${DQM_MC}.root:${TMPDIR}/QCDPU/${DQM_MC}.root \
		--sample NuGunPU:${TMPDIR}/NuGunPU/${DQM_MC}.root:${TMPDIR}/NuGunPU/${DQM_MC}.root \
		--doResponsePlots --doOffsetPlots --doMETPlots --doPFCandPlots

QCD_plots:
	rm -Rf plots
	python3 test/compare.py \
		--sample FlatQCD_noPU:${TMPDIR}/QCD/${DQM_MC}.root:${TMPDIR}/QCD/${DQM_MC}.root \
		--doResponsePlots --doOffsetPlots --doMETPlots --doPFCandPlots

QCDPU_plots:
	rm -Rf plots
	python3 test/compare.py \
		--sample FlatQCD_PU25ns:${TMPDIR}/QCDPU/${DQM_MC}.root:${TMPDIR}/QCDPU/${DQM_MC}.root \
		--doResponsePlots --doOffsetPlots --doMETPlots --doPFCandPlots

NuGunPU_plots:
	rm -Rf plots
	python3 test/compare.py \
		--sample NuGunPU:${TMPDIR}/NuGunPU/${DQM_MC}.root:${TMPDIR}/NuGunPU/${DQM_MC}.root \
		--doResponsePlots --doOffsetPlots --doMETPlots --doPFCandPlots

plots_with_ref:
	rm -Rf plots
	python3 test/compare.py \
		--sample FlatQCD_noPU:${TMPDIR}/QCD/${DQM_MC}.root:${TMPDIR}/QCD_ref/${DQM_MC}.root \
		--sample FlatQCD_PU25ns:${TMPDIR}/QCDPU/${DQM_MC}.root:${TMPDIR}/QCDPU_ref/${DQM_MC}.root \
		--sample NuGunPU:${TMPDIR}/NuGunPU/${DQM_MC}.root:${TMPDIR}/NuGunPU_ref/${DQM_MC}.root \
		--doResponsePlots --doOffsetPlots --doMETPlots --doPFCandPlots
