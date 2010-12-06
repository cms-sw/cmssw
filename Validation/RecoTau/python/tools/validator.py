
'''

Modify a process to produce Validation plots for reco::Taus


'''

import FWCore.ParameterSet.Config as cms
import Validation.RecoTau.tools.prototypes as proto


def validate(process, algorithm, algorithm_info, sequence,
             discriminators=None, suffix=""):
    '''
    Given a producer and a list of discriminators,
    construct a sequence that selects taus passing each discriminator
    in turn and then plots quantities of the filtered taus.

    These new modules will be added to the [sequence].
    '''

    # This returns a function that registers a module in the process and adds
    # it to the given sequence.
    builder = proto.make_process_adder(process, sequence)

    # Make histograms of our RAW taus
    raw_plots = proto.tau_plotter.clone(
        src = cms.InputTag(algorithm_info['producer'])
    )
    # FIXME is it possible this will double count?
    # Add to process
    builder(algorithm + suffix + "Select" + "noSelection" +  "Plots", raw_plots)
    # Select and plot those passing each of the discriminators
    current_tau_src = algorithm_info['producer']

    for discriminator in discriminators:
        selector = proto.tau_disc_selector.clone(
            src = cms.InputTag(current_tau_src),
            discriminator = cms.InputTag(discriminator)
        )
        #current_tau_src = algorithm + "Select" + discriminator
        current_tau_src = algorithm + suffix + "Select" + discriminator
        builder(current_tau_src, selector)
        # Now make plots of those taus passing this selector
        plotter = proto.tau_plotter.clone(
            src = cms.InputTag(current_tau_src)
        )
        builder(current_tau_src + "Plots", plotter)
