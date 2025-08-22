#  ****************************************************************************
#  * Author:
#  *   Grzegorz Jędrzejowski,
#  ****************************************************************************

import ROOT
import os

# --- Configuration ---
# Root file
root_file_path = "simu_2018_tracks.root"
# Set the output directory for plots
output_directory = "./plots/plotsTracks/"
# Define the sub-folders to process
folders_to_process_single_rp = ["RP 3", "RP 23", "RP 103", "RP 123"]
histograms_to_plot_single_rp = ["h2_y_vs_x"]#, "h_x", "h_y", "h_time"]

# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
    print(f"Created output directory: {output_directory}")

# --- Main Plotting Logic ---
def plot_histograms_from_folders(file_path, sub_dirs, hist_names, output_dir):
    """
    Accesses a ROOT file, reads histograms from specified sub-directories,
    and saves the plots to an output directory.
    """

    # Open the ROOT file
    root_file = ROOT.TFile.Open(file_path, "READ")
    if not root_file or root_file.IsZombie():
        print(f"Error: Could not open file {file_path}")
        return

    # Loop through each specified sub-folder
    for sub_dir_name in sub_dirs:
        print(f"Processing directory: {sub_dir_name}")

        # Get the sub-directory from the ROOT file
        # The TDirectoryFile is directly accessible by its name
        sub_dir = root_file.Get(sub_dir_name)
        if not sub_dir:
            print(f"Warning: Sub-directory '{sub_dir_name}' not found. Skipping.")
            continue

        # Loop through each histogram name
        for hist_name in hist_names:
            hist = sub_dir.Get(hist_name)

            if not hist:
                print(f"Warning: Histogram '{hist_name}' not found in '{sub_dir_name}'. Skipping.")
                continue

            hist.SetStats(0)
            hist.SetTitle(f"RP: {sub_dir_name} - {hist_name}")

            # Create a new canvas for each plot
            c = ROOT.TCanvas(f"c_{sub_dir_name.replace(' ', '_')}_{hist_name}", f"{sub_dir_name} - {hist_name}", 800, 600)
            c.SetLogz(1)

            # Draw the histogram based on its type (h2_ for 2D, others for 1D)
            if isinstance(hist, ROOT.TH2):
                hist.Draw("COLZ")
            elif isinstance(hist, ROOT.TH1):
                hist.Draw()
            else:
                print(f"Warning: Unknown histogram type for '{hist_name}' in '{sub_dir_name}'. Skipping.")
                del c
                continue

            # Update the canvas to apply all drawing changes
            c.Update()

            # Save the plot
            safe_file_name = f"{sub_dir_name}_{hist_name}".replace(" ", "_").replace("/", "_")
            output_path = os.path.join(output_dir, f"{safe_file_name}.png")
            c.SaveAs(output_path)
            print(f"Plot saved to {output_path}")

            # Clean up the canvas object
            del c

    # Close the ROOT file
    root_file.Close()
    print("All plots generated and file closed.")

# --- Run the plotter for singleRPPlots ---
plot_histograms_from_folders(
    file_path=root_file_path,
    sub_dirs=folders_to_process_single_rp,
    hist_names=histograms_to_plot_single_rp,
    output_dir=output_directory
)