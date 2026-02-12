"""Generate high-quality surface plots for group-level brain maps.

This script searches for group-level NIfTI files (univariate and MVPA) in specified directories,
projects them onto the fsaverage surface, and saves both static (PNG) and interactive (HTML)
surface plots.
"""

import click
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nilearn import datasets, plotting, surface
from pathlib import Path
from tqdm import tqdm
from typing import List

from compositionality_study.constants import VISUALIZATIONS_DIR

# Visualize everything
PATTERNS = [
    "*.nii.gz",
]

# Patterns to exclude (e.g. intermediate files)
EXCLUDE_PATTERNS = [
    "*.csv",
    "*.png",
]

def find_files(input_dirs: List[Path]) -> List[Path]:
    """Find NIfTI files matching patterns in the input directories."""
    files = []
    for d in input_dirs:
        if not d.exists():
            print(f"Warning: Directory {d} does not exist.")
            continue
            
        for pattern in PATTERNS:
            found = list(d.rglob(pattern))
            for f in found:
                # Basic exclusion
                if any(f.match(ex) for ex in EXCLUDE_PATTERNS):
                    continue
                files.append(f)
    
    # Remove duplicates and sort
    return sorted(list(set(files)))


def format_title(stem: str) -> str:
    """Format the filename stem into a readable title."""
    s = stem
    # Remove common suffixes
    for suffix in [".nii_thr", "_zmap", "_clusterFWE", "_neglogp", "group_"]:
        s = s.replace(suffix, "")
    
    # Replace separators
    s = s.replace("_", " ")
    
    # Start with replacements for known acronyms/abbr
    words = s.split()
    new_words = []
    
    # Mapping for specific terms
    replacements = {
        "img": "Image",
        "txt": "Text",
        "vs": "vs.",
        "loc": "Localizer",
        "comp": "Compositionality",
        "prod": "Production",
        "main": "Main Effect:",
        "interaction": "Interaction",
        "modality": "Modality",
        "high": "High",
        "low": "Low",
        "cross": "Cross",
        "within": "Within",
        "func": "Functional",
        "text2image": "Text -> Image",
        "image2text": "Image -> Text",
        "conjunction": "Conjunction",
        "txt_high_vs_low": "Text High vs. Low", # catch full phrase if needed
    }
    
    # Special handling for "text2image" or similar if they are single tokens
    for w in words:
        # Check if we have a direct replacement
        if w in replacements:
            new_words.append(replacements[w])
        # Check if it looks like "text2image"
        elif "2" in w and not w.isdigit(): 
            # split by 2
            parts = w.split("2")
            if len(parts) == 2:
                p1 = replacements.get(parts[0], parts[0].capitalize())
                p2 = replacements.get(parts[1], parts[1].capitalize())
                new_words.append(f"{p1} -> {p2}")
            else:
                new_words.append(w.capitalize())
        else:
            new_words.append(w.capitalize())

    return " ".join(new_words)


def generate_surface_plots(
    nii_file: Path, 
    output_dir: Path, 
    threshold: float = 1e-6
):
    """Generate surface plots for a single NIfTI file."""
    # Import for colorbar construction
    from matplotlib import cm, colors
    
    # Set nice font
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Calibri', 'Arial', 'DejaVu Sans', 'sans-serif']
    
    # Load fsaverage surface
    fsaverage = datasets.fetch_surf_fsaverage()
    
    # Define stem for filenames
    stem = nii_file.name.replace(".nii.gz", "").replace(".nii", "")
    formatted_title = format_title(stem)
    
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing {nii_file.name} -> {formatted_title}")

    img = nib.load(nii_file)  # type: ignore

    # Check if the image is empty (all zeros)
    data = img.get_fdata()  # type: ignore
    if np.allclose(data, 0):
        print(f"  Skipping {nii_file.name} (empty map)")
        return

    # Calculate global max for consistent scaling across subplots
    vmax = np.nanmax(np.abs(data))

    # Setup the combined figure: 2 rows (Lat/Med), 2 cols (Left/Right)
    # Figure size is large to accommodate high res
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), subplot_kw={'projection': '3d'})
    fig.patch.set_facecolor('white')
    
    # Reduce spacing between subplots to cluster them together
    plt.subplots_adjust(wspace=-0.1, hspace=-0.2, top=0.92, bottom=0.05, left=0.05, right=0.90)
    
    # Define layout mapping: axes[row, col]
    # Row 0: Lateral (Left, Right)
    # Row 1: Medial (Left, Right)
    layout = {
        'left': {'lateral': axes[0, 0], 'medial': axes[1, 0]},
        'right': {'lateral': axes[0, 1], 'medial': axes[1, 1]}
    }

    # Common plotting parameters
    cmap = 'autumn'

    # Project and plot for each hemisphere
    modes = ['left', 'right']
    
    for hemi in modes:
        mesh = fsaverage[f"pial_{hemi}"]
        # Load the surface texture
        texture = surface.vol_to_surf(img, mesh, interpolation='linear', radius=3.0)
        
        # Plot Lateral and Medial for this hemisphere
        for view in ['lateral', 'medial']:
            ax = layout[hemi][view]
            
            plotting.plot_surf_stat_map(
                fsaverage[f"infl_{hemi}"],
                texture,
                hemi=hemi,
                view=view,
                bg_map=fsaverage[f"sulc_{hemi}"],
                title="", # Clear title from nilearn
                threshold=threshold,
                vmax=vmax,
                axes=ax,
                cmap=cmap,
                colorbar=False, # Disable individual colorbars
                bg_on_data=False, # Opaque clusters for clarity
            )
            # Add custom title at the bottom instead of top
            ax.set_title("")

    # Add single shared colorbar using a dedicated axis
    if 'interaction' in stem or 'vs' in stem:
        # Symmetric norm for divergent maps
        norm = colors.Normalize(vmin=-vmax, vmax=vmax)
    else:
        # Positive norm for others
        norm = colors.Normalize(vmin=threshold, vmax=vmax)
        
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    # Position: Left, Bottom, Width, Height
    cbar_ax = fig.add_axes([0.92, 0.25, 0.02, 0.5])  # type: ignore
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=22)

    # Set overall title and save combined figure
    plt.suptitle(formatted_title, fontsize=36, y=0.95, weight='bold') # Added bold
    outfile = output_dir / f"{stem}_surface_combined.png"
    plt.savefig(outfile, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)

@click.command()
@click.option(
    "--input-dirs", 
    "-i", 
    multiple=True, 
    required=True,
    help="Input directories containing NIfTI files (e.g., group maps)."
)
@click.option(
    "--output-dir", 
    "-o", 
    type=click.Path(path_type=Path), 
    default=Path(VISUALIZATIONS_DIR) / "surface_plots",
    help="Directory where plots will be saved."
)
@click.option(
    "--threshold", 
    "-t", 
    type=float, 
    default=1e-6,
    help="Threshold for identifying significant regions in visualization."
)
def main(input_dirs: tuple, output_dir: Path, threshold: float):
    """Generate surface visualizations for brain maps found in input directories."""
    
    input_paths = [Path(d) for d in input_dirs]
    files = find_files(input_paths)
    
    if not files:
        print("No matching NIfTI files found.")
        return

    print(f"Found {len(files)} brain maps to process.")
    output_dir.mkdir(parents=True, exist_ok=True)

    for f in tqdm(files, desc="Generating plots"):
        try:
            generate_surface_plots(f, output_dir, threshold=threshold)
        except Exception as e:
            print(f"Failed to process {f}: {e}")

if __name__ == "__main__":
    main()
