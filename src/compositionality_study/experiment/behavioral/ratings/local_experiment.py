"""Local implementation of the behavioral experiment using Flask."""

import os
from typing import Optional

import pandas as pd
from flask import Flask, redirect, render_template, request, url_for

from compositionality_study.constants import BEHAV_OUTPUT_DIR, DATA_DIR

app = Flask(__name__)

# Global variables to track experiment state
current_subject: Optional[int] = None
current_trial: int = 0
current_modality: Optional[str] = None
responses: list = []


@app.route("/")
def index():
    """Landing page to start experiment and select subject number."""
    return render_template("index.html")


@app.route("/start", methods=["POST"])
def start_experiment():
    """Initialize experiment with subject number and modality."""
    global current_subject, current_modality, current_trial, responses

    current_subject = int(request.form["subject_number"])
    current_modality = request.form["modality"]
    current_trial = 0
    responses = []

    return redirect(url_for("trial"))


@app.route("/trial", methods=["GET", "POST"])
def trial():
    """Handle individual trials of the experiment."""
    global current_trial, responses

    # Load the appropriate stimulus file
    # TODO remove amt later
    stim_file = f"amt_{current_modality}_pairings_{current_subject}.csv"
    stimuli = pd.read_csv(os.path.join(DATA_DIR, "amt", "input", stim_file))

    if request.method == "POST":
        # Record response
        responses.append(
            {
                "Input.high_complex_on": stimuli.iloc[current_trial]["high_complex_on"],
                "Answer.more-complex.label": request.form["response"],
                "trial": current_trial,
            }
        )

        current_trial += 1

        # Check if experiment is complete
        if current_trial >= len(stimuli):
            return redirect(url_for("complete"))

    # Get current trial stimuli
    stim_a = stimuli.iloc[current_trial]["A"]
    stim_b = stimuli.iloc[current_trial]["B"]

    return render_template(
        "trial.html",
        modality=current_modality,
        stim_A=stim_a,
        stim_B=stim_b,
        trial_num=current_trial + 1,
        total_trials=len(stimuli),
    )


@app.route("/complete")
def complete():
    """Save results and complete the experiment."""
    # Save responses
    pd.DataFrame(responses).to_csv(
        os.path.join(BEHAV_OUTPUT_DIR, f"subject_{current_subject}_results.csv"), index=False
    )

    return render_template("complete.html")


if __name__ == "__main__":
    app.run(port=8000)
