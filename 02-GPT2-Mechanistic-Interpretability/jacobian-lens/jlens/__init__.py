# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Jacobian lens: fit and apply the average input-output Jacobian as a readout
of decoder-transformer residuals."""

from jlens._logging import configure_logging
from jlens.fitting import fit, jacobian_for_prompt
from jlens.hf import HFLensModel, Layout, from_hf
from jlens.hooks import ActivationRecorder
from jlens.lens import JacobianLens
from jlens.protocol import LensModel

__all__ = [
    "ActivationRecorder",
    "HFLensModel",
    "JacobianLens",
    "Layout",
    "LensModel",
    "configure_logging",
    "fit",
    "from_hf",
    "jacobian_for_prompt",
]
# __all__ defines the public API this are the objects users are expected to import

# Core Algorithm
    #   JacobianLens
    #   fit
    #   jacobian_for_prompts
# This is the research core.


# Model Integration
    #    from_hf
    #    HFLensModel
    #    Layout
    #    LensModel
# This connects external transformer models to the lens.


# Activation Collection
    #    ActivationRecorder
# Responsible for recording internal activations.


# Infrastructure
    #   configure_logging
# Helpful for experiments, but not part of the algorithm itself.





"""           The Architectural Map.                 """ 

#                    User

#                      │
#                      ▼

#                 import jlens

#                      │
#                      ▼

#                  __init__.py
#                      │
#      ┌───────────────┼────────────────┐
#      │               │                │
#      ▼               ▼                ▼
#  fitting.py       hf.py           hooks.py
#      │               │                │
#      ▼               ▼                ▼
#  Jacobian       HF Wrapper     Activation Recorder
#      │
#      ▼
#  lens.py
#      │
#      ▼
#  JacobianLens   



"""              The Execution Pipeline (Our Current Hypothesis)             """
      
                #Based only on __init__.py, here's the execution flow we expect:
        

                            # Load Hugging Face Model
                            #           │
                            #           ▼
                            #       from_hf()
                            #           │
                            #           ▼
                            #    ActivationRecorder
                            #           │
                            #           ▼
                            #    Collect Residual Streams
                            #           │
                            #           ▼
                            #          fit()
                            #           │
                            #           ▼
                            #     JacobianLens Object
                            #           │
                            #           ▼
                            #  Apply Jacobian to New Activations
                            #           │
                            #           ▼
                            # Interpret the Representation