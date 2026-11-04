"""LightNav-0 baseline glue: subset generation, result conversion, server probe, context window.

The model and its Habitat evaluation are the unmodified upstream clone under
``baselines/LightNav-0``; this package only prepares inputs, converts outputs, and
(``eval_ctx``) applies the paper's context window to the model's frame history from outside.
"""
