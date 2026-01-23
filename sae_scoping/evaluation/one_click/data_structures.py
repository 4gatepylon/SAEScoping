from __future__ import annotations

import pandera.pandas as pa
from pandera.dtypes import Float, String
from pandera.typing import Series

"""This module is meant to expose data structures and schemas that (most of the time) are used privately."""


class LabeledScoreDf(pa.DataFrameModel):
    # https://beartype.readthedocs.io/en/latest/faq/#pandas-data-frames
    label: Series[String]
    score: Series[Float]


class JudgementsDf(pa.DataFrameModel):
    # TODO(Adriano) this seems to not be typechecking? Wtf?
    prompt: Series[String]  # user_request
    response: Series[String]
    seed: Series[String]  # user_request_clean in past versions
    judge_name: Series[String]  # (prompt, judge_name) is what is unique here
    # HYDRATED-template (judge name is 1:1 with non-hydrated template)
    judge_template: Series[String]
    # number in [0, 1] the LLM judge said/gave us; NOTE on error default to 0.0
    judgement_score: Series[Float]
    # explanation of the judgement by LLM judge (or informative text if error)
    judgement_explanation: Series[String]
