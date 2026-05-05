"""Pydantic schemas for plotting configuration and input data."""

from __future__ import annotations

from pydantic import BaseModel, model_validator


class ModelConfig(BaseModel):
    id: str
    display_name: str


class DomainConfig(BaseModel):
    id: str
    display_name: str
    metric: str | None = None


class MethodConfig(BaseModel):
    id: str
    display_name: str
    color: str | None = None
    requires_scope_domain: bool = True


class DomainsConfig(BaseModel):
    default_metric: str
    entries: list[DomainConfig]

    def get_metric(self, domain_id: str) -> str:
        for d in self.entries:
            if d.id == domain_id and d.metric is not None:
                return d.metric
        return self.default_metric

    def get_display_name(self, domain_id: str) -> str:
        for d in self.entries:
            if d.id == domain_id:
                return d.display_name
        return domain_id


class InDomainBarConfig(BaseModel):
    """Config for Figure 1: in-domain bar plot."""
    methods: list[str]
    baseline_method: str = "vanilla"
    # TODO(adriano): review — do we want a title template per model, or a single title?
    title_template: str = "In-Domain Performance ({model})"


class OODTableConfig(BaseModel):
    """Config for Figure 2: OOD text table."""
    our_method: str
    comparison_methods: list[str]
    # TODO(adriano): review — how to define "good" vs "bad" for coloring?
    # Current approach: green if our_method <= best_comparison, yellow otherwise.
    # Should this be configurable, e.g. a threshold or a direction flag?
    title_template: str = "OOD Elicitation ({model})"


class FiguresConfig(BaseModel):
    in_domain_bar: InDomainBarConfig | None = None
    ood_table: OODTableConfig | None = None


class OutputConfig(BaseModel):
    dpi: int = 300
    latex: bool = True
    # TODO(adriano): review — any other output formats needed? SVG?


class PlotConfig(BaseModel):
    models: list[ModelConfig]
    domains: DomainsConfig
    methods: list[MethodConfig]
    figures: FiguresConfig
    output: OutputConfig = OutputConfig()

    def get_method(self, method_id: str) -> MethodConfig:
        for m in self.methods:
            if m.id == method_id:
                return m
        raise ValueError(f"Method '{method_id}' not found in config. Available: {[m.id for m in self.methods]}")

    @model_validator(mode="after")
    def validate_figure_references(self) -> PlotConfig:
        method_ids = {m.id for m in self.methods}
        if self.figures.in_domain_bar:
            fig = self.figures.in_domain_bar
            for mid in fig.methods:
                if mid not in method_ids:
                    raise ValueError(f"in_domain_bar references unknown method '{mid}'")
            if fig.baseline_method not in method_ids:
                raise ValueError(f"in_domain_bar baseline_method '{fig.baseline_method}' not in methods")
        if self.figures.ood_table:
            fig = self.figures.ood_table
            if fig.our_method not in method_ids:
                raise ValueError(f"ood_table our_method '{fig.our_method}' not in methods")
            for mid in fig.comparison_methods:
                if mid not in method_ids:
                    raise ValueError(f"ood_table references unknown comparison method '{mid}'")
        return self


class ScoreRow(BaseModel):
    """One row of the input CSV. Each row is one (model, method, scope, elicitation, metric) -> score."""
    model: str
    method: str
    scope_domain: str | None = None
    elicitation_domain: str
    metric: str
    score: float
