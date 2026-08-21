from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from .physical_properties import (
    DEFAULT_PHYSICAL_PROPERTIES,
    DEFAULT_PHYSICAL_PROPERTY_RANGES,
    PhysicalProperties,
    PhysicalPropertyRanges,
)


@dataclass(frozen=True)
class _SliderSpec:
    field_name: str
    label: str
    display_factor: float
    format_spec: str

    def format(self, value: float) -> str:
        return format(value * self.display_factor, self.format_spec)


_SLIDER_SPECS = (
    _SliderSpec("mass_kg", "Mass (kg)", 1.0, ".3f"),
    _SliderSpec("sliding_friction", "Sliding friction", 1.0, ".4f"),
    _SliderSpec("torsional_friction", "Torsional friction", 1.0, ".5f"),
    _SliderSpec("rolling_friction", "Rolling friction", 1.0, ".6f"),
    _SliderSpec("length_m", "Length (cm)", 100.0, ".2f"),
    _SliderSpec("width_m", "Width (cm)", 100.0, ".2f"),
    _SliderSpec("thickness_m", "Thickness (cm)", 100.0, ".2f"),
)


def show_physics_controls(
    initial: PhysicalProperties = DEFAULT_PHYSICAL_PROPERTIES,
    ranges: PhysicalPropertyRanges = DEFAULT_PHYSICAL_PROPERTY_RANGES,
    *,
    rng: random.Random | None = None,
) -> PhysicalProperties | None:
    """Show a blocking slider dialog.

    Returns the selected properties, or ``None`` when the user cancels.
    Tk is imported lazily so normal and headless runs do not require a display.
    """
    try:
        import tkinter as tk
        from tkinter import ttk
    except ImportError as exc:
        raise RuntimeError(
            "The physical-property controls require Python's tkinter package"
        ) from exc

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        raise RuntimeError(
            "Could not open the physical-property controls. Check that a desktop "
            "display is available, or run without --physics-controls."
        ) from exc

    root.title("FlipUp physical properties")
    root.resizable(False, False)
    root.columnconfigure(0, weight=1)

    selected: PhysicalProperties | None = None
    generator = rng if rng is not None else random.Random()
    variables: dict[str, Any] = {}
    value_labels: dict[str, Any] = {}

    outer = ttk.Frame(root, padding=16)
    outer.grid(row=0, column=0, sticky="nsew")
    outer.columnconfigure(1, weight=1)

    title = ttk.Label(
        outer,
        text="Book physical properties",
        font=("TkDefaultFont", 13, "bold"),
    )
    title.grid(row=0, column=0, columnspan=4, sticky="w")
    instructions = ttk.Label(
        outer,
        text=(
            "Drag the bars to set exact values. Randomize samples every value "
            "uniformly from the range shown at the right."
        ),
        wraplength=590,
        justify="left",
    )
    instructions.grid(
        row=1,
        column=0,
        columnspan=4,
        sticky="w",
        pady=(4, 14),
    )

    def update_value_label(field_name: str) -> None:
        spec = next(item for item in _SLIDER_SPECS if item.field_name == field_name)
        value_labels[field_name].configure(
            text=spec.format(float(variables[field_name].get()))
        )

    for row_offset, spec in enumerate(_SLIDER_SPECS, start=2):
        property_range = getattr(ranges, spec.field_name)
        initial_value = float(getattr(initial, spec.field_name))
        default_value = float(getattr(DEFAULT_PHYSICAL_PROPERTIES, spec.field_name))
        slider_minimum = min(
            property_range.minimum,
            initial_value,
            default_value,
        )
        slider_maximum = max(
            property_range.maximum,
            initial_value,
            default_value,
        )

        ttk.Label(outer, text=spec.label).grid(
            row=row_offset,
            column=0,
            sticky="w",
            padx=(0, 10),
            pady=5,
        )
        variable = tk.DoubleVar(value=initial_value)
        variables[spec.field_name] = variable
        slider = ttk.Scale(
            outer,
            from_=slider_minimum,
            to=slider_maximum,
            variable=variable,
            length=280,
            command=lambda _value, name=spec.field_name: update_value_label(name),
        )
        slider.grid(row=row_offset, column=1, sticky="ew", pady=5)

        value_label = ttk.Label(outer, width=10, anchor="e")
        value_label.grid(
            row=row_offset,
            column=2,
            sticky="e",
            padx=(10, 8),
            pady=5,
        )
        value_labels[spec.field_name] = value_label
        update_value_label(spec.field_name)

        range_text = (
            f"[{spec.format(property_range.minimum)} – "
            f"{spec.format(property_range.maximum)}]"
        )
        ttk.Label(outer, text=range_text, foreground="#666666").grid(
            row=row_offset,
            column=3,
            sticky="e",
            pady=5,
        )

    error_label = ttk.Label(outer, text="", foreground="#b00020", wraplength=590)
    error_label.grid(
        row=2 + len(_SLIDER_SPECS),
        column=0,
        columnspan=4,
        sticky="w",
        pady=(8, 0),
    )

    def set_values(properties: PhysicalProperties) -> None:
        for spec in _SLIDER_SPECS:
            variables[spec.field_name].set(getattr(properties, spec.field_name))
            update_value_label(spec.field_name)
        error_label.configure(text="")

    def randomize() -> None:
        try:
            properties = ranges.sample(generator)
        except ValueError as exc:
            error_label.configure(text=str(exc))
            return
        set_values(properties)

    def accept() -> None:
        nonlocal selected
        try:
            selected = PhysicalProperties(
                **{
                    spec.field_name: float(variables[spec.field_name].get())
                    for spec in _SLIDER_SPECS
                }
            )
        except ValueError as exc:
            error_label.configure(text=str(exc))
            return
        root.destroy()

    def cancel() -> None:
        nonlocal selected
        selected = None
        root.destroy()

    buttons = ttk.Frame(outer)
    buttons.grid(
        row=3 + len(_SLIDER_SPECS),
        column=0,
        columnspan=4,
        sticky="ew",
        pady=(14, 0),
    )
    ttk.Button(buttons, text="Randomize", command=randomize).pack(side="left")
    ttk.Button(
        buttons,
        text="Reset defaults",
        command=lambda: set_values(DEFAULT_PHYSICAL_PROPERTIES),
    ).pack(side="left", padx=(8, 0))
    ttk.Button(buttons, text="Cancel", command=cancel).pack(side="right")
    run_button = ttk.Button(buttons, text="Run FlipUp", command=accept)
    run_button.pack(side="right", padx=(0, 8))

    root.protocol("WM_DELETE_WINDOW", cancel)
    root.bind("<Escape>", lambda _event: cancel())
    root.bind("<Return>", lambda _event: accept())
    run_button.focus_set()

    root.update_idletasks()
    x = max(0, (root.winfo_screenwidth() - root.winfo_reqwidth()) // 2)
    y = max(0, (root.winfo_screenheight() - root.winfo_reqheight()) // 3)
    root.geometry(f"+{x}+{y}")
    root.mainloop()
    return selected
