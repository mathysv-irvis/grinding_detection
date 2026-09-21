import tkinter as tk
from tkinter import ttk

from calibration_config import (
    DEFAULT_VALUES,
    load_values,
    save_values,
)


PARAMETERS = {
    "plate_margin": {
        "label": "Plate margin",
        "min": 0,
        "max": 200,
        "resolution": 1,
    },
    "intensity_min": {
        "label": "Intensity min",
        "min": 0,
        "max": 255,
        "resolution": 1,
    },
    "intensity_max": {
        "label": "Intensity max",
        "min": 0,
        "max": 255,
        "resolution": 1,
    },
    "merge_distance": {
        "label": "Merge distance",
        "min": 0,
        "max": 50,
        "resolution": 1,
    },
    "contrast_threshold": {
        "label": "Contrast threshold",
        "min": 0,
        "max": 100,
        "resolution": 1,
    },
}


class CalibrationUI:
    def __init__(
        self,
        root,
    ):
        self.root = root

        self.root.title("Detection Calibration")

        self.root.resizable(
            False,
            False,
        )

        self.values = load_values()

        self.variables = {}
        self.labels = {}

        self.create_ui()

    def create_ui(self):
        frame = tk.Frame(
            self.root,
            padx=15,
            pady=15,
        )

        frame.pack()

        title = tk.Label(
            frame,
            text="Plate / Paint Calibration",
            font=(
                "TkDefaultFont",
                11,
                "bold",
            ),
        )

        title.grid(
            row=0,
            column=0,
            columnspan=3,
            pady=(0, 15),
        )

        mode_label = tk.Label(
            frame,
            text="Detection mode",
            width=18,
            anchor="w",
        )

        mode_label.grid(
            row=1,
            column=0,
            padx=(0, 10),
            pady=5,
        )

        self.mode_variable = tk.StringVar(
            value=self.values.get(
                "detection_mode",
                "intensity",
            )
        )

        mode_combo = ttk.Combobox(
            frame,
            textvariable=(self.mode_variable),
            values=[
                "intensity",
                "contrast",
            ],
            state="readonly",
            width=27,
        )

        mode_combo.grid(
            row=1,
            column=1,
            columnspan=2,
            pady=5,
        )

        mode_combo.bind(
            "<<ComboboxSelected>>",
            self.mode_changed,
        )

        for row, (
            name,
            config,
        ) in enumerate(
            PARAMETERS.items(),
            start=2,
        ):
            label = tk.Label(
                frame,
                text=config["label"],
                width=18,
                anchor="w",
            )

            label.grid(
                row=row,
                column=0,
                padx=(0, 10),
                pady=5,
            )

            variable = tk.IntVar(value=int(self.values[name]))

            self.variables[name] = variable

            scale = tk.Scale(
                frame,
                variable=variable,
                from_=config["min"],
                to=config["max"],
                resolution=config["resolution"],
                orient=tk.HORIZONTAL,
                length=280,
                showvalue=False,
                highlightthickness=0,
                command=(
                    lambda value, parameter=name: self.slider_changed(
                        parameter,
                        value,
                    )
                ),
            )

            scale.grid(
                row=row,
                column=1,
                pady=5,
            )

            value_label = tk.Label(
                frame,
                text=str(self.values[name]),
                width=6,
                anchor="e",
            )

            value_label.grid(
                row=row,
                column=2,
                padx=(10, 0),
            )

            self.labels[name] = value_label

        button_frame = tk.Frame(frame)

        button_frame.grid(
            row=len(PARAMETERS) + 2,
            column=0,
            columnspan=3,
            pady=(15, 0),
        )

        reset_button = tk.Button(
            button_frame,
            text="Reset",
            width=12,
            command=self.reset,
        )

        reset_button.pack(
            side=tk.LEFT,
            padx=5,
        )

        close_button = tk.Button(
            button_frame,
            text="Close",
            width=12,
            command=self.root.destroy,
        )

        close_button.pack(
            side=tk.LEFT,
            padx=5,
        )

    def mode_changed(
        self,
        event=None,
    ):
        self.values["detection_mode"] = self.mode_variable.get()

        save_values(self.values)

    def slider_changed(
        self,
        name,
        value,
    ):
        value = int(float(value))

        self.values[name] = value

        self.labels[name].config(text=str(value))

        save_values(self.values)

    def reset(self):
        self.values = DEFAULT_VALUES.copy()

        self.mode_variable.set(self.values["detection_mode"])

        for (
            name,
            value,
        ) in self.values.items():
            if name not in self.variables:
                continue

            self.variables[name].set(value)

            self.labels[name].config(text=str(value))

        save_values(self.values)


def main():
    root = tk.Tk()

    CalibrationUI(root)

    root.mainloop()


if __name__ == "__main__":
    main()
