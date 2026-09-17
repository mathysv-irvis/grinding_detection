import tkinter as tk

from calibration_config import (
    DEFAULT_VALUES,
    load_values,
    save_values,
)


PARAMETERS = {
    "merge_distance": {
        "min": 0,
        "max": 50,
        "resolution": 1,
    },
    "intensity_min": {
        "min": 0,
        "max": 255,
        "resolution": 1,
    },
    "intensity_max": {
        "min": 0,
        "max": 255,
        "resolution": 1,
    },
}


class CalibrationUI:
    def __init__(self, root):
        self.root = root

        self.root.title("Calibration")
        self.root.resizable(False, False)

        self.values = load_values()
        self.variables = {}
        self.labels = {}

        self.create_ui()

    def create_ui(self):
        frame = tk.Frame(
            self.root,
            padx=12,
            pady=10,
        )

        frame.pack()

        for row, (name, config) in enumerate(PARAMETERS.items()):
            tk.Label(
                frame,
                text=name,
                width=18,
                anchor="w",
            ).grid(
                row=row,
                column=0,
                padx=(0, 8),
                pady=4,
            )

            variable = tk.IntVar(value=self.values[name])

            self.variables[name] = variable

            scale = tk.Scale(
                frame,
                variable=variable,
                from_=config["min"],
                to=config["max"],
                resolution=config["resolution"],
                orient=tk.HORIZONTAL,
                length=250,
                showvalue=False,
                command=lambda value, n=name: self.slider_changed(n, value),
            )

            scale.grid(
                row=row,
                column=1,
                pady=4,
            )

            value_label = tk.Label(
                frame,
                text=str(self.values[name]),
                width=7,
                anchor="e",
            )

            value_label.grid(
                row=row,
                column=2,
                padx=(8, 0),
            )

            self.labels[name] = value_label

        tk.Button(
            frame,
            text="Reset",
            command=self.reset,
            width=10,
        ).grid(
            row=len(PARAMETERS),
            column=0,
            columnspan=3,
            pady=(12, 0),
        )

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

        for name, value in self.values.items():
            self.variables[name].set(value)

            self.labels[name].config(text=str(value))

        save_values(self.values)


def main():
    root = tk.Tk()

    CalibrationUI(root)

    root.mainloop()


if __name__ == "__main__":
    main()
