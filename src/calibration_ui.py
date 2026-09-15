import tkinter as tk

from calibration_config import load_values, save_values


PARAMETERS = {
    "intensity": {
        "min": 0,
        "max": 255,
        "resolution": 1,
    },
    "K": {
        "min": 2,
        "max": 10,
        "resolution": 1,
    },
    "max_component": {
        "min": 100,
        "max": 10000,
        "resolution": 100,
    },
    "threshold": {
        "min": 0,
        "max": 255,
        "resolution": 1,
    },
    "kernel_size": {
        "min": 1,
        "max": 31,
        "resolution": 2,
    },
}


class CalibrationUI:
    def __init__(self, root):
        self.root = root

        self.root.title("Calibration")
        self.root.resizable(False, False)

        self.values = load_values()
        self.variables = {}

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
                width=16,
                anchor="w",
            ).grid(
                row=row,
                column=0,
                padx=(0, 8),
                pady=3,
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
                length=220,
                showvalue=False,
                command=lambda value, n=name: self.slider_changed(
                    n,
                    value,
                ),
            )

            scale.grid(
                row=row,
                column=1,
                pady=3,
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

            setattr(
                self,
                f"{name}_label",
                value_label,
            )

        tk.Button(
            frame,
            text="Reset",
            command=self.reset,
            width=10,
        ).grid(
            row=len(PARAMETERS),
            column=0,
            columnspan=3,
            pady=(10, 0),
        )

    def slider_changed(self, name, value):
        value = int(float(value))

        if name == "kernel_size":
            if value % 2 == 0:
                value += 1

            value = min(value, 31)

            self.variables[name].set(value)

        self.values[name] = value

        label = getattr(
            self,
            f"{name}_label",
        )

        label.config(text=str(value))

        save_values(self.values)

    def reset(self):
        self.values = load_values()

        for name, value in self.values.items():
            if name not in self.variables:
                continue

            self.variables[name].set(value)

            label = getattr(
                self,
                f"{name}_label",
            )

            label.config(text=str(value))


def main():
    root = tk.Tk()

    CalibrationUI(root)

    root.mainloop()


if __name__ == "__main__":
    main()
