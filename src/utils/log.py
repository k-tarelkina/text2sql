from datetime import datetime


class Logger:
    def __init__(self, mode="prediction"):
        id = datetime.now().strftime("%H-%M-%S")
        self.filename = f"{mode}_{id}.log"

    def write(self, message):
        with open(self.filename, "a") as f:
            date = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
            f.write(f"[{date}] {message} \n")
