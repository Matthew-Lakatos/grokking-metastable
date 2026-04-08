import csv, os, time
class CSVLogger:
    def __init__(self, path, header):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self.header = header
        with open(self.path, 'w', newline='') as f:
            writer = csv.writer(f); writer.writerow(header)
    def log(self, row):
        with open(self.path, 'a', newline='') as f:
            writer = csv.writer(f); writer.writerow(row)
