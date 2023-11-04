from random import randrange
import pandas as pd
import numpy as np

#abstract class element with next and prev, max_iter, and current
class Element:
    def __init__(self):
        self.ts = None
        self.current = 0
        self.max_iter = 0

    def next(self):
        print(NotImplementedError("curr not implemented"))
        return None

    def prev(self):
        print(NotImplementedError("curr not implemented"))
        return None

    def set_random_starting_pos(self, idx):
        print(NotImplementedError("curr not implemented"))
        return None

    def curr(self):
        print(NotImplementedError("curr not implemented"))
        return None

class LoadStream(Element):
    def __init__(self, ts_path, index_col=[0]) -> None:
        super(LoadStream).__init__()
        self.ts = pd.read_csv(ts_path, index_col=index_col)
        self.current = 0
        self.max_iter = len(self.ts)

    def next(self):
        if self.current == self.max_iter:
            self.current = 0
        else:
            self.current += 1
        out = self.ts.iloc[self.current].values.astype(np.float32)
        return out

    def prev(self):
        self.current -= 1
        out = self.ts.iloc[self.current].values.astype(np.float32)
        return out

    def set_random_starting_pos(self, episode_length):
        self.current = randrange(0, self.max_iter - (episode_length + 1), 96)
        # self.current = randrange(0, self.max_iter - (episode_length + 1), 1)

    def curr(self):
        out = self.ts.iloc[self.current].values.astype(np.float32)
        return out
