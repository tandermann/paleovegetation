import numpy as np
import matplotlib.pyplot as plt
from pyrolite.util.time import Timescale

ts = Timescale()
ts.named_age(12)
ts.levels
ts.named_age(3, level="Epoch")


fig = plt.figure(figsize=(7, 1))
level='Epoch'
ldf = ts.data.loc[ts.data.Level == level, :]
ldf = ldf.iloc[:17,:]
for pix, period in ldf.iterrows():
    print(period.Age)
    col = period.Color
    if period.Epoch == 'Holocene':
       col = (254/255, 242/255, 236/255, 1.0)
    if period.Age == 'Chibanian':
        col = (255/255, 242/255, 199/255, 1.0)
    print(period.Color)
    plt.bar(
        np.mean([period.Start,period.End]),
        1,
        facecolor=col,
        #bottom=period.End,
        width=(period.Start - period.End),
        edgecolor="k",
        linewidth = 0.5
    )
plt.xlabel('Age (Ma)')
ax = plt.gca()
right_side = ax.spines["right"]
right_side.set_visible(False)
right_side = ax.spines["top"]
right_side.set_visible(False)
right_side = ax.spines["left"]
right_side.set_visible(False)
ax.get_yaxis().set_visible(False)
plt.xlim(-0.2,30.2)
plt.tight_layout()
ax.invert_xaxis()
fig.savefig('plots/geological_time_scale.pdf', transparent=True)

