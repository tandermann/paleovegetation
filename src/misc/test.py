import numpy as np
import pandas as pd
#data_df = pd.read_clipboard()
data_df.head()
plt.plot(data_df.iloc[:,3],data_df.iloc[:,2],'.')

n_all_cells = 347
year_of_extinction = data_df[data_df['Presence (0) or Absence (1)']==1]['Sampling year'].values
years, counts = np.unique(year_of_extinction, return_counts = True)
total_extinct_cells = np.cumsum(counts)
plt.plot(years,(n_all_cells-total_extinct_cells)/n_all_cells)


plt.hist(year_of_extinction)

